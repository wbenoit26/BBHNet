import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional
from zlib import adler32

import datagen.utils.timeslide_waveforms as utils
import numpy as np
import torch
from datagen.utils.injection import generate_gw
from typeo import scriptify

from aframe.analysis.ledger.injections import (
    InjectionParameterSet,
    LigoResponseSet,
)
from aframe.deploy import condor
from aframe.logging import configure_logging
from ml4gw.gw import (
    compute_network_snr,
    compute_observed_strain,
    get_ifo_geometry,
)


@scriptify
def main(
    start: float,
    stop: float,
    ifos: List[str],
    shifts: List[float],
    background_dir: Path,
    spacing: float,
    buffer: float,
    prior: Callable,
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_duration: float,
    waveform_approximant: str,
    highpass: float,
    snr_threshold: float,
    output_dir: Path,
    log_file: Optional[Path] = None,
    verbose: bool = False,
    seed: Optional[int] = None,
):
    """
    Generates the waveforms for a single segment.

    Args:
        start:
            GPS time of the beginning of the testing segment
        stop:
            GPS time of the end of the testing segment
        ifos:
            List of interferometers to query data from. Expected to be given
            by prefix; e.g. "H1" for Hanford. Should be the same length as
            `shifts`
        shifts:
            The length of time in seconds by which each interferometer's
            timeseries will be shifted
        background_dir:
            Directory containing background data to use for PSD
            calculation. Should have data for each interferometer in
            the format generated by `background.py`
        spacing:
            The amount of time, in seconds, to leave between the end
            of one signal and the start of the next
        buffer:
            The amount of time, in seconds, on either side of the
            segment within which injection times will not be
            generated
        prior:
            A function that returns a Bilby PriorDict when called
        minimum_frequency:
            Minimum frequency of the gravitational wave. The part
            of the gravitational wave at lower frequencies will
            not be generated. Specified in Hz.
        reference_frequency:
            Frequency of the gravitational wave at the state of
            the merger that other quantities are defined with
            reference to
        sample_rate:
            Sample rate of timeseries data, specified in Hz
        waveform_duration:
            Duration of waveform in seconds
        waveform_approximant:
            Name of the waveform approximant to use.
        highpass:
            The frequency to use for a highpass filter, specified
            in Hz
        snr_threshold:
            Minimum SNR of generated waveforms. Sampled parameters
            that result in an SNR below this threshold will be rejected,
            but saved for later use
        output_dir:
            Directory to which the waveform file and rejected parameter
            file will be written
        log_file:
            File containing the logged information
        verbose:
            If True, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.

    Returns:
        The name of the waveform file and the name of the file containing the
        rejected parameters
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file, verbose=verbose)

    if seed is not None:
        fingerprint = str((start, stop) + tuple(shifts))
        worker_hash = adler32(fingerprint.encode())
        logging.info(
            "Seeding data generation with seed {}, "
            "augmented by worker seed {}".format(seed, worker_hash)
        )
        np.random.seed(seed + worker_hash)
        random.seed(seed + worker_hash)

    prior, detector_frame_prior = prior()

    injection_times = utils.calc_segment_injection_times(
        start,
        stop - max(shifts),  # TODO: should account for uneven last batch too
        spacing,
        buffer,
        waveform_duration,
    )
    n_samples = len(injection_times)
    waveform_size = int(sample_rate * waveform_duration)

    zeros = np.zeros((n_samples,))
    parameters = defaultdict(lambda: zeros.copy())
    parameters["gps_time"] = injection_times
    parameters["shift"] = np.array([shifts for _ in range(n_samples)])

    for ifo in ifos:
        empty = np.zeros((n_samples, waveform_size))
        parameters[ifo.lower()] = empty

    tensors, vertices = get_ifo_geometry(*ifos)
    df = 1 / waveform_duration
    try:
        background_path = sorted(background_dir.iterdir())[0]
    except StopIteration:
        raise ValueError(
            f"No files in background data directory {background_dir}"
        )
    logging.info(
        f"Using background file {background_path} for psd calculation"
    )
    psds = utils.load_psds(background_path, ifos, df=df)

    # loop until we've generated enough signals
    # with large enough snr to fill the segment,
    # keeping track of the number of signals rejected
    num_injections, idx = 0, 0
    rejected_params = InjectionParameterSet()
    while n_samples > 0:
        params = prior.sample(n_samples)

        # If a Bilby PriorDict has a conversion function, any
        # extra keys generated by the conversion function will
        # be added into the sampled parameters, but only if
        # we're sampling exactly 1 time. This removes those
        # extra keys
        # TODO: If https://git.ligo.org/lscsoft/bilby/-/merge_requests/1286
        # is merged, remove this
        if n_samples == 1:
            params = {k: params[k] for k in parameters if k in params}

        waveforms = generate_gw(
            params,
            minimum_frequency,
            reference_frequency,
            sample_rate,
            waveform_duration,
            waveform_approximant,
            detector_frame_prior,
        )
        polarizations = {
            "cross": torch.Tensor(waveforms[:, 0, :]),
            "plus": torch.Tensor(waveforms[:, 1, :]),
        }

        projected = compute_observed_strain(
            torch.Tensor(params["dec"]),
            torch.Tensor(params["psi"]),
            torch.Tensor(params["ra"]),
            tensors,
            vertices,
            sample_rate,
            **polarizations,
        )
        # TODO: compute individual ifo snr so we can store that data
        snrs = compute_network_snr(projected, psds, sample_rate, highpass)
        snrs = snrs.numpy()

        # add all snrs: masking will take place in for loop below
        params["snr"] = snrs
        num_injections += len(snrs)
        mask = snrs >= snr_threshold

        # first record any parameters that were
        # rejected during sampling to a separate object
        rejected = {}
        for key in InjectionParameterSet.__dataclass_fields__:
            rejected[key] = params[key][~mask]
        rejected = InjectionParameterSet(**rejected)
        rejected_params.append(rejected)

        # if nothing got accepted, try again
        num_accepted = mask.sum()
        if num_accepted == 0:
            continue

        # insert our accepted parameters into the output array
        start, stop = idx, idx + num_accepted
        for key, value in params.items():
            parameters[key][start:stop] = value[mask]

        # do the same for our accepted projected waveforms
        projected = projected[mask].numpy()
        for i, ifo in enumerate(ifos):
            key = ifo.lower()
            parameters[key][start:stop] = projected[:, i]

        # subtract off the number of samples we accepted
        # from the number we'll need to sample next time,
        # that way we never overshoot our number of desired
        # accepted samples and therefore risk overestimating
        # our total number of injections
        idx += num_accepted
        n_samples -= num_accepted

    parameters["sample_rate"] = sample_rate
    parameters["duration"] = waveform_duration
    parameters["num_injections"] = num_injections

    response_set = LigoResponseSet(**parameters)
    waveform_fname = output_dir / "bns_waveforms.h5"
    utils.io_with_blocking(response_set.write, waveform_fname)

    rejected_fname = output_dir / "bns-rejected-parameters.h5"
    utils.io_with_blocking(rejected_params.write, rejected_fname)

    # TODO: compute probability of all parameters against
    # source and all target priors here then save them somehow
    return waveform_fname, rejected_fname


# until typeo update gets in just take all the same parameter as main
@scriptify
def deploy(
    start: float,
    stop: float,
    Tb: float,
    ifos: List[str],
    shifts: Iterable[float],
    spacing: float,
    buffer: float,
    min_segment_length: float,
    prior: str,
    minimum_frequency: float,
    reference_frequency: float,
    sample_rate: float,
    waveform_duration: float,
    waveform_approximant: str,
    highpass: float,
    snr_threshold: float,
    psd_length: float,
    outdir: Path,
    datadir: Path,
    logdir: Path,
    accounting_group_user: str,
    accounting_group: str,
    request_memory: int = 6000,
    request_disk: int = 1024,
    force_generation: bool = False,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> None:
    """
    Deploy condor jobs to generate waveforms for all segments

    Args:
        start:
            GPS time of the beginning of the testing dataset
        stop:
            GPS time of the end of the testing dataset
        Tb:
            The length of background time in seconds to be generated via
            time shifts
         ifos:
            List of interferometers to query data from. Expected to be given
            by prefix; e.g. "H1" for Hanford. Should be the same length as
            `shifts`
        shifts:
            A list of shifts in seconds. Each value corresponds to the
            the length of time by which an interferometer's timeseries is
            moved during one shift. For example, if `ifos = ["H1", "L1"]`
            and `shifts = [0, 1]`, then the Livingston timeseries will be
            advanced by one second per shift, and Hanford won't be shifted
        spacing:
            The amount of time, in seconds, to leave between the end
            of one signal and the start of the next
        buffer:
            The amount of time, in seconds, on either side of the
            segment within which injection times will not be
            generated
        min_segment_length:
            The shortest a contiguous segment of background can be.
            Specified in seconds
        prior:
            A function that returns a Bilby PriorDict when called
        minimum_frequency:
            Minimum frequency of the gravitational wave. The part
            of the gravitational wave at lower frequencies will
            not be generated. Specified in Hz.
        reference_frequency:
            Frequency of the gravitational wave at the state of
            the merger that other quantities are defined with
            reference to
        sample_rate:
            Sample rate of timeseries data, specified in Hz
        waveform_duration:
            Duration of waveform in seconds
        waveform_approximant:
            Name of the waveform approximant to use.
        highpass:
            The frequency to use for a highpass filter, specified
            in Hz
        snr_threshold:
            Minimum SNR of generated waveforms. Sampled parameters
            that result in an SNR below this threshold will be rejected,
            but saved for later use
        psd_length:
            Length of background in seconds to use for PSD calculation
        outdir:
            Directory to which the condor files will be written
        datadir:
            Directory to which the waveform dataset and rejected parameters
            will be written
        logdir:
            Directory to which the log file will be written
        accounting_group_user:
            Username of the person running the condor jobs
        accounting_group:
            Accounting group for the condor jobs
        request_memory:
            Amount of memory for condor jobs to request in Mb
        request_disk:
            Amount of disk space for condor jobs to request in Mb
        force_generation:
            If False, will not generate data if an existing dataset exists
        verbose:
            If True, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.
    """
    # define some directories:
    # outdir: where we'll write the temporary
    #    files created in each condor job
    # writedir: where we'll write the aggregated
    #    aggregated outputs from each of the
    #    temporary files in outdir
    # condordir: where we'll write the submit file,
    #    queue parameters file, and the log, out, and
    #    err files from each submitted job
    outdir = outdir / "timeslide_waveforms"
    writedir = datadir / "test"
    condordir = outdir / "condor"
    for d in [outdir, writedir, condordir, logdir]:
        d.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "timeslide_waveforms.log", verbose=verbose)

    # check to see if any of the target files are
    # missing or if we've indicated to force
    # generation even if they are
    for fname in ["bns_waveforms.h5", "bns-rejected-parameters.h5"]:
        if not (writedir / fname).exists() or force_generation:
            break
    else:
        # if everything exists and we're not forcing
        # generation, short-circuit here
        logging.info(
            "Timeslide waveform and rejected parameters files "
            "already exist in {} and force_generation is off, "
            "exiting".format(writedir)
        )
        return

    # parse relevant segments based on files in background directory
    segments = utils.segments_from_directory(datadir / "test" / "background")
    shifts_required = utils.get_num_shifts(segments, Tb, max(shifts))

    # create text file from which the condor job will read
    # the start, stop, and shift for each job
    parameters = "start,stop,shift\n"
    for start, stop in segments:
        for i in range(shifts_required):
            # TODO: make this more general
            shift = [(i + 1) * shift for shift in shifts]
            shift = " ".join(map(str, shift))
            # add psd_length to account for the burn in of psd calculation
            parameters += f"{start + psd_length},{stop},{shift}\n"

    # TODO: have typeo do this CLI argument construction?
    arguments = "--start $(start) --stop $(stop) --shifts $(shift) "
    arguments += f"--background-dir {datadir / 'train' / 'background'} "
    arguments += f"--spacing {spacing} --buffer {buffer} "
    arguments += f"--minimum-frequency {minimum_frequency} "
    arguments += f"--reference-frequency {reference_frequency} "
    arguments += f"--sample-rate {sample_rate} "
    arguments += f"--waveform-duration {waveform_duration} "
    arguments += f"--waveform-approximant {waveform_approximant} "
    arguments += f"--highpass {highpass} --snr-threshold {snr_threshold} "
    arguments += f"--ifos {' '.join(ifos)} "
    arguments += f"--prior {prior} "
    arguments += f"--output-dir {outdir}/tmp-$(ProcID) "
    arguments += f"--log-file {logdir}/$(ProcID).log "
    if seed:
        arguments += f"--seed {seed} "

    # create submit file by hand: pycondor doesn't support
    # "queue ... from" syntax
    subfile = condor.make_submit_file(
        executable="generate-timeslide-waveforms",
        name="timeslide_waveforms",
        parameters=parameters,
        arguments=arguments,
        submit_dir=condordir,
        accounting_group=accounting_group,
        accounting_group_user=accounting_group_user,
        clear=True,
        request_memory=request_memory,
        request_disk=request_disk,
        stream_output=True,
        stream_error=True,
    )
    dag_id = condor.submit(subfile)
    logging.info(f"Launching waveform generation jobs with dag id {dag_id}")
    condor.watch(dag_id, condordir)

    # once all jobs are done, merge the output files
    waveform_fname = writedir / "bns_waveforms.h5"
    waveform_files = list(outdir.rglob("bns_waveforms.h5"))
    logging.info(f"Merging output waveforms to file {waveform_fname}")
    LigoResponseSet.aggregate(waveform_files, waveform_fname, clean=True)

    params_fname = writedir / "bns-rejected-parameters.h5"
    param_files = list(outdir.rglob("bns-rejected-parameters.h5"))
    logging.info(f"Merging rejected parameters to file {params_fname}")
    InjectionParameterSet.aggregate(param_files, params_fname, clean=True)

    for dirname in outdir.glob("tmp-*"):
        shutil.rmtree(dirname)

    logging.info("Timeslide waveform generation complete")
