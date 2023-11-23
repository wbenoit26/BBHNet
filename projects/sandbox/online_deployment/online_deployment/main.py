import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from online_deployment.buffer import OutputBuffer, SnapshotWhitener
from online_deployment.dataloading import data_iterator
from online_deployment.trigger import Searcher, Trigger

from aframe.architectures import architecturize
from aframe.logging import configure_logging


@architecturize
@torch.no_grad()
def main(
    architecture: Callable,
    outdir: Path,
    datadir: Path,
    ifos: List[str],
    channel: str,
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    psd_length: float,
    fduration: float,
    integration_window_length: float,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    refractory_period: float = 8,
    far_per_day: float = 1,
    secondary_far_threshold: float = 24,
    verbose: bool = False,
):
    logdir = outdir / "log"
    logdir.mkdir(exist_ok=True, parents=True)
    configure_logging(outdir / "log" / "deploy.log", verbose)

    buffer = OutputBuffer(inference_sampling_rate, integration_window_length)

    # instantiate network and load in its optimized parameters
    weights_path = outdir / "training" / "weights.pt"
    logging.info(f"Build network and loading weights from {weights_path}")

    num_ifos = len(ifos)
    nn = architecture(num_ifos).to("cuda")
    fftlength = fftlength or kernel_length + fduration
    whitener = SnapshotWhitener(
        num_channels=num_ifos,
        psd_length=psd_length,
        kernel_length=kernel_length,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
        fftlength=fftlength,
        highpass=highpass,
    )
    current_state = whitener.get_initial_state().to("cuda")

    weights = torch.load(weights_path)
    nn.load_state_dict(weights)
    nn.eval()

    # set up some objects to use for finding
    # and submitting triggers
    fars = [far_per_day, secondary_far_threshold]
    searcher = Searcher(
        outdir, fars, inference_sampling_rate, refractory_period
    )

    triggers = [
        Trigger(outdir / "triggers"),
        Trigger(outdir / "secondary-triggers"),
    ]
    in_spec = False

    def get_trigger(event):
        fars_hz = [i / 3600 / 24 for i in fars]
        idx = np.digitize(event.far, fars_hz)
        if idx == 0 and not in_spec:
            logging.warning(
                "Not submitting event {} to production trigger "
                "because data is not analysis ready".format(event)
            )
            idx = 1
        return triggers[idx]

    def write_state(X, y, event):
        # TODO: implement this via overlap add
        # More broadly, where should this live?
        window_length = X.shape[1] / sample_rate / 2
        t0 = event.time - window_length + fduration / 2
        whitened = []
        for i in range(4):
            start = int(i * sample_rate)
            stop = start + int(sample_rate * kernel_length)
            x = X[None, :, start:stop]
            x = whitener(x)
            whitened.append(x[0])
        whitened = torch.cat(whitened, axis=1).cpu().numpy()

        state = TimeSeriesDict()
        for i, ifo in enumerate(ifos):
            chan = f"{ifo}:{channel}"
            ts = TimeSeries(
                whitened[i], sample_rate=sample_rate, t0=t0, channel=chan
            )
            state[chan] = ts

        y = y.cpu().numpy()
        pad = int(fduration / 2 * inference_sampling_rate)
        y = y[pad:-pad]
        state["AFRAME"] = TimeSeries(
            y, sample_rate=inference_sampling_rate, t0=t0, channel="AFRAME"
        )

        trigger = get_trigger(event)
        timestamp = int(event.time)
        fname = f"event-{timestamp}.h5"
        state.write(trigger.gdb.write_dir / fname)

    # offset the initial timestamp of our
    # integrated outputs relative to the
    # initial timestamp of the most recently
    # loaded frames
    time_offset = (
        1 / inference_sampling_rate  # end of the first kernel in batch
        - fduration / 2  # account for whitening padding
        - integration_window_length  # account for time to build peak
    )

    logging.info("Beginning search")
    data_it = data_iterator(
        datadir,
        channel,
        ifos,
        sample_rate,
        timeout=10,
    )
    integrated = None  # need this for static linters
    for X, t0, ready in data_it:
        # adjust t0 to represent the timestamp of the
        # leading edge of the input to the network
        if not ready:
            in_spec = False

            # if we had an event in the last frame, we
            # won't get to see its peak, so do our best
            # to build the event with what we have
            if searcher.detecting and not searcher.check_refractory:
                event = searcher.build_event(
                    integrated[-1], t0 - 1, len(integrated) - 1
                )
                trigger = get_trigger(event)
                trigger.submit(event, ifos)
                searcher.detecting = False

            # check if this is because the frame stream stopped
            # being analysis ready, or if it's because frames
            # were dropped within the stream
            if X is not None:
                logging.warning(
                    "Frame {} is not analysis ready. Performing "
                    "inference but ignoring triggers".format(t0)
                )
            else:
                logging.warning(
                    "Missing frame files after timestep {}, "
                    "resetting states".format(t0)
                )

                # first check if there's an event
                # we need to write, even if we don't
                # have all of its future data
                # X, y, event = buffer.finalize()
                # if event is not None:
                #     write_state(X, y, event)
                # buffer.reset_states()
                continue
        elif not in_spec:
            # the frame is analysis ready, but previous frames
            # weren't, so reset our running states and taper
            # the data in to not add frequency artifacts
            logging.info(f"Frame {t0} is ready again, resetting states")
            buffer.reset_state()
            in_spec = True

        X = X.to("cuda")
        batch, current_state = whitener(X, current_state)
        y = nn(batch)[:, 0]
        integrated = buffer.update(y)

        event = searcher.search(integrated, t0 + time_offset)
        if event is not None:
            trigger = get_trigger(event)
            trigger.submit(event, ifos)
            # buffer.event = event

        # slough off old data from our buffer states,
        # but save the states around an event if the
        # buffer has one associated with it
        # X, y, event = buffer.finalize()
        # if event is not None:
        #     write_state(X, y, event)
