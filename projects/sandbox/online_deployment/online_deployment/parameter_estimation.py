import numpy as np
import healpy as hp
import bilby
import pandas as pd

def get_data_for_pe(
    event_time,
    input_buffer,
    fduration,
    pe_window=4,
    event_position=3
):
    buffer_start = input_buffer.t0
    sample_rate = input_buffer.sample_rate
    data = input_buffer.input_buffer

    window_start = event_time - buffer_start - event_position - fduration / 2
    window_start = int(sample_rate * window_start)
    window_end = int(window_start + (pe_window + fduration / 2) * sample_rate)

    psd_data = data[:, :window_start]
    pe_data = data[:, window_start : window_end]

    return psd_data, pe_data

def submit_pe():
    raise NotImplementedError

def cast_samples_as_bilby_result(
    samples,
    inference_params,
    label,
):
    """Cast posterior samples as bilby Result object"""
    posterior = dict()
    for idx, k in enumerate(inference_params):
        posterior[k] = samples.T[idx].flatten()
    posterior = pd.DataFrame(posterior)
    return bilby.result.Result(
        label=label,
        posterior=posterior,
        search_parameter_keys=inference_params,
    )

def plot_mollview(
    ra_samples: np.ndarray,
    dec_samples: np.ndarray,
    nside: int = 32,
    fig = None,
    title = None,
):
    # mask out non physical samples;
    ra_samples_mask = (ra_samples > -np.pi) * (ra_samples < np.pi)
    dec_samples_mask = (dec_samples > 0) * (dec_samples < np.pi)

    net_mask = ra_samples_mask * dec_samples_mask
    ra_samples = ra_samples[net_mask]
    dec_samples = dec_samples[net_mask]

    # calculate number of samples in each pixel
    NPIX = hp.nside2npix(nside)
    ipix = hp.ang2pix(nside, dec_samples, ra_samples)
    ipix = np.sort(ipix)
    uniq, counts = np.unique(ipix, return_counts=True)

    # create empty map and then fill in non-zero pix with counts
    m = np.zeros(NPIX)
    m[np.in1d(range(NPIX), uniq)] = counts

    fig = hp.mollview(m, fig=fig, title=title, hold=True)
    return fig