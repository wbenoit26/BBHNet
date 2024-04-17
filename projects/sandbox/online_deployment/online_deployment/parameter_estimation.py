def get_data_for_pe(
    event_time,
    input_buffer,
    fduration,
    pe_window=4,
    event_position=2.5
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
