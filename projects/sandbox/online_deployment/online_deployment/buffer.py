from typing import List

import torch

from ml4gw.utils.slicing import unfold_windows


class DataBuffer:
    def __init__(
        self,
        ifos: List[str],
        channel: str,
        kernel_length: float,
        sample_rate: float,
        inference_sampling_rate: float,
        integration_window_length: float,
        psd_length: float,
    ):
        self.ifos = ifos
        self.channel = channel
        self.sample_rate = sample_rate
        self.inference_sampling_rate = inference_sampling_rate
        self.psd_length = psd_length

        self.kernel_size = int(kernel_length * sample_rate)
        self.stride = int(sample_rate / inference_sampling_rate)
        self.integrator_size = int(
            integration_window_length * inference_sampling_rate
        )

        self.reset_states()
        self.window = torch.ones((1, 1, self.integrator_size), device="cuda")
        self.window /= self.integrator_size

        taper = torch.hann_window(int(2 * sample_rate))[: int(sample_rate)]
        self.taper = taper.to("cuda")

        self.event = None
        # TODO: de-magic
        self.buffer_length = self.psd_length + 8

    def reset_states(self):
        buffer_size = self.kernel_size + self.psd_length - self.stride
        self.input = torch.zeros((2, int(buffer_size)), device="cuda")
        self.output = torch.zeros((self.integrator_size,), device="cuda")
        self.t0 = None

    def integrate(self, x: torch.Tensor):
        x = x.view(1, 1, -1)
        y = torch.nn.functional.conv1d(x, self.window, padding="valid")
        return y[0, 0].cpu().numpy()

    def get_event_idx(self, offset, sample_rate):
        # TODO
        diff = self.t0 + 1 - self.event.time
        diff += offset
        idx = int(diff * sample_rate)

        # TODO
        half_length = (self.buffer_length - 3) / 2
        half_size = int(half_length * sample_rate)

        start = -idx - half_size
        stop = -idx + half_size
        return start, stop

    def get_state(self):
        start, stop = self.get_event_idx(1, self.sample_rate)
        X = self.input[:, start:stop]

        start, stop = self.get_event_idx(-0.5, self.inference_sampling_rate)
        y = self.output[start:stop]

        event = self.event
        self.event = None
        return X, y, event

    def finalize(self):
        if self.event is None or self.t0 is None:
            X = y = event = None
        elif self.event.time + 2 < self.t0:
            X, y, event = self.get_state()
        else:
            X = y = event = None

        keep_length = self.buffer_length - 1
        start = int(keep_length * self.sample_rate)
        self.input = self.input[:, -start:]

        start = int(keep_length * self.inference_sampling_rate)
        self.output = self.output[-start:]
        return X, y, event

    def make_batch(self, X, t0):
        if self.t0 is None:
            X *= self.taper
        self.t0 = t0

        self.input = torch.cat([self.input, X], axis=1)
        buffer_size = self.kernel_size + X.shape[-1] - self.stride
        X = self.input[:, -buffer_size:]
        return unfold_windows(X, self.kernel_size, self.stride)

    def update(self, y):
        self.output = torch.cat([self.output, y])

        buffer_size = self.integrator_size + len(y)
        y = self.output[-buffer_size:]
        return self.integrate(y)
