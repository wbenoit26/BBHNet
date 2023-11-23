from typing import Optional

import torch

from aframe.architectures import BackgroundSnapshotter, BatchWhitener


class SnapshotWhitener(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        psd_length: float,
        kernel_length: float,
        fduration: float,
        sample_rate: float,
        inference_sampling_rate: float,
        fftlength: float,
        highpass: Optional[float] = None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.snapshotter = BackgroundSnapshotter(
            psd_length=psd_length,
            kernel_length=kernel_length,
            fduration=fduration,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
        ).to("cuda")

        # Updates come in 1 second chunks, so each
        # update will generate a batch of
        # `inference_sampling_rate` overlapping
        # windows to whiten
        batch_size = 1 * inference_sampling_rate
        self.batch_whitener = BatchWhitener(
            kernel_length=kernel_length,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
            batch_size=batch_size,
            fduration=fduration,
            fftlength=fftlength,
            highpass=highpass,
        ).to("cuda")

        self.step_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int(kernel_length * sample_rate)
        self.psd_size = int(psd_length * sample_rate)
        self.filter_size = int(fduration * sample_rate)

    @property
    def state_size(self):
        return (
            self.psd_size
            + self.kernel_size
            + self.filter_size
            - self.step_size
        )

    def get_initial_state(self):
        return torch.zeros((1, self.num_channels, self.state_size))

    def forward(self, update, current_state):
        update = update[None, :, :]
        X, current_state = self.snapshotter(update, current_state)
        return self.batch_whitener(X), current_state


class OutputBuffer:
    def __init__(
        self,
        inference_sampling_rate: float,
        integration_window_length: float,
    ):
        self.integrator_size = int(
            integration_window_length * inference_sampling_rate
        )
        self.window = torch.ones((1, 1, self.integrator_size), device="cuda")
        self.window /= self.integrator_size
        self.reset_state()

    def reset_state(self):
        self.output = torch.zeros((self.integrator_size,), device="cuda")
        self.t0 = None

    def integrate(self, x: torch.Tensor):
        x = x.view(1, 1, -1)
        y = torch.nn.functional.conv1d(x, self.window, padding="valid")
        return y[0, 0].cpu().numpy()

    def update(self, y):
        self.output = torch.cat([self.output, y])

        buffer_size = self.integrator_size + len(y)
        y = self.output[-buffer_size:]
        return self.integrate(y)
