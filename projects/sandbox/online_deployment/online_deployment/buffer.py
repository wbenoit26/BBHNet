import torch
from gwpy.timeseries import TimeSeries, TimeSeriesDict


class OutputBuffer:
    def __init__(
        self,
        inference_sampling_rate: float,
        integration_window_length: float,
        buffer_length: float,
    ):
        self.inference_sampling_rate = inference_sampling_rate
        self.integrator_size = int(
            integration_window_length * inference_sampling_rate
        )
        self.window = torch.ones((1, 1, self.integrator_size), device="cuda")
        self.window /= self.integrator_size
        self.buffer_length = buffer_length
        self.buffer_size = int(buffer_length * inference_sampling_rate)
        self.reset_state()

    def reset_state(self):
        self.output_buffer = torch.zeros((self.buffer_size,), device="cuda")
        self.integrated_buffer = torch.zeros(
            (self.buffer_size,), device="cuda"
        )

    def write(self, write_path):
        buffer = TimeSeriesDict()
        buffer["output"] = TimeSeries(
            self.output_buffer,
            sample_rate=self.inference_sampling_rate,
            t0=self.t0,
            channel="output",
        )
        buffer["integrated"] = TimeSeries(
            self.integrated_buffer,
            sample_rate=self.inference_sampling_rate,
            t0=self.t0,
            channel="output",
        )
        buffer.write(write_path)

    def integrate(self, x: torch.Tensor):
        x = x.view(1, 1, -1)
        y = torch.nn.functional.conv1d(x, self.window, padding="valid")
        return y[0, 0]

    def update(self, y, t0):
        self.output_buffer = torch.cat([self.output_buffer, y])
        self.output_buffer = self.output_buffer[-self.buffer_size :]
        # t0 corresponds to the time of the first sample in the update
        # self.t0 corresponds to the earliest time in the buffer
        update_duration = len(y) / self.inference_sampling_rate
        self.t0 = t0 - (self.buffer_length - update_duration)

        integration_size = self.integrator_size + len(y)
        y = self.output_buffer[-integration_size:]
        integrated = self.integrate(y)
        self.integrated_buffer = torch.cat(
            [self.integrated_buffer, integrated]
        )
        self.integrated_buffer = self.integrated_buffer[-self.buffer_size :]
        return integrated.cpu().numpy()
