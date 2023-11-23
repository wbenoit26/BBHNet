import torch


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

    def integrate(self, x: torch.Tensor):
        x = x.view(1, 1, -1)
        y = torch.nn.functional.conv1d(x, self.window, padding="valid")
        return y[0, 0].cpu().numpy()

    def update(self, y):
        self.output = torch.cat([self.output, y])

        buffer_size = self.integrator_size + len(y)
        y = self.output[-buffer_size:]
        return self.integrate(y)
