import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# code from https://github.com/passalis/dain
class DAIN_Layer(nn.Module):
    def __init__(
        self,
        input_dim=144,
        mode="full",
        mean_lr=0.00001,
        gate_lr=0.001,
        scale_lr=0.00001,
    ):
        super(DAIN_Layer, self).__init__()
        print("Mode = ", mode)

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(
            data=np.eye(input_dim, input_dim)
        )

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(
            data=np.eye(input_dim, input_dim)
        )

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)
        # Step 1:
        avg = torch.mean(x, 2)
        adaptive_avg = self.mean_layer(avg)
        adaptive_avg = adaptive_avg.resize(
            adaptive_avg.size(0), adaptive_avg.size(1), 1
        )
        x = x - adaptive_avg

        # # Step 2:
        std = torch.mean(x**2, 2)
        std = torch.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std)
        adaptive_std[adaptive_std <= self.eps] = 1

        adaptive_std = adaptive_std.resize(
            adaptive_std.size(0), adaptive_std.size(1), 1
        )
        x = x / adaptive_std

        # Step 3:
        avg = torch.mean(x, 2)
        gate = torch.sigmoid(self.gating_layer(avg))
        gate = gate.resize(gate.size(0), gate.size(1), 1)
        x = x * gate
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.x_transform = nn.Identity()

        self.body = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        x = F.relu(self.body(x) + self.x_transform(x))
        return x


class ResNet54(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 16, stride=2),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 32, stride=2),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)


class ResNet54Double(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 32, stride=2),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128, stride=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(32, 64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)[:, 0].unsqueeze(1)


class AresGW(torch.nn.Module):
    def __init__(self, num_ifos):
        super().__init__()
        self.norm = DAIN_Layer(input_dim=num_ifos)
        self.resnet = ResNet54Double()
        self.network = torch.nn.Sequential(self.norm, self.resnet)

    def forward(self, x):
        return self.network(x)
