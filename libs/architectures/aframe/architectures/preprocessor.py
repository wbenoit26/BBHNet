from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram

from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.utils.slicing import unfold_windows


class BackgroundSnapshotter(torch.nn.Module):
    """Update a kernel with a new piece of streaming data"""

    def __init__(
        self,
        psd_length,
        kernel_length,
        fduration,
        sample_rate,
        inference_sampling_rate,
    ) -> None:
        super().__init__()
        state_length = kernel_length + fduration + psd_length
        state_length -= 1 / inference_sampling_rate
        self.state_size = int(state_length * sample_rate)

    def forward(
        self, update: torch.Tensor, snapshot: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        x = torch.cat([snapshot, update], axis=-1)
        snapshot = x[:, :, -self.state_size :]
        return x, snapshot


class PsdEstimator(torch.nn.Module):
    """
    Module that takes a sample of data, splits it into
    a `background_length`-second section of data and the,
    remainder, calculates the PSD of the first section,
    and returns the PSD and the remainder.
    """

    def __init__(
        self,
        length: float,
        sample_rate: float,
        fftlength: float,
        overlap: Optional[float] = None,
        average: str = "mean",
        fast: bool = True,
    ) -> None:
        super().__init__()
        self.size = int(length * sample_rate)
        self.spectral_density = SpectralDensity(
            sample_rate, fftlength, overlap, average, fast=fast
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        splits = [X.size(-1) - self.size, self.size]
        background, X = torch.split(X, splits, dim=-1)
        psds = self.spectral_density(background.double())
        return X, psds


class MultiResolutionSpectrogram(torch.nn.Module):
    """
    Create a batch of multi-resolution spectrograms
    from a batch of timeseries. Input is expected to
    have the shape `(B, C, T)`, where `B` is the number
    of batches, `C` is the number of channels, and `T`
    is the number of time samples.

    For each timeseries, calculate multiple normalized
    spectrograms based on the `Spectrogram` `kwargs` given.
    Combine the spectrograms by taking the maximum value
    from the nearest time-frequncy bin.

    If the largest number of time bins among the spectrograms
    is `N` and the largest number of frequency bins is `M`,
    the output will have dimensions `(B, C, M, N)`

    Args:
        kernel_length:
            The length in seconds of the time dimension
            of the tensor that will be turned into a
            spectrogram
        sample_rate:
            The sample rate of the timeseries in Hz
        kwargs:
            Arguments passed in kwargs will used to create
            `torchaudio.transforms.Spectrogram`s. Each
            argument should be a list of values. Any list
            of length greater than 1 should be the same
            length
    """

    def __init__(
        self, kernel_length: float, sample_rate: float, **kwargs
    ) -> None:
        super().__init__()
        # This method of combination makes sense only when
        # the spectrograms are normalized, so enforce this
        kwargs["normalized"] = [True]
        self.kwargs = self._check_and_format_kwargs(kwargs)

        self.transforms = torch.nn.ModuleList(
            [Spectrogram(**k) for k in self.kwargs]
        )

        dummy_input = torch.ones(int(kernel_length * sample_rate))
        self.shapes = torch.tensor(
            [t(dummy_input).shape for t in self.transforms]
        )

        self.num_freqs = max([shape[0] for shape in self.shapes])
        self.num_times = max([shape[1] for shape in self.shapes])

        bottom_pad = torch.tensor(
            [int(self.num_freqs - shape[0]) for shape in self.shapes]
        )
        right_pad = torch.tensor(
            [int(self.num_times - shape[1]) for shape in self.shapes]
        )
        self.register_buffer("bottom_pad", bottom_pad)
        self.register_buffer("right_pad", right_pad)

        freq_idxs = torch.tensor(
            [
                [int(i * shape[0] / self.num_freqs) for shape in self.shapes]
                for i in range(self.num_freqs)
            ]
        )
        freq_idxs = freq_idxs.repeat(self.num_times, 1, 1).transpose(0, 1)
        time_idxs = torch.tensor(
            [
                [int(i * shape[1] / self.num_times) for shape in self.shapes]
                for i in range(self.num_times)
            ]
        )
        time_idxs = time_idxs.repeat(self.num_freqs, 1, 1)

        self.register_buffer("freq_idxs", freq_idxs)
        self.register_buffer("time_idxs", time_idxs)

    def _check_and_format_kwargs(self, kwargs: Dict[str, List]) -> List:
        lengths = sorted(set([len(v) for v in kwargs.values()]))
        if len(lengths) > 2 or (len(lengths) == 2 and lengths[0] != 1):
            raise ValueError(
                "Spectrogram keyword args should all have the same "
                f"length or be of length one. Got lengths {lengths}"
            )

        if len(lengths) == 2:
            size = lengths[1]
            kwargs = {k: v * int(size / len(v)) for k, v in kwargs.items()}

        return [dict(zip(kwargs, col)) for col in zip(*kwargs.values())]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate spectrograms of the input tensor and
        combine them into a single spectrogram

        Args:
            X:
                Batch of multichannel timeseries which will
                be used to calculate the multi-resolution
                spectrogram. Should have the shape
                `(B, C, T)`, where `B` is the number of
                batches, `C` is the  number of channels,
                and `T` is the number of time samples.
        """
        spectrograms = [t(X) for t in self.transforms]

        left_pad = torch.zeros(len(spectrograms), dtype=torch.int)
        top_pad = torch.zeros(len(spectrograms), dtype=torch.int)
        padded_specs = []
        for spec, left, right, top, bottom in zip(
            spectrograms, left_pad, self.right_pad, top_pad, self.bottom_pad
        ):
            padded_specs.append(F.pad(spec, (left, right, top, bottom)))

        padded_specs = torch.stack(padded_specs)
        remapped_specs = padded_specs[..., self.freq_idxs, self.time_idxs]
        remapped_specs = torch.diagonal(remapped_specs, dim1=0, dim2=-1)

        return torch.max(remapped_specs, axis=-1)[0]


class BatchWhitener(torch.nn.Module):
    """Calculate the PSDs and whiten an entire batch of kernels at once"""

    def __init__(
        self,
        kernel_length: float,
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
        fduration: float,
        fftlength: float,
        n_ffts: torch.Tensor,
        highpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.stride_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int(kernel_length * sample_rate)

        # do foreground length calculation in units of samples,
        # then convert back to length to guard for intification
        strides = (batch_size - 1) * self.stride_size
        fsize = int(fduration * sample_rate)
        size = strides + self.kernel_size + fsize
        length = size / sample_rate
        self.psd_estimator = PsdEstimator(
            length,
            sample_rate,
            fftlength=fftlength,
            overlap=None,
            average="mean",
            fast=highpass is not None,
        )
        self.whitener = Whiten(fduration, sample_rate, highpass)
        self.spectrogram = MultiResolutionSpectrogram(
            kernel_length, sample_rate, n_fft=n_ffts
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, psd = self.psd_estimator(x)
        x = self.whitener(x.double(), psd)
        x = unfold_windows(x, self.kernel_size, self.stride_size)
        x = self.spectrogram(x[:, 0])
        return x
