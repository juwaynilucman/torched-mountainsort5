import torch
import torch.nn as nn

from ..schema import SortingBatch


class OffsetTimesToPeak(nn.Module):
    """Offset spike times so they correspond to actual peaks.

    Reads:  batch.templates, batch.times, batch.labels
    Writes: batch.times, batch.offsets_to_peak
    """

    def __init__(self, detect_sign: int, T1: int):
        super().__init__()
        self.detect_sign = detect_sign
        self.T1 = T1

    def forward(self, batch: SortingBatch) -> SortingBatch:
        templates = batch.templates
        times = batch.times
        labels = batch.labels
        assert templates is not None and times is not None and labels is not None

        offsets = _determine_offsets_to_peak(templates, detect_sign=self.detect_sign, T1=self.T1)
        batch.offsets_to_peak = offsets
        batch.times = _offset_times(times, offsets, labels)
        return batch


def _determine_offsets_to_peak(
    templates: torch.Tensor,
    *,
    detect_sign: int,
    T1: int,
) -> torch.Tensor:
    K = templates.shape[0]
    device = templates.device

    if detect_sign < 0:
        A = -templates
    elif detect_sign > 0:
        A = templates
    else:
        A = torch.abs(templates)

    offsets = torch.zeros(K, dtype=torch.int32, device=device)
    for k in range(K):
        peak_channel = int(torch.argmax(torch.max(A[k], dim=0).values).item())
        peak_time = int(torch.argmax(A[k][:, peak_channel]).item())
        offsets[k] = peak_time - T1

    return offsets


def _offset_times(
    times: torch.Tensor,
    offsets: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if len(labels) == 0:
        return times

    times2 = torch.zeros_like(times)
    K = int(labels.max().item())

    for k in range(1, K + 1):
        inds = torch.where(labels == k)[0]
        times2[inds] = times[inds] + offsets[k - 1]

    return times2
