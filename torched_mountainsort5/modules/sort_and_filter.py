import torch
import torch.nn as nn

from ..schema import SortingBatch, SortingParameters


class SortTimes(nn.Module):
    """Sort times (and labels) into ascending order after offset adjustments.

    Reads:  batch.times, batch.labels
    Writes: batch.times, batch.labels
    """

    def forward(self, batch: SortingBatch) -> SortingBatch:
        times = batch.times
        labels = batch.labels
        assert times is not None and labels is not None

        sort_inds = torch.argsort(times, stable=True)
        batch.times = times[sort_inds]
        batch.labels = labels[sort_inds]
        return batch


class RemoveOutOfBounds(nn.Module):
    """Remove spikes whose times fall outside the valid trace range.

    Reads:  batch.times, batch.labels, batch.traces
    Writes: batch.times, batch.labels
    """

    def __init__(self, params: SortingParameters):
        super().__init__()
        self.T1 = params.snippet_T1
        self.T2 = params.snippet_T2

    def forward(self, batch: SortingBatch) -> SortingBatch:
        times = batch.times
        labels = batch.labels
        assert times is not None and labels is not None and batch.traces is not None

        N = batch.num_timepoints
        valid = (times >= self.T1) & (times < N - self.T2)
        batch.times = times[valid]
        batch.labels = labels[valid]
        return batch


class ReorderUnits(nn.Module):
    """Relabel clusters so units are ordered by peak channel index.

    Reads:  batch.labels, batch.peak_channel_indices
    Writes: batch.labels
    """

    def forward(self, batch: SortingBatch) -> SortingBatch:
        labels = batch.labels
        peak_channels = batch.peak_channel_indices
        assert labels is not None and peak_channels is not None

        K = int(labels.max().item()) if len(labels) > 0 else 0
        if K == 0:
            return batch

        # Mark empty clusters with inf so they sort to the end
        aa = peak_channels.to(torch.float32).clone()
        for k in range(1, K + 1):
            if (labels == k).sum() == 0:
                aa[k - 1] = float("inf")

        # Double-argsort gives the rank (i.e., new label) for each old label
        new_labels_mapping = torch.argsort(torch.argsort(aa, stable=True), stable=True) + 1
        batch.labels = new_labels_mapping[labels - 1]
        return batch
