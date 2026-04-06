import torch
import torch.nn as nn

from ..schema import SortingBatch


class RemoveDuplicateTimes(nn.Module):
    """Remove duplicate spike times (keeps first occurrence).

    Reads:  batch.times, batch.channel_indices
    Writes: batch.times, batch.channel_indices
    """

    def forward(self, batch: SortingBatch) -> SortingBatch:
        times = batch.times
        channel_indices = batch.channel_indices
        assert times is not None and channel_indices is not None

        if len(times) == 0:
            return batch

        keep = torch.where(torch.diff(times) > 0)[0]
        keep = torch.cat([torch.tensor([0], device=times.device), keep + 1])

        batch.times = times[keep]
        batch.channel_indices = channel_indices[keep]
        return batch
