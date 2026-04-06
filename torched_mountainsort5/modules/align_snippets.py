import torch
import torch.nn as nn

from ..schema import SortingBatch


class AlignSnippets(nn.Module):
    """Roll each cluster's snippets by its alignment offset and update times.

    Reads:  batch.snippets, batch.times, batch.labels, batch.alignment_offsets
    Writes: batch.snippets, batch.times
    """

    def forward(self, batch: SortingBatch) -> SortingBatch:
        snippets = batch.snippets
        times = batch.times
        labels = batch.labels
        offsets = batch.alignment_offsets
        assert snippets is not None and times is not None
        assert labels is not None and offsets is not None

        batch.snippets = _align_snippets(snippets, offsets, labels)
        # Subtract offsets to correspond to shifting the template
        batch.times = _offset_times(times, -offsets, labels)
        return batch


def _align_snippets(
    snippets: torch.Tensor,
    offsets: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if len(labels) == 0:
        return snippets

    aligned = torch.zeros_like(snippets)
    K = int(labels.max().item())

    for k in range(1, K + 1):
        inds = torch.where(labels == k)[0]
        if len(inds) > 0:
            aligned[inds] = torch.roll(snippets[inds], shifts=int(offsets[k - 1].item()), dims=1)

    return aligned


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
