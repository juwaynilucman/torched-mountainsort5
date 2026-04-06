import torch
import torch.nn as nn
from typing import Optional

from ..schema import SortingBatch, SortingParameters


class ExtractSnippets(nn.Module):
    """Extract spike snippets from traces using advanced indexing.

    Reads:  batch.traces, batch.times, batch.channel_indices, batch.channel_locations
    Writes: batch.snippets
    """

    def __init__(self, params: SortingParameters):
        super().__init__()
        self.T1 = params.snippet_T1
        self.T2 = params.snippet_T2
        self.mask_radius = params.snippet_mask_radius

    def forward(self, batch: SortingBatch) -> SortingBatch:
        traces = batch.traces
        times = batch.times
        channel_indices = batch.channel_indices
        channel_locations = batch.channel_locations
        assert traces is not None and times is not None

        batch.snippets = self._extract(traces, times, channel_indices, channel_locations)
        return batch

    def _extract(
        self,
        traces: torch.Tensor,
        times: torch.Tensor,
        channel_indices: Optional[torch.Tensor],
        channel_locations: Optional[torch.Tensor],
    ) -> torch.Tensor:
        M = traces.shape[1]
        L = times.shape[0]
        device = traces.device

        if L == 0:
            return torch.zeros((0, self.T1 + self.T2, M), dtype=traces.dtype, device=device)

        # (L, T) matrix of sample indices to extract
        window = torch.arange(-self.T1, self.T2, device=device)
        extract_indices = times.unsqueeze(1) + window.unsqueeze(0)  # (L, T)

        # Advanced indexing: pull all snippets at once -> (L, T, M)
        snippets = traces[extract_indices]

        # Mask out channels beyond the spatial radius of each spike's peak channel
        if self.mask_radius is not None and channel_indices is not None and channel_locations is not None:
            dists = torch.cdist(channel_locations, channel_locations)
            adj_matrix = dists <= self.mask_radius
            valid_channels = adj_matrix[channel_indices]           # (L, M) bool
            valid_mask = valid_channels.unsqueeze(1).to(snippets.dtype)  # (L, 1, M)
            snippets *= valid_mask

        return snippets
