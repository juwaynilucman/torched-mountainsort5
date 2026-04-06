import torch
import torch.nn as nn

from ..schema import SortingBatch


class ComputeTemplates(nn.Module):
    """Compute median templates per cluster.

    Reads:  batch.snippets, batch.labels
    Writes: batch.templates, batch.peak_channel_indices
    """

    def forward(self, batch: SortingBatch) -> SortingBatch:
        snippets = batch.snippets
        labels = batch.labels
        assert snippets is not None and labels is not None

        templates = _compute_templates(snippets, labels)
        batch.templates = templates

        K = templates.shape[0]
        if K > 0:
            # Peak channel = channel with the most negative minimum across time
            min_per_channel = templates.min(dim=1).values  # (K, M)
            batch.peak_channel_indices = torch.argmin(min_per_channel, dim=1)  # (K,)
        else:
            batch.peak_channel_indices = torch.zeros(0, dtype=torch.int64, device=snippets.device)

        return batch


def _compute_templates(
    snippets: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    L, T, M = snippets.shape

    if L == 0:
        return torch.zeros((0, T, M), dtype=torch.float32, device=snippets.device)

    K = int(labels.max().item())
    templates = torch.zeros((K, T, M), dtype=torch.float32, device=snippets.device)

    for k in range(1, K + 1):
        cluster = snippets[labels == k]
        if cluster.shape[0] == 0:
            templates[k - 1] = float("nan")
        else:
            templates[k - 1] = torch.quantile(cluster, 0.5, dim=0)

    return templates
