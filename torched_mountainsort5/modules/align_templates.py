import torch
import torch.nn as nn

from ..schema import SortingBatch


class AlignTemplates(nn.Module):
    """Compute optimal alignment offsets for templates via pairwise cross-correlation.

    Reads:  batch.templates
    Writes: batch.alignment_offsets
    """

    def forward(self, batch: SortingBatch) -> SortingBatch:
        templates = batch.templates
        assert templates is not None

        batch.alignment_offsets = _align_templates(templates)
        return batch


def _compute_pairwise_optimal_offset(
    template1: torch.Tensor, template2: torch.Tensor
) -> tuple:
    T = template1.shape[0]
    best_inner_product = -float("inf")
    best_offset = 0
    for offset in range(T):
        ip = torch.sum(torch.roll(template1, shifts=offset, dims=0) * template2).item()
        if ip > best_inner_product:
            best_inner_product = ip
            best_offset = offset
    if best_offset > T // 2:
        best_offset = best_offset - T
    return best_offset, best_inner_product


def _align_templates(templates: torch.Tensor) -> torch.Tensor:
    K = templates.shape[0]
    device = templates.device

    offsets = torch.zeros(K, dtype=torch.int32, device=device)
    pairwise_optimal_offsets = torch.zeros((K, K), dtype=torch.int32, device=device)
    pairwise_inner_products = torch.zeros((K, K), dtype=torch.float32, device=device)

    for k1 in range(K):
        for k2 in range(K):
            offset, ip = _compute_pairwise_optimal_offset(templates[k1], templates[k2])
            pairwise_optimal_offsets[k1, k2] = offset
            pairwise_inner_products[k1, k2] = ip

    for _ in range(20):
        something_changed = False
        for k1 in range(K):
            weighted_sum = 0.0
            total_weight = 0.0
            for k2 in range(K):
                if k1 != k2:
                    offset = pairwise_optimal_offsets[k1, k2] + offsets[k2]
                    weight = pairwise_inner_products[k1, k2]
                    weighted_sum += weight.item() * offset.item()
                    total_weight += weight.item()
            if total_weight > 0:
                avg_offset = int(weighted_sum / total_weight)
            else:
                avg_offset = 0
            if avg_offset != offsets[k1].item():
                something_changed = True
                offsets[k1] = avg_offset
        if not something_changed:
            break

    return offsets
