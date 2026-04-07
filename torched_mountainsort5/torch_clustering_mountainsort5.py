"""Optimized MountainSort5 pipeline using pure-PyTorch ISO-SPLIT clustering.

This module is a variant of mountainsort5.py that replaces the C++ based
ISO-SPLIT clustering calls (isosplit6 + sklearn PCA + scipy hierarchical
clustering) with the fully PyTorch-native implementation from the
``isosplit6_torch`` package.  All clustering computation stays on the input
device (including GPU), eliminating CPU round-trips.

The rest of the sorting pipeline (detection, alignment, post-processing)
is identical to the original.
"""

import torch
import torch.nn as nn

from .schema import SortingBatch, SortingParameters
from .modules.detect_spikes import DetectSpikes
from .modules.extract_snippets import ExtractSnippets
from .modules.remove_duplicates import RemoveDuplicateTimes
from .modules.compute_pca import ComputePCA
from .modules.compute_templates import ComputeTemplates
from .modules.align_templates import AlignTemplates
from .modules.align_snippets import AlignSnippets
from .modules.offset_times import OffsetTimesToPeak
from .modules.sort_and_filter import SortTimes, RemoveOutOfBounds, ReorderUnits

from isosplit6_torch import Isosplit6Clustering as _Isosplit6ClusteringTorch


class _TorchIsosplit6Adapter(nn.Module):
    """Adapter that wraps isosplit6_torch.Isosplit6Clustering for SortingBatch.

    The upstream torch module expects a plain (N, D) tensor and returns (N,)
    labels.  This adapter bridges it to the SortingBatch protocol used by the
    MountainSort5 pipeline (reads batch.features, writes batch.labels).
    """

    def __init__(self, params: SortingParameters):
        super().__init__()
        self.clustering = _Isosplit6ClusteringTorch(
            npca_per_subdivision=params.npca_per_subdivision,
        )

    def forward(self, batch: SortingBatch) -> SortingBatch:
        features = batch.features
        assert features is not None
        batch.labels = self.clustering(features)
        return batch


class TorchClusteringMountainSort5(nn.Module):
    """Top-level orchestrator for the MountainSort5 sorting pipeline.

    This variant uses the pure-PyTorch ISO-SPLIT clustering algorithm from
    ``isosplit6_torch`` instead of the C++ isosplit6 library, enabling full
    GPU-resident computation during the clustering stages.

    Each stage is an nn.Module that reads from and writes to a SortingBatch.
    The forward() method mirrors the logic of sorting_scheme1().
    """

    def __init__(self, params: SortingParameters, sampling_frequency: float):
        super().__init__()
        self.params = params
        self.sampling_frequency = sampling_frequency

        # -- detection --
        self.detect_spikes = DetectSpikes(params, sampling_frequency)
        self.remove_duplicates = RemoveDuplicateTimes()
        self.extract_snippets = ExtractSnippets(params)

        # -- first pass: PCA + clustering + templates --
        self.compute_pca = ComputePCA(params)
        self.clustering = _TorchIsosplit6Adapter(params)
        self.compute_templates = ComputeTemplates()

        # -- alignment (used when skip_alignment=False) --
        self.align_templates = AlignTemplates()
        self.align_snippets = AlignSnippets()

        # -- second pass after alignment: reuses compute_pca, clustering, compute_templates --
        self.offset_times_to_peak = OffsetTimesToPeak(params.detect_sign, params.snippet_T1)

        # -- post-processing --
        self.sort_times = SortTimes()
        self.remove_out_of_bounds = RemoveOutOfBounds(params)
        self.reorder_units = ReorderUnits()

    def forward(self, batch: SortingBatch) -> SortingBatch:
        # -- detection --
        batch = self.detect_spikes(batch)
        batch = self.remove_duplicates(batch)
        batch = self.extract_snippets(batch)

        # -- first pass --
        batch = self.compute_pca(batch)
        batch = self.clustering(batch)
        batch = self.compute_templates(batch)

        # -- alignment --
        if not self.params.skip_alignment:
            batch = self.align_templates(batch)
            batch = self.align_snippets(batch)

            # second pass: re-cluster aligned snippets
            batch.features = None  # clear stale features
            batch = self.compute_pca(batch)
            batch = self.clustering(batch)
            batch = self.compute_templates(batch)

            # offset times to actual peaks
            batch = self.offset_times_to_peak(batch)

        # -- post-processing --
        batch = self.sort_times(batch)
        batch = self.remove_out_of_bounds(batch)
        batch = self.reorder_units(batch)

        return batch
