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
from .modules.clustering import Isosplit6Clustering
from .modules.offset_times import OffsetTimesToPeak
from .modules.sort_and_filter import SortTimes, RemoveOutOfBounds, ReorderUnits


class MountainSort5(nn.Module):
    """Top-level orchestrator for the MountainSort5 sorting pipeline.

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
        self.clustering = Isosplit6Clustering(params)
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
