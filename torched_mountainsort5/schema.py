from dataclasses import dataclass, field
from typing import Optional, List, Union

import torch


@dataclass
class SortingParameters:
    """Parameters for the MountainSort5 sorting pipeline.

    Mirrors Scheme1SortingParameters but lives in the torched package
    so it has no numpy/spikeinterface dependency.
    """
    detect_threshold: float = 5.5
    detect_channel_radius: Optional[float] = None
    detect_time_radius_msec: float = 0.5
    detect_sign: int = -1
    snippet_T1: int = 20
    snippet_T2: int = 20
    snippet_mask_radius: Optional[float] = None
    npca_per_channel: int = 3
    npca_per_subdivision: int = 10
    skip_alignment: bool = False


@dataclass
class SortingBatch:
    """Data contract that flows between pipeline modules.

    Every field is an Optional[Tensor].  Each module reads the fields it
    needs and writes the fields it produces — the orchestrator never
    inspects internals.

    Fields are populated progressively as data moves through the pipeline:
      traces -> times/channel_indices -> snippets -> features -> labels -> templates
    """
    # -- raw input --
    traces: Optional[torch.Tensor] = None                # (N, M)
    channel_locations: Optional[torch.Tensor] = None      # (M, D)
    sampling_frequency: Optional[float] = None

    # -- spike detection --
    times: Optional[torch.Tensor] = None                  # (L,)   int64
    channel_indices: Optional[torch.Tensor] = None        # (L,)   int32

    # -- snippets / features --
    snippets: Optional[torch.Tensor] = None               # (L, T, M) float32
    features: Optional[torch.Tensor] = None               # (L, npca) float32

    # -- clustering --
    labels: Optional[torch.Tensor] = None                 # (L,)   int32

    # -- templates --
    templates: Optional[torch.Tensor] = None              # (K, T, M) float32
    peak_channel_indices: Optional[torch.Tensor] = None   # (K,)   int64

    # -- alignment offsets (populated when skip_alignment=False) --
    alignment_offsets: Optional[torch.Tensor] = None      # (K,)   int32
    offsets_to_peak: Optional[torch.Tensor] = None        # (K,)   int32

    @property
    def device(self) -> torch.device:
        """Infer device from the first populated tensor."""
        for v in self.__dict__.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")

    @property
    def num_channels(self) -> int:
        assert self.traces is not None
        return self.traces.shape[1]

    @property
    def num_timepoints(self) -> int:
        assert self.traces is not None
        return self.traces.shape[0]

    @property
    def num_spikes(self) -> int:
        assert self.times is not None
        return self.times.shape[0]