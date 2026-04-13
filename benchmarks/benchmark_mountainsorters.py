"""
MountainSort5 benchmark.

Compares five execution paths at identical 4-stage boundaries:
  - original:       Original MountainSort5 (low-level NumPy functions called directly)
  - cpp_iso_cpu:    PyTorched MountainSort5 with C++ isosplit6 (CPU)
  - cpp_iso_gpu:    PyTorched MountainSort5 with C++ isosplit6 (GPU)
  - torch_iso_cpu:  PyTorched MountainSort5 with PyTorch isosplit6 (CPU)
  - torch_iso_gpu:  PyTorched MountainSort5 with PyTorch isosplit6 (GPU)

Stages:
  A  Detection & Extraction   (detect_spikes, remove_duplicates, extract_snippets)
  B  Clustering (1st pass)     (compute_pca, isosplit6, compute_templates)
  C  Alignment & Re-Clustering (align_templates, align_snippets, 2nd PCA+cluster+templates, offset_to_peak)
  D  Post-processing           (sort_times, remove_out_of_bounds, reorder_units)
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

try:
    from threadpoolctl import threadpool_info
    HAS_THREADPOOLCTL = True
except ImportError:
    HAS_THREADPOOLCTL = False
import scipy.io as sio
import spikeinterface.core as si
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import probeinterface as pi

import mountainsort5 as ms5
from mountainsort5.core.detect_spikes import detect_spikes
from mountainsort5.core.remove_duplicate_times import remove_duplicate_times
from mountainsort5.core.extract_snippets import extract_snippets
from mountainsort5.core.compute_pca_features import compute_pca_features
from mountainsort5.core.isosplit6_subdivision_method import isosplit6_subdivision_method
from mountainsort5.core.compute_templates import compute_templates
from mountainsort5.core.align_templates import align_templates
from mountainsort5.core.align_snippets import align_snippets
from mountainsort5.core.offset_times import offset_times
from mountainsort5.core.determine_offsets_to_peak import determine_offsets_to_peak

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

STAGE_NAMES = ["A_detection", "B_clustering", "C_alignment", "D_postprocessing"]
ALL_TARGETS = ["original", "cpp_iso_cpu", "cpp_iso_gpu", "torch_iso_cpu", "torch_iso_gpu"]
TARGET_LABELS = {
    "original":      "Original MountainSort5",
    "cpp_iso_cpu":   "PyTorched MountainSort5 with C++ isosplit6 (CPU)",
    "cpp_iso_gpu":   "PyTorched MountainSort5 with C++ isosplit6 (GPU)",
    "torch_iso_cpu": "PyTorched MountainSort5 with PyTorch isosplit6 (CPU)",
    "torch_iso_gpu": "PyTorched MountainSort5 with PyTorch isosplit6 (GPU)",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    sampling_freq: int = 30000
    num_channels: int = 384
    dtype: str = "int16"
    npx_bin_path: Path = Path(r"C:\Users\juway\Documents\Marquees-smith\c46\subset_data\raw_1pct.bin")
    chan_map_path: Path = Path(r"D:\chanMap.mat")
    results_dir: Path = Path("results")
    n_runs: int = 3
    targets: List[str] = field(default_factory=lambda: ALL_TARGETS)
    scheme1_params: ms5.Scheme1SortingParameters = field(default_factory=lambda: ms5.Scheme1SortingParameters(
        detect_channel_radius=150,
        detect_threshold=5.5,
        snippet_mask_radius=150,
        skip_alignment=False,
    ))


# ---------------------------------------------------------------------------
# Resource & timer helpers
# ---------------------------------------------------------------------------
@dataclass
class StageResources:
    """Resource usage observed during a single stage."""
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    cpu_percent: float = 0.0
    process_threads: int = 0
    torch_threads: Optional[int] = None
    torch_interop_threads: Optional[int] = None
    blas_info: Optional[List[dict]] = None
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0

    @property
    def rss_delta_mb(self) -> float:
        return self.rss_after_mb - self.rss_before_mb


class StageMonitor:
    """Context-manager that captures wall time AND resource usage per stage."""

    def __init__(self, use_cuda_sync: bool = False):
        self.use_cuda_sync = use_cuda_sync
        self.elapsed: float = 0.0
        self.resources = StageResources()

    def __enter__(self):
        proc = psutil.Process()
        self.resources.rss_before_mb = proc.memory_info().rss / 1e6
        # Prime cpu_percent measurement (first call returns 0.0)
        proc.cpu_percent()
        self._proc = proc

        if self.use_cuda_sync:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.use_cuda_sync:
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start

        proc = self._proc
        self.resources.rss_after_mb = proc.memory_info().rss / 1e6
        self.resources.cpu_percent = proc.cpu_percent()
        self.resources.process_threads = proc.num_threads()

        if HAS_TORCH:
            self.resources.torch_threads = torch.get_num_threads()
            self.resources.torch_interop_threads = torch.get_num_interop_threads()

        if HAS_THREADPOOLCTL:
            self.resources.blas_info = [
                {"library": p["internal_api"], "num_threads": p["num_threads"]}
                for p in threadpool_info()
            ]

        if self.use_cuda_sync:
            self.resources.gpu_allocated_mb = torch.cuda.max_memory_allocated() / 1e6
            self.resources.gpu_reserved_mb = torch.cuda.max_memory_reserved() / 1e6


@dataclass
class RunTimings:
    A_detection: float = 0.0
    B_clustering: float = 0.0
    C_alignment: float = 0.0
    D_postprocessing: float = 0.0

    @property
    def total(self) -> float:
        return self.A_detection + self.B_clustering + self.C_alignment + self.D_postprocessing


@dataclass
class RunResources:
    A_detection: StageResources = field(default_factory=StageResources)
    B_clustering: StageResources = field(default_factory=StageResources)
    C_alignment: StageResources = field(default_factory=StageResources)
    D_postprocessing: StageResources = field(default_factory=StageResources)


# ---------------------------------------------------------------------------
# Data loading & preprocessing  (done ONCE)
# ---------------------------------------------------------------------------
def load_and_preprocess(cfg: BenchmarkConfig) -> Tuple[si.BaseRecording, np.ndarray, np.ndarray]:
    """Load raw recording, apply bandpass + whiten, return (recording, traces, channel_locations).

    The returned traces are the fully-materialised float32 array held in memory.
    """
    print(f"Loading recording: {cfg.npx_bin_path}")
    recording = si.read_binary(
        cfg.npx_bin_path,
        sampling_frequency=cfg.sampling_freq,
        dtype=cfg.dtype,
        num_channels=cfg.num_channels,
    )
    mat = sio.loadmat(cfg.chan_map_path)
    positions = np.column_stack((mat["xcoords"].flatten(), mat["ycoords"].flatten()))
    probe = pi.Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 6})
    probe.set_device_channel_indices(np.arange(cfg.num_channels))
    recording = recording.set_probe(probe)

    # --- NEW: Slice the recording for development ---
    # Example 1: Grab the first 32 channels
    chan_range = range(155, 167)  # 155:167
    channels_to_keep = recording.get_channel_ids()[chan_range]   # Adjust this range as needed for testing
    
    # Example 2: Grab specific channel IDs (if you know where a good unit is)
    # channels_to_keep = [10, 11, 12, 13, 14, 15] 
    
    print(f"Slicing recording to {len(channels_to_keep)} channels for rapid testing...")
    recording_sliced = recording.select_channels(channel_ids=channels_to_keep)
    
    print("Preprocessing (bandpass + whiten) ...")
    rec_filtered = spre.bandpass_filter(recording_sliced, freq_min=300, freq_max=6000, dtype="float32")
    rec_preprocessed = spre.whiten(rec_filtered, seed=42)

    # Materialise into RAM so every run reads from the same master array
    print("Materialising preprocessed traces into RAM ...")
    traces = rec_preprocessed.get_traces().astype(np.float32, copy=False)
    channel_locations = rec_preprocessed.get_channel_locations()
    print(f"  traces shape: {traces.shape}  ({traces.nbytes / 1e6:.1f} MB)")

    return rec_preprocessed, traces, channel_locations


# ---------------------------------------------------------------------------
# Runner:  Original MountainSort5  (calls low-level NumPy functions directly)
# ---------------------------------------------------------------------------
def run_original(
    traces_master: np.ndarray,
    channel_locations: np.ndarray,
    sampling_frequency: float,
    params: ms5.Scheme1SortingParameters,
) -> Tuple[RunTimings, RunResources, np.ndarray, np.ndarray]:
    traces = np.copy(traces_master)
    N, M = traces.shape
    timings = RunTimings()
    resources = RunResources()
    time_radius = int(math.ceil(params.detect_time_radius_msec / 1000 * sampling_frequency))
    npca = params.npca_per_channel * M

    # --- Stage A: Detection & Extraction ---
    m = StageMonitor()
    with m:
        times, channel_indices = detect_spikes(
            traces=traces,
            channel_locations=channel_locations,
            time_radius=time_radius,
            channel_radius=params.detect_channel_radius,
            detect_threshold=params.detect_threshold,
            detect_sign=params.detect_sign,
            margin_left=params.snippet_T1,
            margin_right=params.snippet_T2,
            verbose=False,
        )
        times, channel_indices = remove_duplicate_times(times, channel_indices)
        snippets = extract_snippets(
            traces=traces,
            channel_locations=channel_locations,
            mask_radius=params.snippet_mask_radius,
            times=times,
            channel_indices=channel_indices,
            T1=params.snippet_T1,
            T2=params.snippet_T2,
        )
    timings.A_detection = m.elapsed
    resources.A_detection = m.resources
    L, T = snippets.shape[0], snippets.shape[1]

    # --- Stage B: Clustering (1st pass) ---
    m = StageMonitor()
    with m:
        features = compute_pca_features(snippets.reshape((L, T * M)), npca=npca)
        labels = isosplit6_subdivision_method(X=features, npca_per_subdivision=params.npca_per_subdivision)
        K = int(np.max(labels)) if len(labels) > 0 else 0
        templates_ = compute_templates(snippets=snippets, labels=labels)
        peak_channel_indices = [int(np.argmin(np.min(templates_[i], axis=0))) for i in range(K)]
    timings.B_clustering = m.elapsed
    resources.B_clustering = m.resources

    # --- Stage C: Alignment & Re-Clustering ---
    m = StageMonitor()
    with m:
        if not params.skip_alignment:
            offsets_ = align_templates(templates_)
            snippets = align_snippets(snippets, offsets_, labels)
            times = offset_times(times, -offsets_, labels)

            # 2nd pass
            features = compute_pca_features(snippets.reshape((L, T * M)), npca=npca)
            labels = isosplit6_subdivision_method(X=features, npca_per_subdivision=params.npca_per_subdivision)
            K = int(np.max(labels)) if len(labels) > 0 else 0
            templates_ = compute_templates(snippets=snippets, labels=labels)
            peak_channel_indices = [int(np.argmin(np.min(templates_[i], axis=0))) for i in range(K)]

            offsets_to_peak = determine_offsets_to_peak(templates_, detect_sign=params.detect_sign, T1=params.snippet_T1)
            times = offset_times(times, offsets_to_peak, labels)
    timings.C_alignment = m.elapsed
    resources.C_alignment = m.resources

    # --- Stage D: Post-processing ---
    m = StageMonitor()
    with m:
        sort_inds = np.argsort(times, kind="stable")
        times = times[sort_inds]
        labels = labels[sort_inds]

        inds_ok = np.where((times >= params.snippet_T1) & (times < N - params.snippet_T2))[0]
        times = times[inds_ok]
        labels = labels[inds_ok]

        aa = np.array([float(x) for x in peak_channel_indices])
        for k in range(1, K + 1):
            if len(np.where(labels == k)[0]) == 0:
                aa[k - 1] = np.inf
        new_labels_map = np.argsort(np.argsort(aa, kind="stable"), kind="stable") + 1
        labels = new_labels_map[labels - 1]
    timings.D_postprocessing = m.elapsed
    resources.D_postprocessing = m.resources

    return timings, resources, np.asarray(times), np.asarray(labels)


# ---------------------------------------------------------------------------
# Runner:  Torched (CPU or GPU)  – calls nn.Module stages directly
# ---------------------------------------------------------------------------
def _make_torched_params(params: ms5.Scheme1SortingParameters):
    from torched_mountainsort5.schema import SortingParameters
    return SortingParameters(
        detect_threshold=params.detect_threshold,
        detect_channel_radius=params.detect_channel_radius,
        detect_time_radius_msec=params.detect_time_radius_msec,
        detect_sign=params.detect_sign,
        snippet_T1=params.snippet_T1,
        snippet_T2=params.snippet_T2,
        snippet_mask_radius=params.snippet_mask_radius,
        npca_per_channel=params.npca_per_channel,
        npca_per_subdivision=params.npca_per_subdivision,
        skip_alignment=params.skip_alignment,
    )


def run_torched(
    traces_master: np.ndarray,
    channel_locations: np.ndarray,
    sampling_frequency: float,
    params: ms5.Scheme1SortingParameters,
    device: str,
    use_torch_iso: bool = False,
) -> Tuple[RunTimings, RunResources, np.ndarray, np.ndarray]:
    from torched_mountainsort5.schema import SortingBatch
    from torched_mountainsort5.mountainsort5 import MountainSort5
    from torched_mountainsort5.torch_clustering_mountainsort5 import TorchIsosplit6MountainSort5

    dev = torch.device(device)
    use_cuda_sync = dev.type == "cuda"

    torched_params = _make_torched_params(params)
    model_cls = TorchIsosplit6MountainSort5 if use_torch_iso else MountainSort5
    model = model_cls(torched_params, sampling_frequency).to(dev)

    traces_t = torch.as_tensor(np.copy(traces_master), dtype=torch.float32, device=dev)
    chan_locs_t = torch.as_tensor(channel_locations.copy(), dtype=torch.float32, device=dev)

    batch = SortingBatch(
        traces=traces_t,
        channel_locations=chan_locs_t,
        sampling_frequency=sampling_frequency,
    )

    timings = RunTimings()
    resources = RunResources()

    # --- Stage A: Detection & Extraction ---
    m = StageMonitor(use_cuda_sync)
    with m:
        batch = model.detect_spikes(batch)
        batch = model.remove_duplicates(batch)
        batch = model.extract_snippets(batch)
    timings.A_detection = m.elapsed
    resources.A_detection = m.resources

    # --- Stage B: Clustering (1st pass) ---
    m = StageMonitor(use_cuda_sync)
    with m:
        batch = model.compute_pca(batch)
        batch = model.clustering(batch)
        batch = model.compute_templates(batch)
    timings.B_clustering = m.elapsed
    resources.B_clustering = m.resources

    # --- Stage C: Alignment & Re-Clustering ---
    m = StageMonitor(use_cuda_sync)
    with m:
        if not torched_params.skip_alignment:
            batch = model.align_templates(batch)
            batch = model.align_snippets(batch)

            batch.features = None  # clear stale features
            batch = model.compute_pca(batch)
            batch = model.clustering(batch)
            batch = model.compute_templates(batch)

            batch = model.offset_times_to_peak(batch)
    timings.C_alignment = m.elapsed
    resources.C_alignment = m.resources

    # --- Stage D: Post-processing ---
    m = StageMonitor(use_cuda_sync)
    with m:
        batch = model.sort_times(batch)
        batch = model.remove_out_of_bounds(batch)
        batch = model.reorder_units(batch)
    timings.D_postprocessing = m.elapsed
    resources.D_postprocessing = m.resources

    out_times = batch.times.cpu().numpy()
    out_labels = batch.labels.cpu().numpy()
    return timings, resources, out_times, out_labels


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def save_outputs(results_dir: Path, target: str, run_idx: int, times: np.ndarray, labels: np.ndarray):
    out_dir = results_dir / target / f"run_{run_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "spike_times.npy", times)
    np.save(out_dir / "labels.npy", labels)


def run_target(
    target: str,
    cfg: BenchmarkConfig,
    traces: np.ndarray,
    channel_locations: np.ndarray,
    sampling_frequency: float,
) -> Tuple[List[RunTimings], List[RunResources], List[np.ndarray], List[np.ndarray]]:
    all_timings: List[RunTimings] = []
    all_resources: List[RunResources] = []
    all_times: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    # CUDA warm-up (one throw-away run)
    if target in ("cpp_iso_gpu", "torch_iso_gpu"):
        use_torch_iso = target == "torch_iso_gpu"
        print("  CUDA warm-up run ...")
        run_torched(traces, channel_locations, sampling_frequency, cfg.scheme1_params, "cuda", use_torch_iso=use_torch_iso)
        torch.cuda.empty_cache()

    for i in range(cfg.n_runs):
        np.random.seed(42)
        if HAS_TORCH:
            torch.manual_seed(42)

        print(f"  Run {i + 1}/{cfg.n_runs} ... ", end="", flush=True)

        if target == "original":
            timings, res, t_out, l_out = run_original(traces, channel_locations, sampling_frequency, cfg.scheme1_params)
        elif target == "cpp_iso_cpu":
            timings, res, t_out, l_out = run_torched(traces, channel_locations, sampling_frequency, cfg.scheme1_params, "cpu")
        elif target == "cpp_iso_gpu":
            timings, res, t_out, l_out = run_torched(traces, channel_locations, sampling_frequency, cfg.scheme1_params, "cuda")
        elif target == "torch_iso_cpu":
            timings, res, t_out, l_out = run_torched(traces, channel_locations, sampling_frequency, cfg.scheme1_params, "cpu", use_torch_iso=True)
        elif target == "torch_iso_gpu":
            timings, res, t_out, l_out = run_torched(traces, channel_locations, sampling_frequency, cfg.scheme1_params, "cuda", use_torch_iso=True)
        else:
            raise ValueError(f"Unknown target: {target}")

        print(f"total={timings.total:.3f}s  (A={timings.A_detection:.3f}  B={timings.B_clustering:.3f}  C={timings.C_alignment:.3f}  D={timings.D_postprocessing:.3f})")

        save_outputs(cfg.results_dir, target, i, t_out, l_out)
        all_timings.append(timings)
        all_resources.append(res)
        all_times.append(t_out)
        all_labels.append(l_out)

    return all_timings, all_resources, all_times, all_labels


# ---------------------------------------------------------------------------
# Summary & Validation
# ---------------------------------------------------------------------------
def print_summary_table(results: Dict[str, List[RunTimings]]):
    label_width = max((len(TARGET_LABELS[t]) for t in results), default=18)
    label_width = max(label_width, len("Target"))
    total_width = label_width + 2 + (16 * (len(STAGE_NAMES) + 1))

    print("\n" + "=" * total_width)
    print("PERFORMANCE SUMMARY (seconds)")
    print("=" * total_width)
    header = f"{'Target':<{label_width}}"
    for stage in STAGE_NAMES + ["Total"]:
        header += f"{'':>2}{stage:>14}"
    print(header)
    print("-" * total_width)

    for target, timings_list in results.items():
        if not timings_list:
            continue
        vals = {s: [] for s in STAGE_NAMES + ["Total"]}
        for t in timings_list:
            vals["A_detection"].append(t.A_detection)
            vals["B_clustering"].append(t.B_clustering)
            vals["C_alignment"].append(t.C_alignment)
            vals["D_postprocessing"].append(t.D_postprocessing)
            vals["Total"].append(t.total)

        row = f"{TARGET_LABELS[target]:<{label_width}}"
        for stage in STAGE_NAMES + ["Total"]:
            mean = np.mean(vals[stage])
            std = np.std(vals[stage])
            row += f"  {mean:>6.3f}+/-{std:<5.3f}"
        print(row)

    print("=" * total_width)


def print_resource_table(results: Dict[str, List[RunResources]]):
    """Print per-target resource usage: threads, CPU%, RSS, and GPU memory."""
    print("\n" + "=" * 120)
    print("RESOURCE USAGE (per stage, averaged over runs)")
    print("=" * 120)

    for target, resources_list in results.items():
        if not resources_list:
            continue
        label = TARGET_LABELS[target]
        print(f"\n  {label}")
        print(f"  {'Stage':<20} {'CPU%':>7} {'Threads':>8} {'Torch Thr':>10} {'BLAS Thr':>9} {'RSS Δ MB':>9} {'GPU Alloc MB':>13} {'GPU Resv MB':>12}")
        print(f"  {'-' * 100}")

        for stage in STAGE_NAMES:
            cpu_pcts, n_threads, torch_thr, blas_thr = [], [], [], []
            rss_deltas, gpu_allocs, gpu_resvs = [], [], []

            for res in resources_list:
                sr: StageResources = getattr(res, stage)
                cpu_pcts.append(sr.cpu_percent)
                n_threads.append(sr.process_threads)
                torch_thr.append(sr.torch_threads if sr.torch_threads is not None else 0)
                rss_deltas.append(sr.rss_delta_mb)
                gpu_allocs.append(sr.gpu_allocated_mb)
                gpu_resvs.append(sr.gpu_reserved_mb)

                if sr.blas_info:
                    blas_thr.append(max(p["num_threads"] for p in sr.blas_info))
                else:
                    blas_thr.append(0)

            row = f"  {stage:<20}"
            row += f" {np.mean(cpu_pcts):>6.1f}%"
            row += f" {int(np.mean(n_threads)):>8}"
            row += f" {int(np.mean(torch_thr)):>10}"
            row += f" {int(np.mean(blas_thr)):>9}"
            row += f" {np.mean(rss_deltas):>+9.1f}"
            if any(v > 0 for v in gpu_allocs):
                row += f" {np.mean(gpu_allocs):>13.1f}"
                row += f" {np.mean(gpu_resvs):>12.1f}"
            else:
                row += f" {'n/a':>13}"
                row += f" {'n/a':>12}"
            print(row)

    print("\n" + "=" * 120)


def fidelity_check(
    results: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]],
):
    """Port parity (original vs each PyTorched variant) and determinism (run 0 vs run n-1)."""
    print("\n" + "=" * 90)
    print("FIDELITY CHECKS")
    print("=" * 90)

    # --- Port parity ---
    pairs = [
        ("original", "cpp_iso_cpu"),
        ("original", "cpp_iso_gpu"),
        ("original", "torch_iso_cpu"),
        ("original", "torch_iso_gpu"),
    ]
    for t1, t2 in pairs:
        if t1 not in results or t2 not in results:
            continue
        times_a, labels_a = results[t1][0][0], results[t1][1][0]  # run 0
        times_b, labels_b = results[t2][0][0], results[t2][1][0]

        times_match = np.array_equal(times_a, times_b)
        labels_match = np.array_equal(labels_a, labels_b)

        label_pair = f"{TARGET_LABELS[t1]} vs {TARGET_LABELS[t2]}"
        if times_match and labels_match:
            print(f"  {label_pair}:  IDENTICAL (times and labels)")
        else:
            common = np.intersect1d(times_a, times_b)
            frac = len(common) / max(len(times_a), 1)
            print(f"  {label_pair}:  spikes={len(times_a)} vs {len(times_b)}, "
                  f"shared_times={len(common)} ({frac:.4f}), labels_match={labels_match}")

    # --- Determinism (run 0 vs last run within each target) ---
    print()
    for target, (all_times, all_labels) in results.items():
        label = TARGET_LABELS[target]
        if len(all_times) < 2:
            print(f"  {label} determinism:  skipped (only 1 run)")
            continue
        t0, t_last = all_times[0], all_times[-1]
        l0, l_last = all_labels[0], all_labels[-1]
        if np.array_equal(t0, t_last) and np.array_equal(l0, l_last):
            print(f"  {label} determinism:  PASS (run 0 == run {len(all_times) - 1})")
        else:
            common = np.intersect1d(t0, t_last)
            frac = len(common) / max(len(t0), 1)
            print(f"  {label} determinism:  DRIFT  shared_times={len(common)}/{len(t0)} ({frac:.4f})")

    print("=" * 90)


def _stage_resources_to_dict(sr: StageResources) -> dict:
    d = {
        "rss_before_mb": round(sr.rss_before_mb, 1),
        "rss_after_mb": round(sr.rss_after_mb, 1),
        "rss_delta_mb": round(sr.rss_delta_mb, 1),
        "cpu_percent": round(sr.cpu_percent, 1),
        "process_threads": sr.process_threads,
    }
    if sr.torch_threads is not None:
        d["torch_threads"] = sr.torch_threads
        d["torch_interop_threads"] = sr.torch_interop_threads
    if sr.blas_info:
        d["blas_info"] = sr.blas_info
    if sr.gpu_allocated_mb > 0:
        d["gpu_allocated_mb"] = round(sr.gpu_allocated_mb, 1)
        d["gpu_reserved_mb"] = round(sr.gpu_reserved_mb, 1)
    return d


def save_json_report(
    cfg: BenchmarkConfig,
    timing_results: Dict[str, List[RunTimings]],
    resource_results: Dict[str, List[RunResources]],
):
    report = {"n_runs": cfg.n_runs, "targets": {}}
    for target, timings_list in timing_results.items():
        resources_list = resource_results.get(target, [])
        runs = []
        for i, t in enumerate(timings_list):
            run_entry = {
                "A_detection": t.A_detection,
                "B_clustering": t.B_clustering,
                "C_alignment": t.C_alignment,
                "D_postprocessing": t.D_postprocessing,
                "total": t.total,
            }
            if i < len(resources_list):
                res = resources_list[i]
                run_entry["resources"] = {
                    stage: _stage_resources_to_dict(getattr(res, stage))
                    for stage in STAGE_NAMES
                }
            runs.append(run_entry)
        report["targets"][target] = runs
    out_path = cfg.results_dir / "benchmark_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------
def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Triple-target MountainSort5 benchmark")
    parser.add_argument("-n", "--n-runs", type=int, default=3, help="Number of timed runs per target")
    parser.add_argument("--npx-bin", type=Path, help="Path to raw .bin recording")
    parser.add_argument("--chan-map", type=Path, help="Path to chanMap.mat")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=ALL_TARGETS,
        choices=ALL_TARGETS,
        help="Which targets to benchmark",
    )
    args = parser.parse_args()

    cfg = BenchmarkConfig(n_runs=args.n_runs, targets=args.targets, results_dir=args.results_dir)
    if args.npx_bin:
        cfg.npx_bin_path = args.npx_bin
    if args.chan_map:
        cfg.chan_map_path = args.chan_map
    return cfg


def main():
    cfg = parse_args()

    # Validate targets
    gpu_targets = {"cpp_iso_gpu", "torch_iso_gpu"}
    torch_targets = {"cpp_iso_cpu", "cpp_iso_gpu", "torch_iso_cpu", "torch_iso_gpu"}
    if gpu_targets & set(cfg.targets) and not HAS_CUDA:
        print("WARNING: GPU targets requested but CUDA not available. Removing from targets.")
        cfg.targets = [t for t in cfg.targets if t not in gpu_targets]
    if torch_targets & set(cfg.targets) and not HAS_TORCH:
        print("WARNING: torch-based targets requested but torch not installed. Removing.")
        cfg.targets = [t for t in cfg.targets if t not in torch_targets]

    if not cfg.targets:
        print("No valid targets remaining. Exiting.")
        return

    # --- Preprocess once ---
    rec_preprocessed, traces_master, channel_locations = load_and_preprocess(cfg)
    sampling_frequency = rec_preprocessed.sampling_frequency

    # --- Run each target ---
    timing_results: Dict[str, List[RunTimings]] = {}
    resource_results: Dict[str, List[RunResources]] = {}
    output_results: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}

    for target in cfg.targets:
        label = TARGET_LABELS[target]
        print(f"\n{'=' * 60}")
        print(f"  TARGET: {label}  [{target}]  ({cfg.n_runs} runs)")
        print(f"{'=' * 60}")

        if target in torch_targets:
            torch.cuda.empty_cache() if HAS_CUDA else None

        timings, resources, all_times, all_labels = run_target(
            target, cfg, traces_master, channel_locations, sampling_frequency,
        )
        timing_results[target] = timings
        resource_results[target] = resources
        output_results[target] = (all_times, all_labels)

    # --- Report ---
    print_summary_table(timing_results)
    print_resource_table(resource_results)
    fidelity_check(output_results)
    save_json_report(cfg, timing_results, resource_results)

    # --- Plot ---
    try:
        from plotting.stacked_runtime_bar import plot_stacked_runtime
        plot_stacked_runtime(
            timing_results,
            TARGET_LABELS,
            cfg.results_dir / "benchmark_runtime.pdf",
        )
    except ImportError as e:
        print(f"Skipping runtime plot (matplotlib not available): {e}")


if __name__ == "__main__":
    main()
