"""
Seven-configuration MountainSort5 benchmark.

Evaluates the optimization × determinism configuration matrix:

  #1  original               Original MountainSort5 (NumPy, CPU)
  #2  not_opt_no_det         TorchIsosplit6MountainSort5, GPU, no determinism
  #3  opt_no_det             MountainSort5 (C++ isosplit6), GPU, no determinism
  #4  not_opt_relaxed_det    TorchIsosplit6MountainSort5, GPU, relaxed determinism
  #5  opt_relaxed_det        MountainSort5 (C++ isosplit6), GPU, relaxed determinism
  #6  not_opt_full_det       TorchIsosplit6MountainSort5, GPU, full determinism
  #7  opt_full_det           MountainSort5 (C++ isosplit6), GPU, full determinism

For each configuration the script:
  - Runs N repetitions, timing four pipeline stages and total wall-clock.
  - Records resource usage (CPU %, RSS, GPU memory, threads, BLAS info).
  - Assesses run-to-run determinism via the fidelity module.
  - Assesses port parity against the original (#1).
  - Produces a JSON report and a horizontal stacked bar plot.
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

from torched_mountainsort5.determinism import set_determinism, DeterminismMode

from fidelity import (
    assess_determinism,
    assess_port_parity,
    print_determinism_report,
    print_port_parity_report,
)

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False


# ---------------------------------------------------------------------------
# Configuration matrix
# ---------------------------------------------------------------------------

STAGE_NAMES = ["A_detection", "B_clustering", "C_alignment", "D_postprocessing"]

ALL_CONFIGS = [
    "original",
    "not_opt_no_det",
    "opt_no_det",
    "not_opt_relaxed_det",
    "opt_relaxed_det",
    "not_opt_full_det",
    "opt_full_det",
]

CONFIG_LABELS: Dict[str, str] = {
    "original":             "#1 Original MountainSort5",
    "not_opt_no_det":       "#2 Not optimized, not deterministic",
    "opt_no_det":           "#3 Optimized, not deterministic",
    "not_opt_relaxed_det":  "#4 Not optimized, relaxed determinism",
    "opt_relaxed_det":      "#5 Optimized, relaxed determinism",
    "not_opt_full_det":     "#6 Not optimized, full determinism",
    "opt_full_det":         "#7 Optimized, full determinism",
}

# Short labels for the bar plot y-axis.
CONFIG_SHORT_LABELS: Dict[str, str] = {
    "original":             "#1 original",
    "not_opt_no_det":       "#2 not opt, no det",
    "opt_no_det":           "#3 opt, no det",
    "not_opt_relaxed_det":  "#4 not opt, relaxed det",
    "opt_relaxed_det":      "#5 opt, relaxed det",
    "not_opt_full_det":     "#6 not opt, full det",
    "opt_full_det":         "#7 opt, full det",
}

# Map each config to its properties.
CONFIG_PROPERTIES: Dict[str, dict] = {
    "original":             {"pipeline": "original",       "device": "cpu",  "determinism": "none"},
    "not_opt_no_det":       {"pipeline": "torch_isosplit", "device": "cuda", "determinism": "none"},
    "opt_no_det":           {"pipeline": "cpp_isosplit",   "device": "cuda", "determinism": "none"},
    "not_opt_relaxed_det":  {"pipeline": "torch_isosplit", "device": "cuda", "determinism": "relaxed"},
    "opt_relaxed_det":      {"pipeline": "cpp_isosplit",   "device": "cuda", "determinism": "relaxed"},
    "not_opt_full_det":     {"pipeline": "torch_isosplit", "device": "cuda", "determinism": "full"},
    "opt_full_det":         {"pipeline": "cpp_isosplit",   "device": "cuda", "determinism": "full"},
}


# ---------------------------------------------------------------------------
# Benchmark parameters
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
    configs: List[str] = field(default_factory=lambda: list(ALL_CONFIGS))
    scheme1_params: ms5.Scheme1SortingParameters = field(default_factory=lambda: ms5.Scheme1SortingParameters(
        detect_channel_radius=150,
        detect_threshold=5.5,
        snippet_mask_radius=150,
        skip_alignment=False,
    ))


# ---------------------------------------------------------------------------
# Resource & timer helpers (reused from benchmark_mountainsorters.py)
# ---------------------------------------------------------------------------
@dataclass
class StageResources:
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
    def __init__(self, use_cuda_sync: bool = False):
        self.use_cuda_sync = use_cuda_sync
        self.elapsed: float = 0.0
        self.resources = StageResources()

    def __enter__(self):
        proc = psutil.Process()
        self.resources.rss_before_mb = proc.memory_info().rss / 1e6
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
# Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_and_preprocess(cfg: BenchmarkConfig) -> Tuple[si.BaseRecording, np.ndarray, np.ndarray]:
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

    chan_range = range(0, 384)
    channels_to_keep = recording.get_channel_ids()[chan_range]
    print(f"Slicing recording to {len(channels_to_keep)} channels for rapid testing...")
    recording_sliced = recording.select_channels(channel_ids=channels_to_keep)

    print("Preprocessing (bandpass + whiten) ...")
    rec_filtered = spre.bandpass_filter(recording_sliced, freq_min=300, freq_max=6000, dtype="float32")
    rec_preprocessed = spre.whiten(rec_filtered, seed=42)

    print("Materialising preprocessed traces into RAM ...")
    traces = rec_preprocessed.get_traces().astype(np.float32, copy=False)
    channel_locations = rec_preprocessed.get_channel_locations()
    print(f"  traces shape: {traces.shape}  ({traces.nbytes / 1e6:.1f} MB)")

    return rec_preprocessed, traces, channel_locations


# ---------------------------------------------------------------------------
# Runner: Original MountainSort5
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

    m = StageMonitor()
    with m:
        times, channel_indices = detect_spikes(
            traces=traces, channel_locations=channel_locations,
            time_radius=time_radius, channel_radius=params.detect_channel_radius,
            detect_threshold=params.detect_threshold, detect_sign=params.detect_sign,
            margin_left=params.snippet_T1, margin_right=params.snippet_T2, verbose=False,
        )
        times, channel_indices = remove_duplicate_times(times, channel_indices)
        snippets = extract_snippets(
            traces=traces, channel_locations=channel_locations,
            mask_radius=params.snippet_mask_radius, times=times,
            channel_indices=channel_indices, T1=params.snippet_T1, T2=params.snippet_T2,
        )
    timings.A_detection = m.elapsed
    resources.A_detection = m.resources
    L, T = snippets.shape[0], snippets.shape[1]

    m = StageMonitor()
    with m:
        features = compute_pca_features(snippets.reshape((L, T * M)), npca=npca)
        labels = isosplit6_subdivision_method(X=features, npca_per_subdivision=params.npca_per_subdivision)
        K = int(np.max(labels)) if len(labels) > 0 else 0
        templates_ = compute_templates(snippets=snippets, labels=labels)
        peak_channel_indices = [int(np.argmin(np.min(templates_[i], axis=0))) for i in range(K)]
    timings.B_clustering = m.elapsed
    resources.B_clustering = m.resources

    m = StageMonitor()
    with m:
        if not params.skip_alignment:
            offsets_ = align_templates(templates_)
            snippets = align_snippets(snippets, offsets_, labels)
            times = offset_times(times, -offsets_, labels)
            features = compute_pca_features(snippets.reshape((L, T * M)), npca=npca)
            labels = isosplit6_subdivision_method(X=features, npca_per_subdivision=params.npca_per_subdivision)
            K = int(np.max(labels)) if len(labels) > 0 else 0
            templates_ = compute_templates(snippets=snippets, labels=labels)
            peak_channel_indices = [int(np.argmin(np.min(templates_[i], axis=0))) for i in range(K)]
            offsets_to_peak = determine_offsets_to_peak(templates_, detect_sign=params.detect_sign, T1=params.snippet_T1)
            times = offset_times(times, offsets_to_peak, labels)
    timings.C_alignment = m.elapsed
    resources.C_alignment = m.resources

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
# Runner: Torched pipelines (CPU or GPU)
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

    m = StageMonitor(use_cuda_sync)
    with m:
        batch = model.detect_spikes(batch)
        batch = model.remove_duplicates(batch)
        batch = model.extract_snippets(batch)
    timings.A_detection = m.elapsed
    resources.A_detection = m.resources

    m = StageMonitor(use_cuda_sync)
    with m:
        batch = model.compute_pca(batch)
        batch = model.clustering(batch)
        batch = model.compute_templates(batch)
    timings.B_clustering = m.elapsed
    resources.B_clustering = m.resources

    m = StageMonitor(use_cuda_sync)
    with m:
        if not torched_params.skip_alignment:
            batch = model.align_templates(batch)
            batch = model.align_snippets(batch)
            batch.features = None
            batch = model.compute_pca(batch)
            batch = model.clustering(batch)
            batch = model.compute_templates(batch)
            batch = model.offset_times_to_peak(batch)
    timings.C_alignment = m.elapsed
    resources.C_alignment = m.resources

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
def save_outputs(results_dir: Path, config_name: str, run_idx: int,
                 times: np.ndarray, labels: np.ndarray):
    out_dir = results_dir / config_name / f"run_{run_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "spike_times.npy", times)
    np.save(out_dir / "labels.npy", labels)


def run_config(
    config_name: str,
    cfg: BenchmarkConfig,
    traces: np.ndarray,
    channel_locations: np.ndarray,
    sampling_frequency: float,
) -> Tuple[List[RunTimings], List[RunResources], List[np.ndarray], List[np.ndarray]]:
    props = CONFIG_PROPERTIES[config_name]
    pipeline = props["pipeline"]
    device = props["device"]
    determinism: DeterminismMode = props["determinism"]

    all_timings: List[RunTimings] = []
    all_resources: List[RunResources] = []
    all_times: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    # CUDA warm-up for GPU configs.
    if device == "cuda" and pipeline != "original":
        use_torch_iso = pipeline == "torch_isosplit"
        print("  CUDA warm-up run ...")
        set_determinism("none")
        run_torched(traces, channel_locations, sampling_frequency,
                    cfg.scheme1_params, device, use_torch_iso=use_torch_iso)
        torch.cuda.empty_cache()

    for i in range(cfg.n_runs):
        # Apply determinism mode BEFORE each run (re-seeds, resets flags).
        set_determinism(determinism)
        np.random.seed(42)

        print(f"  Run {i + 1}/{cfg.n_runs} ... ", end="", flush=True)

        if pipeline == "original":
            timings, res, t_out, l_out = run_original(
                traces, channel_locations, sampling_frequency, cfg.scheme1_params,
            )
        else:
            use_torch_iso = pipeline == "torch_isosplit"
            timings, res, t_out, l_out = run_torched(
                traces, channel_locations, sampling_frequency,
                cfg.scheme1_params, device, use_torch_iso=use_torch_iso,
            )

        print(f"total={timings.total:.3f}s  "
              f"(A={timings.A_detection:.3f}  B={timings.B_clustering:.3f}  "
              f"C={timings.C_alignment:.3f}  D={timings.D_postprocessing:.3f})")

        save_outputs(cfg.results_dir, config_name, i, t_out, l_out)
        all_timings.append(timings)
        all_resources.append(res)
        all_times.append(t_out)
        all_labels.append(l_out)

    return all_timings, all_resources, all_times, all_labels


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary_table(results: Dict[str, List[RunTimings]]):
    label_width = max((len(CONFIG_LABELS[c]) for c in results), default=18)
    label_width = max(label_width, len("Configuration"))
    total_width = label_width + 2 + (16 * (len(STAGE_NAMES) + 1))

    print("\n" + "=" * total_width)
    print("PERFORMANCE SUMMARY (seconds)")
    print("=" * total_width)
    header = f"{'Configuration':<{label_width}}"
    for stage in STAGE_NAMES + ["Total"]:
        header += f"{'':>2}{stage:>14}"
    print(header)
    print("-" * total_width)

    for config, timings_list in results.items():
        if not timings_list:
            continue
        vals = {s: [] for s in STAGE_NAMES + ["Total"]}
        for t in timings_list:
            vals["A_detection"].append(t.A_detection)
            vals["B_clustering"].append(t.B_clustering)
            vals["C_alignment"].append(t.C_alignment)
            vals["D_postprocessing"].append(t.D_postprocessing)
            vals["Total"].append(t.total)

        row = f"{CONFIG_LABELS[config]:<{label_width}}"
        for stage in STAGE_NAMES + ["Total"]:
            mean = np.mean(vals[stage])
            std = np.std(vals[stage])
            row += f"  {mean:>6.3f}+/-{std:<5.3f}"
        print(row)

    print("=" * total_width)


def print_resource_table(results: Dict[str, List[RunResources]]):
    print("\n" + "=" * 120)
    print("RESOURCE USAGE (per stage, averaged over runs)")
    print("=" * 120)

    for config, resources_list in results.items():
        if not resources_list:
            continue
        label = CONFIG_LABELS[config]
        print(f"\n  {label}")
        print(f"  {'Stage':<20} {'CPU%':>7} {'Threads':>8} {'Torch Thr':>10} "
              f"{'BLAS Thr':>9} {'RSS Δ MB':>9} {'GPU Alloc MB':>13} {'GPU Resv MB':>12}")
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
    determinism_reports: Dict[str, dict],
    parity_reports: List[dict],
):
    report = {
        "n_runs": cfg.n_runs,
        "configurations": {},
        "determinism": determinism_reports,
        "port_parity": parity_reports,
    }
    for config, timings_list in timing_results.items():
        resources_list = resource_results.get(config, [])
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
        report["configurations"][config] = runs

    out_path = cfg.results_dir / "configurations_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to {out_path}")


# ---------------------------------------------------------------------------
# Plotting (reuses stacked bar layout)
# ---------------------------------------------------------------------------
def plot_stacked_runtime(
    timing_results: Dict[str, List[RunTimings]],
    out_path: Path,
    baseline: str = "original",
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Skipping plot (matplotlib not available).")
        return

    STAGE_COLORS = {
        "A_detection":      "#8FD3B8",
        "B_clustering":     "#9BBCE0",
        "C_alignment":      "#E6A3BD",
        "D_postprocessing": "#C7BCE6",
    }
    STAGE_DISPLAY = {
        "A_detection":      "Detection",
        "B_clustering":     "Clustering",
        "C_alignment":      "Alignment",
        "D_postprocessing": "Postproc.",
    }

    ordered = [c for c in ALL_CONFIGS if c in timing_results and timing_results[c]]
    if not ordered:
        return

    def _mean_stages(tl):
        return {s: float(np.mean([getattr(t, s) for t in tl])) for s in STAGE_NAMES}

    stage_means = {c: _mean_stages(timing_results[c]) for c in ordered}
    totals = {c: sum(stage_means[c].values()) for c in ordered}

    if baseline not in totals:
        baseline = ordered[0]
    baseline_total = totals[baseline]

    n_rows = len(ordered)
    y_positions = np.arange(n_rows)[::-1]

    fig_w = 6.5
    fig_h = max(1.8, 0.38 * n_rows + 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    max_total = max(totals.values())

    for row_idx, config in enumerate(ordered):
        stages = stage_means[config]
        left = 0.0
        y = y_positions[row_idx]
        for stage in STAGE_NAMES:
            width = stages[stage]
            ax.barh(y, width, left=left, height=0.5, color=STAGE_COLORS[stage])
            left += width
        norm = totals[config] / baseline_total
        ax.text(left + max_total * 0.01, y, f"{norm:.2f}x", va="center", ha="left", fontsize=7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([CONFIG_SHORT_LABELS.get(c, c) for c in ordered], fontsize=7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Runtime (seconds)", fontsize=8)
    ax.set_xticks([])
    ax.set_xlim(0, max_total * 1.18)

    legend_patches = [
        mpatches.Patch(color=STAGE_COLORS[s], label=STAGE_DISPLAY[s])
        for s in STAGE_NAMES
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), ncol=1, frameon=True,
              handlelength=0.8, handleheight=0.8, labelspacing=0.2,
              columnspacing=0.6, borderpad=0.3)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Runtime plot saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------
def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Seven-configuration MountainSort5 benchmark")
    parser.add_argument("-n", "--n-runs", type=int, default=3,
                        help="Number of timed runs per configuration")
    parser.add_argument("--npx-bin", type=Path, help="Path to raw .bin recording")
    parser.add_argument("--chan-map", type=Path, help="Path to chanMap.mat")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Output directory")
    parser.add_argument("--configs", nargs="+", default=ALL_CONFIGS,
                        choices=ALL_CONFIGS,
                        help="Which configurations to benchmark")
    args = parser.parse_args()

    cfg = BenchmarkConfig(n_runs=args.n_runs, configs=args.configs,
                          results_dir=args.results_dir)
    if args.npx_bin:
        cfg.npx_bin_path = args.npx_bin
    if args.chan_map:
        cfg.chan_map_path = args.chan_map
    return cfg


def main():
    cfg = parse_args()

    # Validate: drop GPU configs if CUDA unavailable.
    gpu_configs = {c for c, p in CONFIG_PROPERTIES.items() if p["device"] == "cuda"}
    torch_configs = {c for c in ALL_CONFIGS if c != "original"}

    if gpu_configs & set(cfg.configs) and not HAS_CUDA:
        print("WARNING: GPU configs requested but CUDA not available. Removing.")
        cfg.configs = [c for c in cfg.configs if c not in gpu_configs]
    if torch_configs & set(cfg.configs) and not HAS_TORCH:
        print("WARNING: Torch configs requested but torch not installed. Removing.")
        cfg.configs = [c for c in cfg.configs if c not in torch_configs]

    if not cfg.configs:
        print("No valid configurations remaining. Exiting.")
        return

    # --- Preprocess once ---
    rec_preprocessed, traces_master, channel_locations = load_and_preprocess(cfg)
    sampling_frequency = rec_preprocessed.sampling_frequency

    # --- Run each configuration ---
    timing_results: Dict[str, List[RunTimings]] = {}
    resource_results: Dict[str, List[RunResources]] = {}
    output_results: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}

    for config in cfg.configs:
        label = CONFIG_LABELS[config]
        print(f"\n{'=' * 70}")
        print(f"  CONFIG: {label}  [{config}]  ({cfg.n_runs} runs)")
        print(f"{'=' * 70}")

        if config in torch_configs and HAS_CUDA:
            torch.cuda.empty_cache()

        timings, resources, all_times, all_labels = run_config(
            config, cfg, traces_master, channel_locations, sampling_frequency,
        )
        timing_results[config] = timings
        resource_results[config] = resources
        output_results[config] = (all_times, all_labels)

    # --- Fidelity analysis ---
    print("\n" + "=" * 90)
    print("DETERMINISM ANALYSIS")
    print("=" * 90)

    determinism_reports = {}
    for config in cfg.configs:
        all_t, all_l = output_results[config]
        report = assess_determinism(CONFIG_LABELS[config], all_t, all_l)
        print_determinism_report(report)
        determinism_reports[config] = report.to_dict()

    print("\n" + "=" * 90)
    print("PORT PARITY (vs Original)")
    print("=" * 90)

    parity_reports = []
    if "original" in output_results:
        ref_t, ref_l = output_results["original"][0][0], output_results["original"][1][0]
        for config in cfg.configs:
            if config == "original":
                continue
            ported_t, ported_l = output_results[config][0][0], output_results[config][1][0]
            report = assess_port_parity(
                CONFIG_LABELS["original"], ref_t, ref_l,
                CONFIG_LABELS[config], ported_t, ported_l,
            )
            print_port_parity_report(report)
            parity_reports.append(report.to_dict())

    # --- Reports ---
    print_summary_table(timing_results)
    print_resource_table(resource_results)
    save_json_report(cfg, timing_results, resource_results,
                     determinism_reports, parity_reports)

    # --- Plot ---
    plot_stacked_runtime(timing_results, cfg.results_dir / "configurations_runtime.pdf")

    # Reset determinism to default at the end.
    set_determinism("none")


if __name__ == "__main__":
    main()
