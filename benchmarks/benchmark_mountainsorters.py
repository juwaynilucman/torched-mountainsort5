"""
MountainSort5 benchmark.

Compares five execution paths at identical 4-stage boundaries:
  - Original MS5 (CPU):  low-level NumPy functions called directly
  - Torched MS5 (CPU):   nn.Module pipeline with device='cpu'
  - Torched MS5 (GPU):   nn.Module pipeline with device='cuda'
  - Optim MS5 (CPU):     nn.Module pipeline with PyTorch-native ISO-SPLIT, device='cpu'
  - Optim MS5 (GPU):     nn.Module pipeline with PyTorch-native ISO-SPLIT, device='cuda'

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
ALL_TARGETS = ["original_cpu", "torched_cpu", "torched_gpu", "optim_cpu", "optim_gpu"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    sampling_freq: int = 30000
    num_channels: int = 384
    dtype: str = "int16"
    npx_bin_path: Path = Path(r"C:\Users\juway\Documents\Marquees-smith\c46\subset_data\raw_2pct.bin")
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
# Timer helpers
# ---------------------------------------------------------------------------
class StageTimer:
    """Context-manager timer with optional CUDA synchronisation."""

    def __init__(self, use_cuda_sync: bool = False):
        self.use_cuda_sync = use_cuda_sync
        self.elapsed: float = 0.0

    def __enter__(self):
        if self.use_cuda_sync:
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.use_cuda_sync:
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start


@dataclass
class RunTimings:
    A_detection: float = 0.0
    B_clustering: float = 0.0
    C_alignment: float = 0.0
    D_postprocessing: float = 0.0

    @property
    def total(self) -> float:
        return self.A_detection + self.B_clustering + self.C_alignment + self.D_postprocessing


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
# Runner:  Original CPU  (calls low-level NumPy functions directly)
# ---------------------------------------------------------------------------
def run_original_cpu(
    traces_master: np.ndarray,
    channel_locations: np.ndarray,
    sampling_frequency: float,
    params: ms5.Scheme1SortingParameters,
) -> Tuple[RunTimings, np.ndarray, np.ndarray]:
    traces = np.copy(traces_master)
    N, M = traces.shape
    timings = RunTimings()
    time_radius = int(math.ceil(params.detect_time_radius_msec / 1000 * sampling_frequency))
    npca = params.npca_per_channel * M

    # --- Stage A: Detection & Extraction ---
    t = StageTimer()
    with t:
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
    timings.A_detection = t.elapsed
    L, T = snippets.shape[0], snippets.shape[1]

    # --- Stage B: Clustering (1st pass) ---
    t = StageTimer()
    with t:
        features = compute_pca_features(snippets.reshape((L, T * M)), npca=npca)
        labels = isosplit6_subdivision_method(X=features, npca_per_subdivision=params.npca_per_subdivision)
        K = int(np.max(labels)) if len(labels) > 0 else 0
        templates_ = compute_templates(snippets=snippets, labels=labels)
        peak_channel_indices = [int(np.argmin(np.min(templates_[i], axis=0))) for i in range(K)]
    timings.B_clustering = t.elapsed

    # --- Stage C: Alignment & Re-Clustering ---
    t = StageTimer()
    with t:
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
    timings.C_alignment = t.elapsed

    # --- Stage D: Post-processing ---
    t = StageTimer()
    with t:
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
    timings.D_postprocessing = t.elapsed

    return timings, np.asarray(times), np.asarray(labels)


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
    use_optim: bool = False,
) -> Tuple[RunTimings, np.ndarray, np.ndarray]:
    from torched_mountainsort5.schema import SortingBatch
    from torched_mountainsort5.mountainsort5 import MountainSort5
    from torched_mountainsort5.torch_clustering_mountainsort5 import TorchIsosplit6MountainSort5

    dev = torch.device(device)
    use_cuda_sync = dev.type == "cuda"

    torched_params = _make_torched_params(params)
    model_cls = TorchIsosplit6MountainSort5 if use_optim else MountainSort5
    model = model_cls(torched_params, sampling_frequency).to(dev)

    traces_t = torch.as_tensor(np.copy(traces_master), dtype=torch.float32, device=dev)
    chan_locs_t = torch.as_tensor(channel_locations.copy(), dtype=torch.float32, device=dev)

    batch = SortingBatch(
        traces=traces_t,
        channel_locations=chan_locs_t,
        sampling_frequency=sampling_frequency,
    )

    timings = RunTimings()

    # --- Stage A: Detection & Extraction ---
    t = StageTimer(use_cuda_sync)
    with t:
        batch = model.detect_spikes(batch)
        batch = model.remove_duplicates(batch)
        batch = model.extract_snippets(batch)
    timings.A_detection = t.elapsed

    # --- Stage B: Clustering (1st pass) ---
    t = StageTimer(use_cuda_sync)
    with t:
        batch = model.compute_pca(batch)
        batch = model.clustering(batch)
        batch = model.compute_templates(batch)
    timings.B_clustering = t.elapsed

    # --- Stage C: Alignment & Re-Clustering ---
    t = StageTimer(use_cuda_sync)
    with t:
        if not torched_params.skip_alignment:
            batch = model.align_templates(batch)
            batch = model.align_snippets(batch)

            batch.features = None  # clear stale features
            batch = model.compute_pca(batch)
            batch = model.clustering(batch)
            batch = model.compute_templates(batch)

            batch = model.offset_times_to_peak(batch)
    timings.C_alignment = t.elapsed

    # --- Stage D: Post-processing ---
    t = StageTimer(use_cuda_sync)
    with t:
        batch = model.sort_times(batch)
        batch = model.remove_out_of_bounds(batch)
        batch = model.reorder_units(batch)
    timings.D_postprocessing = t.elapsed

    out_times = batch.times.cpu().numpy()
    out_labels = batch.labels.cpu().numpy()
    return timings, out_times, out_labels


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
) -> Tuple[List[RunTimings], List[np.ndarray], List[np.ndarray]]:
    all_timings: List[RunTimings] = []
    all_times: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    # CUDA warm-up (one throw-away run)
    if target in ("torched_gpu", "optim_gpu"):
        use_optim = target == "optim_gpu"
        print("  CUDA warm-up run ...")
        run_torched(traces, channel_locations, sampling_frequency, cfg.scheme1_params, "cuda", use_optim=use_optim)
        torch.cuda.empty_cache()

    for i in range(cfg.n_runs):
        np.random.seed(42)
        if HAS_TORCH:
            torch.manual_seed(42)

        print(f"  Run {i + 1}/{cfg.n_runs} ... ", end="", flush=True)

        if target == "original_cpu":
            timings, t_out, l_out = run_original_cpu(traces, channel_locations, sampling_frequency, cfg.scheme1_params)
        elif target == "torched_cpu":
            timings, t_out, l_out = run_torched(traces, channel_locations, sampling_frequency, cfg.scheme1_params, "cpu")
        elif target == "torched_gpu":
            timings, t_out, l_out = run_torched(traces, channel_locations, sampling_frequency, cfg.scheme1_params, "cuda")
        elif target == "optim_cpu":
            timings, t_out, l_out = run_torched(traces, channel_locations, sampling_frequency, cfg.scheme1_params, "cpu", use_optim=True)
        elif target == "optim_gpu":
            timings, t_out, l_out = run_torched(traces, channel_locations, sampling_frequency, cfg.scheme1_params, "cuda", use_optim=True)
        else:
            raise ValueError(f"Unknown target: {target}")

        print(f"total={timings.total:.3f}s  (A={timings.A_detection:.3f}  B={timings.B_clustering:.3f}  C={timings.C_alignment:.3f}  D={timings.D_postprocessing:.3f})")

        save_outputs(cfg.results_dir, target, i, t_out, l_out)
        all_timings.append(timings)
        all_times.append(t_out)
        all_labels.append(l_out)

    return all_timings, all_times, all_labels


# ---------------------------------------------------------------------------
# Summary & Validation
# ---------------------------------------------------------------------------
def print_summary_table(results: Dict[str, List[RunTimings]]):
    print("\n" + "=" * 90)
    print("PERFORMANCE SUMMARY (seconds)")
    print("=" * 90)
    header = f"{'Target':<18}"
    for stage in STAGE_NAMES + ["Total"]:
        header += f"{'':>2}{stage:>14}"
    print(header)
    print("-" * 90)

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

        row = f"{target:<18}"
        for stage in STAGE_NAMES + ["Total"]:
            mean = np.mean(vals[stage])
            std = np.std(vals[stage])
            row += f"  {mean:>6.3f}+/-{std:<5.3f}"
        print(row)

    print("=" * 90)


def fidelity_check(
    results: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]],
):
    """Port parity (original_cpu vs torched_gpu) and determinism (run 0 vs run n-1)."""
    print("\n" + "=" * 90)
    print("FIDELITY CHECKS")
    print("=" * 90)

    # --- Port parity ---
    pairs = [
        ("original_cpu", "torched_cpu"),
        ("original_cpu", "torched_gpu"),
        ("original_cpu", "optim_cpu"),
        ("original_cpu", "optim_gpu"),
    ]
    for t1, t2 in pairs:
        if t1 not in results or t2 not in results:
            continue
        times_a, labels_a = results[t1][0][0], results[t1][1][0]  # run 0
        times_b, labels_b = results[t2][0][0], results[t2][1][0]

        times_match = np.array_equal(times_a, times_b)
        labels_match = np.array_equal(labels_a, labels_b)

        if times_match and labels_match:
            print(f"  {t1} vs {t2}:  IDENTICAL (times and labels)")
        else:
            common = np.intersect1d(times_a, times_b)
            frac = len(common) / max(len(times_a), 1)
            print(f"  {t1} vs {t2}:  spikes={len(times_a)} vs {len(times_b)}, "
                  f"shared_times={len(common)} ({frac:.4f}), labels_match={labels_match}")

    # --- Determinism (run 0 vs last run within each target) ---
    print()
    for target, (all_times, all_labels) in results.items():
        if len(all_times) < 2:
            print(f"  {target} determinism:  skipped (only 1 run)")
            continue
        t0, t_last = all_times[0], all_times[-1]
        l0, l_last = all_labels[0], all_labels[-1]
        if np.array_equal(t0, t_last) and np.array_equal(l0, l_last):
            print(f"  {target} determinism:  PASS (run 0 == run {len(all_times) - 1})")
        else:
            common = np.intersect1d(t0, t_last)
            frac = len(common) / max(len(t0), 1)
            print(f"  {target} determinism:  DRIFT  shared_times={len(common)}/{len(t0)} ({frac:.4f})")

    print("=" * 90)


def save_json_report(
    cfg: BenchmarkConfig,
    timing_results: Dict[str, List[RunTimings]],
):
    report = {"n_runs": cfg.n_runs, "targets": {}}
    for target, timings_list in timing_results.items():
        runs = []
        for t in timings_list:
            runs.append({
                "A_detection": t.A_detection,
                "B_clustering": t.B_clustering,
                "C_alignment": t.C_alignment,
                "D_postprocessing": t.D_postprocessing,
                "total": t.total,
            })
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
    gpu_targets = {"torched_gpu", "optim_gpu"}
    torch_targets = {"torched_cpu", "torched_gpu", "optim_cpu", "optim_gpu"}
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
    output_results: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}

    for target in cfg.targets:
        print(f"\n{'=' * 60}")
        print(f"  TARGET: {target}  ({cfg.n_runs} runs)")
        print(f"{'=' * 60}")

        if target in torch_targets:
            torch.cuda.empty_cache() if HAS_CUDA else None

        timings, all_times, all_labels = run_target(
            target, cfg, traces_master, channel_locations, sampling_frequency,
        )
        timing_results[target] = timings
        output_results[target] = (all_times, all_labels)

    # --- Report ---
    print_summary_table(timing_results)
    fidelity_check(output_results)
    save_json_report(cfg, timing_results)


if __name__ == "__main__":
    main()
