"""Horizontal stacked bar plot of per-stage runtime for each benchmark target.

Each row is one target (original, cpp_iso_cpu, ...), each segment is one of the
four benchmark stages (A_detection, B_clustering, C_alignment, D_postprocessing).
Bar widths are mean seconds across runs; the label to the right of each bar
shows total runtime normalized to the baseline target (default: "original").
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Short row labels keep the y-axis compact.
SHORT_LABELS: Dict[str, str] = {
    "original":      "original",
    "cpp_iso_cpu":   "cpp iso, cpu",
    "cpp_iso_gpu":   "cpp iso, gpu",
    "torch_iso_cpu": "torch iso, cpu",
    "torch_iso_gpu": "torch iso, gpu",
}

# Four colors picked from the reference palette so stages read clearly.
STAGE_COLORS: Dict[str, str] = {
    "A_detection":      "#8FD3B8",  # teal-green
    "B_clustering":     "#9BBCE0",  # blue
    "C_alignment":      "#E6A3BD",  # pink
    "D_postprocessing": "#C7BCE6",  # purple
}

STAGE_DISPLAY: Dict[str, str] = {
    "A_detection":      "Detection",
    "B_clustering":     "Clustering",
    "C_alignment":      "Alignment",
    "D_postprocessing": "Postproc.",
}

STAGE_ORDER: List[str] = [
    "A_detection",
    "B_clustering",
    "C_alignment",
    "D_postprocessing",
]


def _mean_stages(timings_list) -> Dict[str, float]:
    """Average per-stage elapsed seconds across all runs of one target."""
    return {
        "A_detection":      float(np.mean([t.A_detection for t in timings_list])),
        "B_clustering":     float(np.mean([t.B_clustering for t in timings_list])),
        "C_alignment":      float(np.mean([t.C_alignment for t in timings_list])),
        "D_postprocessing": float(np.mean([t.D_postprocessing for t in timings_list])),
    }


def plot_stacked_runtime(
    timing_results: Dict[str, List],
    target_labels: Dict[str, str],
    out_path: Path,
    baseline: str = "original",
) -> None:
    """Render a horizontal stacked bar plot of benchmark runtimes.

    Parameters
    ----------
    timing_results : dict[str, list[RunTimings]]
        Mapping from target name to its list of per-run timings, as built in
        ``benchmark_mountainsorters.main``.
    target_labels : dict[str, str]
        Canonical long labels for each target (unused for display; kept for
        parity with the caller and potential future use).
    out_path : Path
        Destination PDF.
    baseline : str
        Target key whose total runtime defines the 1.00x reference.
    """
    # Preserve the canonical ordering from SHORT_LABELS, but drop any target
    # the caller did not actually benchmark.
    ordered_targets = [t for t in SHORT_LABELS if t in timing_results and timing_results[t]]
    if not ordered_targets:
        print("plot_stacked_runtime: nothing to plot (no timing results).")
        return

    stage_means = {t: _mean_stages(timing_results[t]) for t in ordered_targets}
    totals = {t: sum(stage_means[t].values()) for t in ordered_targets}

    if baseline not in totals:
        baseline = ordered_targets[0]
        print(f"plot_stacked_runtime: baseline missing, falling back to '{baseline}'.")
    baseline_total = totals[baseline]

    n_rows = len(ordered_targets)
    bar_h = 0.5
    y_positions = np.arange(n_rows)[::-1]  # first target on top

    fig_w = 5.2
    fig_h = max(1.4, 0.32 * n_rows + 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    max_total = max(totals.values())

    for row_idx, target in enumerate(ordered_targets):
        stages = stage_means[target]
        left = 0.0
        y = y_positions[row_idx]
        for stage in STAGE_ORDER:
            width = stages[stage]
            ax.barh(y, width, left=left, height=bar_h, color=STAGE_COLORS[stage])
            left += width

        norm = totals[target] / baseline_total
        ax.text(
            left + max_total * 0.01, y, f"{norm:.2f}x",
            va="center", ha="left", fontsize=7,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [SHORT_LABELS.get(t, t) for t in ordered_targets],
        fontsize=7,
    )

    ax.set_axisbelow(True)
    ax.set_xlabel("Runtime (seconds)", fontsize=8)
    ax.set_xticks([])

    # Give the "N.NNx" annotation room to breathe.
    ax.set_xlim(0, max_total * 1.15)

    legend_patches = [
        mpatches.Patch(color=STAGE_COLORS[s], label=STAGE_DISPLAY[s])
        for s in STAGE_ORDER
    ]
    ax.legend(
        handles=legend_patches,
        fontsize=7,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=1,
        frameon=True,
        handlelength=0.8,
        handleheight=0.8,
        labelspacing=0.2,
        columnspacing=0.6,
        borderpad=0.3,
    )

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Runtime plot saved to {out_path}")
