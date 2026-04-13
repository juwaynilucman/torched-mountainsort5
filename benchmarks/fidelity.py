"""Fidelity and determinism measurement utilities.

Provides functions to quantify:
  1. **Port parity** — how closely a ported configuration reproduces the
     original algorithm's outputs (spike times and unit labels).
  2. **Run-to-run determinism** — whether repeated runs of the *same*
     configuration produce identical outputs.

Metrics
-------
- Spike-time Jaccard index: |intersection| / |union| of detected spike times.
- Spike-time exact match: whether times arrays are identical.
- Label agreement: fraction of shared spikes whose unit labels agree
  (after optimal relabeling via the Hungarian algorithm).
- Determinism verdict: ``"deterministic"`` (100 % match across all runs),
  ``"relaxed-deterministic"`` (final outputs match but intermediates may not),
  or ``"non-deterministic"`` (outputs vary across runs).
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Per-pair comparison ──────────────────────────────────────────────────


@dataclass
class PairwiseResult:
    """Result of comparing two (times, labels) arrays."""
    n_spikes_a: int
    n_spikes_b: int
    n_shared_times: int
    jaccard_times: float
    times_identical: bool
    label_agreement: float  # on shared spikes, after relabeling
    labels_identical: bool

    def to_dict(self) -> dict:
        return asdict(self)


def compare_outputs(
    times_a: np.ndarray,
    labels_a: np.ndarray,
    times_b: np.ndarray,
    labels_b: np.ndarray,
) -> PairwiseResult:
    """Compare two sets of spike-sorting outputs.

    Parameters
    ----------
    times_a, labels_a : ndarray
        Spike times and unit labels from run A.
    times_b, labels_b : ndarray
        Spike times and unit labels from run B.

    Returns
    -------
    PairwiseResult
        Detailed agreement metrics.
    """
    times_identical = np.array_equal(times_a, times_b)
    labels_identical = times_identical and np.array_equal(labels_a, labels_b)

    shared = np.intersect1d(times_a, times_b)
    union = np.union1d(times_a, times_b)
    n_shared = len(shared)
    jaccard = n_shared / max(len(union), 1)

    # Label agreement on shared spikes (after optimal relabeling).
    if n_shared > 0 and HAS_SCIPY:
        label_agr = _label_agreement_on_shared(
            times_a, labels_a, times_b, labels_b, shared,
        )
    elif n_shared > 0:
        # Fallback: raw agreement without relabeling.
        idx_a = np.isin(times_a, shared)
        idx_b = np.isin(times_b, shared)
        label_agr = float(np.mean(labels_a[idx_a] == labels_b[idx_b]))
    else:
        label_agr = 0.0

    return PairwiseResult(
        n_spikes_a=len(times_a),
        n_spikes_b=len(times_b),
        n_shared_times=n_shared,
        jaccard_times=jaccard,
        times_identical=times_identical,
        label_agreement=label_agr,
        labels_identical=labels_identical,
    )


def _label_agreement_on_shared(
    times_a: np.ndarray,
    labels_a: np.ndarray,
    times_b: np.ndarray,
    labels_b: np.ndarray,
    shared_times: np.ndarray,
) -> float:
    """Compute label agreement on shared spikes, with Hungarian relabeling.

    Spike times may contain duplicates (different channels at the same time),
    so we align on sorted shared times via index-based lookup.
    """
    # Build masks for shared spikes.
    mask_a = np.isin(times_a, shared_times)
    mask_b = np.isin(times_b, shared_times)
    la = labels_a[mask_a]
    lb = labels_b[mask_b]

    # If lengths differ after masking (duplicate times), truncate to min.
    n = min(len(la), len(lb))
    la, lb = la[:n], lb[:n]

    if n == 0:
        return 0.0

    # Build confusion matrix and solve optimal label mapping.
    K_a = int(la.max()) + 1
    K_b = int(lb.max()) + 1
    confusion = np.zeros((K_a, K_b), dtype=np.int64)
    for i in range(n):
        confusion[la[i], lb[i]] += 1

    # Hungarian algorithm minimises cost; we want to maximise agreement.
    row_ind, col_ind = linear_sum_assignment(-confusion)

    # Map B's labels to A's label space.
    remap = np.zeros(K_b, dtype=np.int64)
    for r, c in zip(row_ind, col_ind):
        remap[c] = r
    lb_remapped = remap[lb]

    return float(np.mean(la == lb_remapped))


# ── Multi-run determinism check ──────────────────────────────────────────


DeterminismVerdict = Literal["deterministic", "non-deterministic"]


@dataclass
class DeterminismReport:
    """Aggregated determinism assessment for N runs of one configuration."""
    config_name: str
    n_runs: int
    all_pairwise: List[PairwiseResult] = field(default_factory=list)
    verdict: DeterminismVerdict = "non-deterministic"
    min_jaccard: float = 0.0
    max_jaccard: float = 0.0
    mean_jaccard: float = 0.0
    min_label_agreement: float = 0.0
    max_label_agreement: float = 0.0
    mean_label_agreement: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["all_pairwise"] = [p.to_dict() for p in self.all_pairwise]
        return d


def assess_determinism(
    config_name: str,
    all_times: List[np.ndarray],
    all_labels: List[np.ndarray],
) -> DeterminismReport:
    """Assess run-to-run determinism for one configuration.

    Compares every pair of runs (i, j) with i < j and produces an
    aggregate verdict.

    Parameters
    ----------
    config_name : str
        Human-readable name for the configuration.
    all_times : list of ndarray
        Spike times from each run.
    all_labels : list of ndarray
        Unit labels from each run.

    Returns
    -------
    DeterminismReport
    """
    n = len(all_times)
    report = DeterminismReport(config_name=config_name, n_runs=n)

    if n < 2:
        report.verdict = "deterministic"
        return report

    pairwise: List[PairwiseResult] = []
    for i in range(n):
        for j in range(i + 1, n):
            pw = compare_outputs(
                all_times[i], all_labels[i],
                all_times[j], all_labels[j],
            )
            pairwise.append(pw)

    report.all_pairwise = pairwise

    jaccards = [p.jaccard_times for p in pairwise]
    label_agrs = [p.label_agreement for p in pairwise]

    report.min_jaccard = float(np.min(jaccards))
    report.max_jaccard = float(np.max(jaccards))
    report.mean_jaccard = float(np.mean(jaccards))
    report.min_label_agreement = float(np.min(label_agrs))
    report.max_label_agreement = float(np.max(label_agrs))
    report.mean_label_agreement = float(np.mean(label_agrs))

    # Verdict: deterministic iff ALL pairs have identical times and labels.
    all_identical = all(p.times_identical and p.labels_identical for p in pairwise)
    report.verdict = "deterministic" if all_identical else "non-deterministic"

    return report


# ── Port parity (reference vs ported) ────────────────────────────────────


@dataclass
class PortParityReport:
    """How closely a ported configuration matches the reference (original)."""
    reference_name: str
    ported_name: str
    comparison: Optional[PairwiseResult] = None

    def to_dict(self) -> dict:
        d = {
            "reference_name": self.reference_name,
            "ported_name": self.ported_name,
        }
        if self.comparison is not None:
            d["comparison"] = self.comparison.to_dict()
        return d


def assess_port_parity(
    reference_name: str,
    ref_times: np.ndarray,
    ref_labels: np.ndarray,
    ported_name: str,
    ported_times: np.ndarray,
    ported_labels: np.ndarray,
) -> PortParityReport:
    """Compare a ported configuration's first run against the reference."""
    result = compare_outputs(ref_times, ref_labels, ported_times, ported_labels)
    return PortParityReport(
        reference_name=reference_name,
        ported_name=ported_name,
        comparison=result,
    )


# ── Pretty-printing ─────────────────────────────────────────────────────


def print_determinism_report(report: DeterminismReport) -> None:
    """Print a human-readable determinism report."""
    print(f"\n  {report.config_name}  ({report.n_runs} runs)")
    print(f"    Verdict: {report.verdict.upper()}")
    if report.n_runs >= 2:
        print(f"    Jaccard (times):    min={report.min_jaccard:.6f}  "
              f"mean={report.mean_jaccard:.6f}  max={report.max_jaccard:.6f}")
        print(f"    Label agreement:    min={report.min_label_agreement:.6f}  "
              f"mean={report.mean_label_agreement:.6f}  max={report.max_label_agreement:.6f}")


def print_port_parity_report(report: PortParityReport) -> None:
    """Print a human-readable port parity report."""
    c = report.comparison
    if c is None:
        print(f"\n  {report.reference_name} vs {report.ported_name}: NO DATA")
        return
    print(f"\n  {report.reference_name} vs {report.ported_name}")
    print(f"    Spikes: {c.n_spikes_a} vs {c.n_spikes_b}")
    print(f"    Times identical: {c.times_identical}")
    print(f"    Jaccard (times): {c.jaccard_times:.6f}")
    print(f"    Label agreement: {c.label_agreement:.6f}")
    print(f"    Labels identical: {c.labels_identical}")
