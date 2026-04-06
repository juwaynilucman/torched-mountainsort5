"""End-to-end validation: torched MountainSort5 vs original sorting_scheme1.

Runs both pipelines on the same synthetic recording and asserts that
the spike times and labels agree exactly.

Usage:
    pytest torched_mountainsort5/tests/test_pipeline.py -v -s
"""
import gc
import shutil
import tempfile

import numpy as np
import pytest
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

import mountainsort5 as ms5
from mountainsort5.util import create_cached_recording

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

from torched_mountainsort5 import MountainSort5, SortingBatch, SortingParameters
from torched_mountainsort5.io.from_spikeinterface import from_spikeinterface
from torched_mountainsort5.io.to_spikeinterface import to_spikeinterface


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

SORTING_PARAMS = ms5.Scheme1SortingParameters(snippet_mask_radius=50)

TORCHED_PARAMS = SortingParameters(
    detect_threshold=SORTING_PARAMS.detect_threshold,
    detect_channel_radius=SORTING_PARAMS.detect_channel_radius,
    detect_time_radius_msec=SORTING_PARAMS.detect_time_radius_msec,
    detect_sign=SORTING_PARAMS.detect_sign,
    snippet_T1=SORTING_PARAMS.snippet_T1,
    snippet_T2=SORTING_PARAMS.snippet_T2,
    snippet_mask_radius=SORTING_PARAMS.snippet_mask_radius,
    npca_per_channel=SORTING_PARAMS.npca_per_channel,
    npca_per_subdivision=SORTING_PARAMS.npca_per_subdivision,
    skip_alignment=SORTING_PARAMS.skip_alignment or False,
)


# ---------------------------------------------------------------------------
# Fixtures — run both pipelines ONCE, share results across all tests
# ---------------------------------------------------------------------------

def _seed_all():
    np.random.seed(42)
    if HAS_CUDA:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture(scope="module")
def preprocessed_recording():
    recording, _ = se.toy_example(
        duration=20,
        num_channels=8,
        num_units=16,
        sampling_frequency=30000,
        num_segments=1,
        seed=0,
    )
    rec_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000, dtype=np.float32)
    return spre.whiten(rec_filtered)


@pytest.fixture(scope="module")
def sorting_original(preprocessed_recording):
    _seed_all()
    tmpdir = tempfile.mkdtemp()
    try:
        recording_cached = create_cached_recording(preprocessed_recording, folder=tmpdir)
        sorting = ms5.sorting_scheme1(
            recording_cached,
            sorting_parameters=SORTING_PARAMS,
            use_gpu=True,
        )
        del recording_cached
        gc.collect()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return sorting


@pytest.fixture(scope="module")
def sorting_torched(preprocessed_recording):
    _seed_all()
    device = torch.device("cuda")
    batch = from_spikeinterface(preprocessed_recording, device=device)
    model = MountainSort5(TORCHED_PARAMS, sampling_frequency=preprocessed_recording.sampling_frequency)
    with torch.no_grad():
        batch = model(batch)
    return to_spikeinterface(batch)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_sorted_times(sorting):
    vector = sorting.to_spike_vector()
    return np.sort(vector["sample_index"])


def _extract_unit_spike_trains(sorting):
    return {
        uid: sorting.get_unit_spike_train(uid)
        for uid in sorting.get_unit_ids()
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
class TestTorchedVsOriginal:

    def test_global_spike_times_match(self, sorting_original, sorting_torched):
        """All clustered spike times (ignoring labels) must be identical."""
        times_orig = _extract_sorted_times(sorting_original)
        times_torch = _extract_sorted_times(sorting_torched)

        print(f"\nOriginal: {len(times_orig)} spikes, "
              f"Torched: {len(times_torch)} spikes")

        assert len(times_orig) == len(times_torch), (
            f"Spike count mismatch: original={len(times_orig)}, torched={len(times_torch)}"
        )
        np.testing.assert_array_equal(
            times_orig, times_torch,
            err_msg="Global spike times differ between original and torched pipelines",
        )

    def test_unit_count_matches(self, sorting_original, sorting_torched):
        """Both pipelines must find the same number of units."""
        n_orig = len(sorting_original.get_unit_ids())
        n_torch = len(sorting_torched.get_unit_ids())
        print(f"\nOriginal: {n_orig} units, Torched: {n_torch} units")

        assert n_orig == n_torch, (
            f"Unit count mismatch: original={n_orig}, torched={n_torch}"
        )

    def test_per_unit_spike_trains_match(self, sorting_original, sorting_torched):
        """Each unit's spike train must be identical across both pipelines."""
        trains_orig = _extract_unit_spike_trains(sorting_original)
        trains_torch = _extract_unit_spike_trains(sorting_torched)

        assert set(trains_orig.keys()) == set(trains_torch.keys()), (
            f"Unit IDs differ: original={set(trains_orig.keys())}, "
            f"torched={set(trains_torch.keys())}"
        )

        for uid in trains_orig:
            np.testing.assert_array_equal(
                trains_orig[uid],
                trains_torch[uid],
                err_msg=f"Spike train for unit {uid} differs",
            )
        print(f"\nAll {len(trains_orig)} unit spike trains match exactly.")
