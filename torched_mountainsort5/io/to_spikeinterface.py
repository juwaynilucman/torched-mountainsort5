import spikeinterface as si
from packaging import version

from ..schema import SortingBatch


def to_spikeinterface(batch: SortingBatch) -> si.BaseSorting:
    """Convert a completed SortingBatch into a SpikeInterface sorting object.

    This is the single output boundary where torch->numpy conversion happens.
    """
    assert batch.times is not None and batch.labels is not None
    assert batch.sampling_frequency is not None

    times_np = batch.times.cpu().numpy()
    labels_np = batch.labels.cpu().numpy()

    if version.parse(si.__version__) < version.parse("0.102.2"):
        return si.NumpySorting.from_times_labels(
            [times_np], [labels_np], sampling_frequency=batch.sampling_frequency
        )
    else:
        return si.NumpySorting.from_samples_and_labels(
            [times_np], [labels_np], sampling_frequency=batch.sampling_frequency
        )
