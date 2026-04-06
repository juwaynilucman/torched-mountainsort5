import torch
import spikeinterface as si

from ..schema import SortingBatch


def from_spikeinterface(
    recording: si.BaseRecording,
    *,
    device: torch.device = torch.device("cuda"),
) -> SortingBatch:
    """Convert a SpikeInterface recording into a SortingBatch on the target device.

    This is the single input boundary where numpy->torch conversion happens.
    """
    traces_np = recording.get_traces()
    channel_locations_np = recording.get_channel_locations()

    return SortingBatch(
        traces=torch.as_tensor(traces_np, dtype=torch.float32, device=device),
        channel_locations=torch.as_tensor(channel_locations_np, dtype=torch.float32, device=device),
        sampling_frequency=recording.sampling_frequency,
    )
