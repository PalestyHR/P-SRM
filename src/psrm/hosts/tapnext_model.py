from pathlib import Path
from contextlib import nullcontext
import sys,io
import numpy as np
import torch
import mediapy as media
from PIL import Image
MODEL_SIZE=256
def decode_example(example: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    for encoded in example["video"]:
        with Image.open(io.BytesIO(encoded)) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    frames = np.stack(frames, axis=0)
    frames_model = (
        np.asarray(media.resize_video(frames, (MODEL_SIZE, MODEL_SIZE)), dtype=np.float32)
        / 255.0
        * 2.0
        - 1.0
    )
    points = np.asarray(example["points"], dtype=np.float32) * np.asarray(
        [MODEL_SIZE, MODEL_SIZE], dtype=np.float32
    )
    occluded = np.asarray(example["occluded"], dtype=bool)
    has_visible = (~occluded).any(axis=1)
    points = points[has_visible]
    occluded = occluded[has_visible]
    if not len(points):
        raise ValueError("example has no visible query")
    query_frames = np.asarray(
        [np.flatnonzero(~row)[0] for row in occluded], dtype=np.int64
    )
    query_points = np.stack(
        [
            query_frames.astype(np.float32),
            points[np.arange(len(points)), query_frames, 1],
            points[np.arange(len(points)), query_frames, 0],
        ],
        axis=-1,
    )
    return frames_model, points, occluded, query_points

def load_model(tapnet_root: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(tapnet_root.resolve()))
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    return TAPNextPP.from_checkpoint(
        checkpoint.resolve(), device=device, input_resolution=MODEL_SIZE
    )

def decode_track_logits(track_logits: torch.Tensor) -> torch.Tensor:
    logits_y, logits_x = track_logits.chunk(2, dim=-1)
    argmax_y = logits_y.argmax(dim=-1, keepdim=True)
    argmax_x = logits_x.argmax(dim=-1, keepdim=True)
    index = torch.arange(256, device=track_logits.device).repeat(
        *argmax_y.shape[:-1], 1
    )
    mask_y = (torch.abs(argmax_y - index) <= 20).float()
    mask_x = (torch.abs(argmax_x - index) <= 20).float()
    probability_y = torch.softmax(logits_y * 0.5, dim=-1) * mask_y
    probability_x = torch.softmax(logits_x * 0.5, dim=-1) * mask_x
    probability_y /= probability_y.sum(dim=-1, keepdim=True)
    probability_x /= probability_x.sum(dim=-1, keepdim=True)
    track_y = torch.sum(probability_y * index, dim=-1)[..., None]
    track_x = torch.sum(probability_x * index, dim=-1)[..., None]
    return torch.cat([track_y, track_x], dim=-1) + 0.5

def infer_sequence(
    model,
    frames_model: np.ndarray,
    query_points_tyx: np.ndarray,
    device: torch.device,
    autocast_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    video = torch.from_numpy(frames_model[None]).to(device).float()
    query = torch.from_numpy(query_points_tyx[None]).to(device).float()
    tracks_per_frame = []
    logits_per_frame = []
    visibility_per_frame = []
    state = None
    for frame_index in range(video.shape[1]):
        context = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if autocast_enabled
            else nullcontext()
        )
        with torch.no_grad(), context:
            if state is None:
                tracks, track_logits, visible_logits, state = model._model(
                    video=video[:, frame_index : frame_index + 1],
                    query_points=query,
                )
            else:
                tracks, track_logits, visible_logits, state = model._model(
                    video=video[:, frame_index : frame_index + 1], state=state
                )
            decoded = decode_track_logits(track_logits)
        if not torch.equal(decoded, tracks):
            difference = float((decoded - tracks).abs().max().item())
            raise RuntimeError(
                f"official coordinate decoder mismatch at frame {frame_index}: {difference}"
            )
        tracks_per_frame.append(tracks.cpu().float().numpy())
        logits_per_frame.append(track_logits.cpu().float().numpy())
        visibility_per_frame.append(visible_logits.cpu().float().numpy())

    tracks_yx = np.concatenate(tracks_per_frame, axis=1)[0]
    track_logits = np.concatenate(logits_per_frame, axis=1)[0]
    visible_logits = np.concatenate(visibility_per_frame, axis=1)[0, :, :, 0]
    tracks_xy = tracks_yx[..., ::-1].transpose(1, 0, 2).copy()
    track_logits = track_logits.transpose(1, 0, 2).copy()
    visible_logits = visible_logits.transpose(1, 0).copy()
    return tracks_xy, track_logits, visible_logits

def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, np.float64) * 0.5
    values -= values.max(axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=-1, keepdims=True)

def official_axis_distribution(logits: np.ndarray) -> np.ndarray:
    raw = np.asarray(logits, np.float64)
    peak = raw.argmax(axis=-1)
    index = np.arange(256, dtype=np.int64)
    mask = np.abs(index[None, :] - peak[:, None]) <= 20
    probability = stable_softmax(raw) * mask
    probability /= probability.sum(axis=-1, keepdims=True)
    return probability.astype(np.float32)

def sample_axis_candidate_centered(
    probability: np.ndarray, candidate_coordinate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count = probability.shape[0]
    offsets = np.arange(32, dtype=np.float32) - 15.5
    source_index = candidate_coordinate[:, None].astype(np.float32) + offsets - 0.5
    valid = (source_index >= 0.0) & (source_index <= 255.0)
    clipped = np.clip(source_index, 0.0, 255.0)
    lower = np.floor(clipped).astype(np.int64)
    upper = np.minimum(lower + 1, 255)
    weight = clipped - lower
    row = np.arange(count, dtype=np.int64)[:, None]
    sampled = probability[row, lower] * (1.0 - weight) + probability[row, upper] * weight
    sampled *= valid
    return sampled.astype(np.float32), valid

def candidate_centered_evidence(
    probability_x: np.ndarray,
    probability_y: np.ndarray,
    candidate_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_x, valid_x = sample_axis_candidate_centered(probability_x, candidate_xy[:, 0])
    local_y, valid_y = sample_axis_candidate_centered(probability_y, candidate_xy[:, 1])
    task = local_y[:, :, None] * local_x[:, None, :]
    valid = valid_y[:, :, None] & valid_x[:, None, :]
    return task.astype(np.float32), valid.astype(np.float32)
