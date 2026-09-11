"""Streaming data interface for Online TAPIR P-SRM training."""

from __future__ import annotations

import csv
from urllib.parse import quote
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class VideoRecord:
    segment: str
    path: Path
    rows: int
    positives: int
    fold: int


def load_fold_manifest(manifest: Path, evidence_root: Path) -> list[VideoRecord]:
    records: list[VideoRecord] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            segment = row["segment"]
            records.append(
                VideoRecord(
                    segment=segment,
                    path=evidence_root / row.get("file", quote(segment, safe="-_ .") + ".npz"),
                    rows=int(row["rows"]),
                    positives=int(row["positives"]),
                    fold=int(row["fold"]),
                )
            )
    if not records or len({record.segment for record in records}) != len(records):
        raise ValueError("the grouped fold manifest must contain unique non-empty groups")
    if {record.fold for record in records} != {0, 1, 2, 3, 4}:
        raise ValueError("the frozen Design fold manifest must contain folds 0..4")
    return records


def _bilinear_support_batch(
    candidate_xy: np.ndarray,
    candidate_map_xy: np.ndarray,
    shape: tuple[int, int] = (32, 32),
    frame_size: tuple[int, int] = (256, 256),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = np.asarray(candidate_xy, np.float32).reshape(-1, 2)
    candidate_map = np.asarray(candidate_map_xy, np.float32).reshape(-1, 2)
    if len(candidate) != len(candidate_map):
        raise ValueError("candidate coordinate arrays are not aligned")
    height, width = shape
    frame_height, frame_width = frame_size
    in_frame = (
        (candidate[:, 0] >= 0.0)
        & (candidate[:, 0] < frame_width)
        & (candidate[:, 1] >= 0.0)
        & (candidate[:, 1] < frame_height)
    )
    clipped_x = np.clip(candidate_map[:, 0], 0.0, width - 1.0)
    clipped_y = np.clip(candidate_map[:, 1], 0.0, height - 1.0)
    boundary = in_frame & (
        (clipped_x != candidate_map[:, 0]) | (clipped_y != candidate_map[:, 1])
    )
    support = np.zeros((len(candidate), height, width), np.float32)
    active = np.flatnonzero(in_frame)
    if active.size:
        x, y = clipped_x[active], clipped_y[active]
        x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        wx, wy = x - x0, y - y0
        np.add.at(support, (active, y0, x0), (1.0 - wx) * (1.0 - wy))
        np.add.at(support, (active, y0, x1), wx * (1.0 - wy))
        np.add.at(support, (active, y1, x0), (1.0 - wx) * wy)
        np.add.at(support, (active, y1, x1), wx * wy)
    return support, in_frame, boundary


def build_dense_projection(
    task_evidence: np.ndarray,
    candidate_xy: np.ndarray,
    candidate_map_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the three-plane dense projection and mechanical metadata."""

    evidence = np.asarray(task_evidence, np.float32)
    if evidence.ndim != 3 or tuple(evidence.shape[1:]) != (32, 32):
        raise ValueError(f"task_evidence must have shape [B,32,32], got {evidence.shape}")
    if not np.isfinite(evidence).all():
        raise ValueError("task_evidence contains non-finite values")
    support, in_frame, boundary = _bilinear_support_batch(candidate_xy, candidate_map_xy)
    valid = np.ones_like(evidence, dtype=np.float32)
    dense = np.stack([evidence, support, valid], axis=1)
    candidate = np.asarray(candidate_xy, np.float32).reshape(-1, 2)
    metadata = np.column_stack(
        [candidate[:, 0] / 256.0, candidate[:, 1] / 256.0, in_frame, boundary]
    ).astype(np.float32)
    return dense, metadata


def iter_video_batches(
    records: Sequence[VideoRecord],
    batch_size: int,
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield batches without loading the full 1.16M-row tensor into RAM."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = np.random.RandomState(seed + 1009 * epoch)
    video_order = np.arange(len(records))
    if shuffle:
        rng.shuffle(video_order)
    for video_position in video_order:
        record = records[int(video_position)]
        if not record.path.exists():
            raise FileNotFoundError(record.path)
        with np.load(record.path) as payload:
            required = {
                "segment_name",
                "query_index",
                "frame_index",
                "native_margin",
                "outcome",
                "dense_evidence",
                "candidate_metadata",
                "candidate_error",
                "gt_visible",
            }
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"{record.path.name} missing fields: {sorted(missing)}")
            n = int(payload["outcome"].size)
            if n != record.rows or str(payload["segment_name"]) != record.segment:
                raise ValueError(f"fold-manifest mismatch for {record.segment}")
            # Materialize each compressed NPZ member once per video. Accessing a
            # compressed member through ``NpzFile`` inside every mini-batch would
            # decompress the complete member repeatedly while yielding identical
            # values. Keeping the arrays for the current video avoids that I/O
            # amplification without changing samples, order, or model inputs.
            dense_all = payload["dense_evidence"].astype(np.float32)
            metadata_all = payload["candidate_metadata"].astype(np.float32)
            query_index_all = payload["query_index"].astype(np.int32)
            frame_index_all = payload["frame_index"].astype(np.int32)
            native_margin_all = payload["native_margin"].astype(np.float32)
            outcome_all = payload["outcome"].astype(np.int8)
            candidate_error_all = payload["candidate_error"].astype(np.float32)
            gt_visible_all = payload["gt_visible"].astype(bool)
            indices = np.arange(n)
            if shuffle:
                rng.shuffle(indices)
            for start in range(0, n, batch_size):
                chosen = indices[start : start + batch_size]
                dense = dense_all[chosen]
                metadata = metadata_all[chosen]
                if dense.shape != (len(chosen), 3, 32, 32):
                    raise ValueError(f"{record.path.name} has invalid dense evidence")
                if metadata.shape != (len(chosen), 4):
                    raise ValueError(f"{record.path.name} has invalid candidate metadata")
                if not np.isfinite(dense).all() or not np.isfinite(metadata).all():
                    raise ValueError(f"{record.path.name} contains non-finite model input")
                outcome = outcome_all[chosen]
                yield {
                    "dense": dense,
                    "metadata": metadata,
                    "label": (outcome == 0).astype(np.float32),
                    "segment": np.repeat(record.segment, len(chosen)),
                    "query_index": query_index_all[chosen],
                    "frame_index": frame_index_all[chosen],
                    "native_margin": native_margin_all[chosen],
                    "outcome": outcome,
                    "candidate_error": candidate_error_all[chosen],
                    "gt_visible": gt_visible_all[chosen],
                    "fold": np.full(len(chosen), record.fold, np.int8),
                }


def fold_local_pos_weight(records: Sequence[VideoRecord]) -> float:
    positives = sum(record.positives for record in records)
    rows = sum(record.rows for record in records)
    negatives = rows - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("training fold contains an empty class")
    return float(negatives / positives)


def fold_local_visibility_pos_weight(records: Sequence[VideoRecord]) -> float:
    """Compute the visible-vs-invisible weight from the outer-train fold only."""

    visible = 0
    rows = 0
    for record in records:
        if not record.path.exists():
            raise FileNotFoundError(record.path)
        with np.load(record.path) as payload:
            if "gt_visible" not in payload.files:
                raise ValueError(f"{record.path.name} missing gt_visible")
            values = payload["gt_visible"].astype(bool)
            if values.size != record.rows:
                raise ValueError(f"fold-manifest mismatch for {record.segment}")
            visible += int(values.sum())
            rows += int(values.size)
    invisible = rows - visible
    if visible <= 0 or invisible <= 0:
        raise ValueError("training fold contains an empty visibility class")
    return float(invisible / visible)
