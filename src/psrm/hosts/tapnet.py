#!/usr/bin/env python3
"""Faithful TAP-Net evaluation plus reject evidence and causal history export.

The official TAP-Net model and checkpoint are left unchanged. This runner only
adds a Design-split filter, deterministic multi-GPU sharding, official TAP-Vid
metric aggregation, and read-only export of tensors already computed by the
same forward pass.  The optional history package contains runtime-legal native
outputs for every post-query evaluation row, including native-valid rows.  It
contains no ground truth and exists only to derive causal history features.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import sys
import time
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
from PIL import Image

from tapnet.models import tapnet_model
from tapnet.utils import optimizers


# The 2022 checkpoint pickles an Optax state through the repository's former
# module path. This compatibility alias changes neither weights nor inference.
sys.modules.setdefault("tapnet.optimizers", optimizers)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_segment_order(csv_path: Path) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    with csv_path.open(newline="") as handle:
        for row in csv.reader(handle):
            segment = f"{row[0]}_{int(row[1]):06}_{int(row[2]):06}"
            if segment not in seen:
                seen.add(segment)
                order.append(segment)
    return order


def load_examples(data_root: Path) -> list[dict]:
    examples: list[dict] = []
    for shard in sorted((data_root / "pickle_development_10").glob("*.pkl")):
        with shard.open("rb") as handle:
            values = pickle.load(handle)
        if isinstance(values, dict):
            values = list(values.values())
        examples.extend(values)
    return examples


def decode_first_query(example: dict, qmax: int):
    frames = np.stack(
        [np.asarray(Image.open(io.BytesIO(blob)).convert("RGB")) for blob in example["video"]],
        axis=0,
    )
    frames = media.resize_video(frames, (256, 256)).astype(np.float32) / 255.0 * 2.0 - 1.0
    gt_tracks = np.asarray(example["points"], dtype=np.float32).copy()
    gt_tracks *= np.asarray([256.0, 256.0], dtype=np.float32)
    gt_occ = np.asarray(example["occluded"], dtype=bool)
    valid = np.sum(~gt_occ, axis=1) > 0
    gt_tracks = gt_tracks[valid]
    gt_occ = gt_occ[valid]
    if len(gt_tracks) > qmax:
        raise RuntimeError(f"query count {len(gt_tracks)} exceeds qmax={qmax}")
    query = np.zeros((qmax, 3), dtype=np.float32)
    query_frames = np.zeros(len(gt_tracks), dtype=np.int32)
    for qi in range(len(gt_tracks)):
        frame = int(np.flatnonzero(~gt_occ[qi])[0])
        query_frames[qi] = frame
        x, y = gt_tracks[qi, frame]
        query[qi] = np.asarray([frame, y, x], dtype=np.float32)
    return frames[None], query[None], query_frames, gt_tracks, gt_occ


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
) -> dict[str, np.ndarray]:
    """Official TAP-Vid first-query equations, copied without modification."""
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
    query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    query_frame = np.round(query_points[..., 0]).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0
    summing_axis = (1, 2)
    metrics: dict[str, np.ndarray] = {}
    metrics["occlusion_accuracy"] = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points, axis=summing_axis
    ) / np.sum(evaluation_points, axis=summing_axis)
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in (1, 2, 4, 8, 16):
        within_dist = np.sum(np.square(pred_tracks - gt_tracks), axis=-1) < thresh**2
        is_correct = within_dist & visible
        count_correct = np.sum(is_correct & evaluation_points, axis=summing_axis)
        count_visible_points = np.sum(visible & evaluation_points, axis=summing_axis)
        frac_correct = count_correct / count_visible_points
        metrics[f"pts_within_{thresh}"] = frac_correct
        all_frac_within.append(frac_correct)
        true_positives = np.sum(is_correct & pred_visible & evaluation_points, axis=summing_axis)
        gt_positives = np.sum(visible & evaluation_points, axis=summing_axis)
        false_positives = np.sum(
            pred_visible & ~is_correct & evaluation_points, axis=summing_axis
        )
        jaccard = true_positives / (gt_positives + false_positives)
        metrics[f"jaccard_{thresh}"] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_pts_within_thresh"] = np.mean(np.stack(all_frac_within, axis=1), axis=1)
    metrics["average_jaccard"] = np.mean(np.stack(all_jaccard, axis=1), axis=1)
    return metrics


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    create = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if create:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--design-ids", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--qmax", type=int, default=64)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--export-evidence", action="store_true")
    parser.add_argument("--export-history", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    evidence_root = args.output_root / "evidence"
    history_root = args.output_root / "native_history"
    if args.export_evidence:
        evidence_root.mkdir(exist_ok=True)
    if args.export_history:
        history_root.mkdir(exist_ok=True)
    per_video_path = args.output_root / f"per_video_shard{args.shard_index}.csv"
    summary_path = args.output_root / f"summary_shard{args.shard_index}.json"

    from .common import read_ids
    design_ids = set(read_ids(args.design_ids))
    segment_order = load_segment_order(args.data_root / "tapvid_kinetics_development.csv")
    examples = load_examples(args.data_root)
    if len(segment_order) != len(examples):
        raise RuntimeError(f"segment/example mismatch: {len(segment_order)} vs {len(examples)}")
    selected = [
        (global_index, segment, example)
        for global_index, (segment, example) in enumerate(zip(segment_order, examples))
        if segment in design_ids and global_index % args.num_shards == args.shard_index
    ]
    if args.max_examples is not None:
        selected = selected[: args.max_examples]

    checkpoint = np.load(args.checkpoint, allow_pickle=True).item()

    def forward(video, query_points):
        return tapnet_model.TAPNet()(video, is_training=False, query_points=query_points,
                                     query_chunk_size=32, get_query_feats=True)

    transformed = hk.transform_with_state(forward)
    apply = jax.jit(transformed.apply)
    metric_keys = [
        "average_jaccard", "average_pts_within_thresh", "occlusion_accuracy",
        "jaccard_1", "jaccard_2", "jaccard_4", "jaccard_8", "jaccard_16",
        "pts_within_1", "pts_within_2", "pts_within_4", "pts_within_8",
        "pts_within_16",
    ]
    fieldnames = [
        "global_index", "segment_name", "queries", "frames", "eval_rows",
        "native_valid", "native_reject", "C", "L", "A", "reject_rate",
        "correct_rejected_prevalence",
        *[f"native_{key}" for key in metric_keys],
        "native_TP", "native_TN", "native_FP", "native_FN", "native_F1",
        *[f"oracle_{key}" for key in metric_keys],
        "oracle_TP", "oracle_TN", "oracle_FP", "oracle_FN", "oracle_F1",
        "elapsed_sec",
    ]
    completed: set[str] = set()
    if per_video_path.exists():
        with per_video_path.open(newline="") as handle:
            completed = {row["segment_name"] for row in csv.DictReader(handle)}

    print(
        f"START shard={args.shard_index}/{args.num_shards} selected={len(selected)} "
        f"completed={len(completed)} devices={jax.devices()}", flush=True
    )
    for ordinal, (global_index, segment, example) in enumerate(selected, start=1):
        if segment in completed:
            continue
        start = time.time()
        video, padded_query, query_frames, gt_tracks, gt_occ = decode_first_query(example, args.qmax)
        output, _ = apply(
            checkpoint["params"], checkpoint["state"], None,
            jnp.asarray(video), jnp.asarray(padded_query),
        )
        q = len(gt_tracks)
        pred_tracks = np.asarray(output["tracks"])[0, :q]
        occ_logits = np.asarray(output["occlusion"])[0, :q]
        pred_occ = sigmoid(occ_logits) > 0.5
        query = padded_query[:, :q]
        metrics = compute_tapvid_metrics(
            query, gt_occ[None], gt_tracks[None], pred_occ[None], pred_tracks[None]
        )

        eye = np.eye(gt_tracks.shape[1], dtype=np.int8)
        eval_mask = (np.cumsum(eye, axis=1) - eye)[query_frames] > 0
        rejected = pred_occ & eval_mask
        errors = np.linalg.norm(pred_tracks - gt_tracks, axis=-1)
        c_mask = rejected & ~gt_occ & (errors <= 8.0)
        l_mask = rejected & ~gt_occ & (errors > 8.0)
        a_mask = rejected & gt_occ
        correct_candidate = (~gt_occ) & (errors <= 8.0) & eval_mask
        native_positive = (~pred_occ) & eval_mask
        native_tp = int(np.sum(native_positive & correct_candidate))
        native_fp = int(np.sum(native_positive & ~correct_candidate))
        native_fn = int(np.sum((~native_positive) & correct_candidate & eval_mask))
        native_tn = int(np.sum((~native_positive) & ~correct_candidate & eval_mask))
        native_f1 = 2.0 * native_tp / max(2 * native_tp + native_fp + native_fn, 1)

        oracle_occ = pred_occ.copy()
        oracle_occ[c_mask] = False
        oracle_metrics = compute_tapvid_metrics(
            query, gt_occ[None], gt_tracks[None], oracle_occ[None], pred_tracks[None]
        )
        oracle_positive = (~oracle_occ) & eval_mask
        oracle_tp = int(np.sum(oracle_positive & correct_candidate))
        oracle_fp = int(np.sum(oracle_positive & ~correct_candidate))
        oracle_fn = int(np.sum((~oracle_positive) & correct_candidate & eval_mask))
        oracle_tn = int(np.sum((~oracle_positive) & ~correct_candidate & eval_mask))
        oracle_f1 = 2.0 * oracle_tp / max(2 * oracle_tp + oracle_fp + oracle_fn, 1)
        native_reject = int(np.sum(rejected))
        eval_rows = int(np.sum(eval_mask))
        counts = {
            "native_valid": int(np.sum((~pred_occ) & eval_mask)),
            "native_reject": native_reject,
            "C": int(np.sum(c_mask)),
            "L": int(np.sum(l_mask)),
            "A": int(np.sum(a_mask)),
        }

        if args.export_history:
            # Keep every post-query evaluation row so the temporal module can
            # update its state from past native-valid predictions.  Ground
            # truth, labels, and correctness-derived fields are intentionally
            # absent from this runtime-only package.
            history_qi, history_ti = np.nonzero(eval_mask)
            np.savez_compressed(
                history_root / f"{global_index:04d}_{segment}.npz",
                segment_name=np.asarray(segment),
                query_index=history_qi.astype(np.int16),
                frame_index=history_ti.astype(np.int16),
                query_frame=query_frames[history_qi].astype(np.int16),
                candidate_xy=pred_tracks[history_qi, history_ti].astype(np.float32),
                native_margin=(-occ_logits[history_qi, history_ti]).astype(np.float32),
                native_visible_score=(1.0 - sigmoid(occ_logits[history_qi, history_ti])).astype(np.float32),
                native_valid=(~pred_occ[history_qi, history_ti]).astype(bool),
            )

        if args.export_evidence and native_reject:
            feature_grid = np.asarray(output["feature_grid"])[0]
            query_feats = np.asarray(output["query_feats"])[0, :q]
            cost_volume = np.einsum("nc,thwc->nthw", query_feats, feature_grid)
            qi, ti = np.nonzero(rejected)
            outcome = np.where(c_mask[qi, ti], 0, np.where(l_mask[qi, ti], 1, 2)).astype(np.int8)
            np.savez_compressed(
                evidence_root / f"{global_index:04d}_{segment}.npz",
                source_id=np.asarray([f"{segment}|q{int(x)}|t{int(y)}" for x, y in zip(qi, ti)]),
                query_index=qi.astype(np.int16), frame=ti.astype(np.int16),
                candidate_xy=pred_tracks[qi, ti].astype(np.float32),
                occlusion_logit=occ_logits[qi, ti].astype(np.float32),
                native_margin=(-occ_logits[qi, ti]).astype(np.float32),
                gt_visible=(~gt_occ[qi, ti]), gt_xy=gt_tracks[qi, ti].astype(np.float32),
                candidate_error_px=errors[qi, ti].astype(np.float32), outcome_cla=outcome,
                task_evidence=cost_volume[qi, ti].astype(np.float16),
            )

        row = {
            "global_index": global_index,
            "segment_name": segment,
            "queries": q,
            "frames": gt_tracks.shape[1],
            "eval_rows": eval_rows,
            **counts,
            "reject_rate": native_reject / eval_rows if eval_rows else 0.0,
            "correct_rejected_prevalence": counts["C"] / native_reject if native_reject else 0.0,
            **{f"native_{key}": float(metrics[key][0]) for key in metric_keys},
            "native_TP": native_tp,
            "native_TN": native_tn,
            "native_FP": native_fp,
            "native_FN": native_fn,
            "native_F1": native_f1,
            **{f"oracle_{key}": float(oracle_metrics[key][0]) for key in metric_keys},
            "oracle_TP": oracle_tp,
            "oracle_TN": oracle_tn,
            "oracle_FP": oracle_fp,
            "oracle_FN": oracle_fn,
            "oracle_F1": oracle_f1,
            "elapsed_sec": time.time() - start,
        }
        append_csv(per_video_path, fieldnames, row)
        summary_path.write_text(json.dumps({
            "status": "running", "shard_index": args.shard_index,
            "num_shards": args.num_shards, "selected": len(selected),
            "completed": ordinal, "last_segment": segment,
        }, indent=2))
        print(
            f"DONE {ordinal}/{len(selected)} index={global_index} segment={segment} "
            f"AJ={row['native_average_jaccard']:.6f} "
            f"oracleAJ={row['oracle_average_jaccard']:.6f} "
            f"reject={native_reject} C={counts['C']} "
            f"elapsed={row['elapsed_sec']:.1f}s", flush=True
        )

    summary_path.write_text(json.dumps({
        "status": "complete", "shard_index": args.shard_index,
        "num_shards": args.num_shards, "selected": len(selected),
    }, indent=2))


if __name__ == "__main__":
    main()
