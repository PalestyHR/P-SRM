"""Run frozen Online TAPIR once and export candidate-aligned evidence/history.

This is the external-split counterpart of the established Design exporter. It
keeps Online TAPIR's causal state, candidate coordinates, visibility decision,
and 0.5 decision threshold unchanged while exposing the already-computed
low-resolution cost volume and causal native-valid history.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
from PIL import Image
from tapnet.models import tapir_model
from tapnet.utils import model_utils


THRESHOLDS = (1, 2, 4, 8, 16)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def load_segment_order(path: Path) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            segment = f"{row[0]}_{int(row[1]):06}_{int(row[2]):06}"
            if segment not in seen:
                seen.add(segment)
                order.append(segment)
    return order


def selected_examples(data_root: Path, ids: set[str]):
    segment_order = load_segment_order(data_root / "tapvid_kinetics_development.csv")
    global_index = 0
    selected = 0
    for shard in sorted((data_root / "pickle_development_10").glob("*.pkl")):
        with shard.open("rb") as handle:
            examples = pickle.load(handle)
        for example in examples:
            segment = segment_order[global_index]
            if segment in ids:
                selected += 1
                yield global_index, segment, example
            global_index += 1
        del examples
    if global_index != len(segment_order) or selected != len(ids):
        raise RuntimeError(
            f"annotation/ID mismatch: examples={global_index}, order={len(segment_order)}, selected={selected}, ids={len(ids)}"
        )


def decode_example(example: dict):
    frames = [
        np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"), dtype=np.uint8)
        for encoded in example["video"]
    ]
    frames = media.resize_video(np.stack(frames), (256, 256)).astype(np.uint8)
    tracks = example["points"].astype(np.float32) * np.asarray([256.0, 256.0], np.float32)
    occluded = example["occluded"].astype(bool)
    keep = np.sum(~occluded, axis=1) > 0
    tracks, occluded = tracks[keep], occluded[keep]
    query_frames = np.zeros(len(tracks), np.int32)
    query_points = np.zeros((len(tracks), 3), np.float32)
    for index in range(len(tracks)):
        frame = int(np.flatnonzero(~occluded[index])[0])
        x, y = tracks[index, frame]
        query_frames[index] = frame
        query_points[index] = [frame, y, x]
    return frames, tracks, occluded, query_frames, query_points


def load_native_rows(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = {row["segment_name"]: row for row in reader}
    if not fields or len(rows) == 0:
        raise ValueError("native metric table is empty")
    return fields, rows


def tapvid_metrics_first_query(
    query_frames: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
) -> dict[str, float]:
    evaluation = np.arange(gt_occluded.shape[1])[None, :] > query_frames[:, None]
    visible = ~gt_occluded
    pred_visible = ~pred_occluded
    metrics = {
        "occlusion_accuracy": float(
            np.sum((pred_occluded == gt_occluded) & evaluation) / np.sum(evaluation)
        )
    }
    squared_error = np.sum(np.square(pred_tracks - gt_tracks), axis=-1)
    fractions, jaccards = [], []
    for threshold in THRESHOLDS:
        within = squared_error < float(threshold * threshold)
        correct = within & visible
        count_correct = np.sum(correct & evaluation)
        count_visible = np.sum(visible & evaluation)
        fraction = float(count_correct / count_visible)
        true_positive = np.sum(correct & pred_visible & evaluation)
        false_positive = np.sum(
            (((~visible) & pred_visible) | ((~within) & pred_visible)) & evaluation
        )
        jaccard = float(true_positive / (count_visible + false_positive))
        metrics[f"pts_within_{threshold}"] = fraction
        metrics[f"jaccard_{threshold}"] = jaccard
        fractions.append(fraction)
        jaccards.append(jaccard)
    metrics["average_pts_within_thresh"] = float(np.mean(fractions))
    metrics["average_jaccard"] = float(np.mean(jaccards))
    return metrics


def binary_counts(
    query_frames: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
) -> dict[str, int]:
    evaluation = np.arange(gt_occluded.shape[1])[None, :] > query_frames[:, None]
    correct = (~gt_occluded) & (np.linalg.norm(pred_tracks - gt_tracks, axis=-1) <= 8.0)
    positive = ~pred_occluded
    return {
        "TP": int(np.sum(evaluation & positive & correct)),
        "TN": int(np.sum(evaluation & (~positive) & (~correct))),
        "FP": int(np.sum(evaluation & positive & (~correct))),
        "FN": int(np.sum(evaluation & (~positive) & correct)),
    }


def binary_f1(counts: dict[str, int]) -> float:
    denominator = 2 * counts["TP"] + counts["FP"] + counts["FN"]
    return 2.0 * counts["TP"] / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qmax", type=int, default=64)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    evidence_root = args.output_root / "evidence"
    history_root = args.output_root / "native_history"
    evidence_root.mkdir(exist_ok=True)
    history_root.mkdir(exist_ok=True)
    from .common import read_ids
    ids = set(read_ids(args.ids))
    checkpoint = np.load(args.checkpoint, allow_pickle=True).item()
    tapir = tapir_model.ParameterizedTAPIR(
        params=checkpoint["params"],
        state=checkpoint["state"],
        tapir_kwargs={"use_causal_conv": True, "bilinear_interp_with_depthwise_conv": False},
    )

    def online_model_init(frames, points):
        grids = tapir.get_feature_grids(frames, is_training=False)
        return tapir.get_query_features(
            frames, is_training=False, query_points=points, feature_grids=grids
        )

    def online_model_predict(frames, features, causal_context):
        grids = tapir.get_feature_grids(frames, is_training=False)
        cost_volume = jnp.einsum("bnc,bthwc->btnhw", features.lowres[0], grids.lowres[0])
        trajectories = tapir.estimate_trajectories(
            frames.shape[-3:-1],
            is_training=False,
            feature_grids=grids,
            query_features=features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=causal_context,
            get_causal_context=True,
        )
        new_context = trajectories.pop("causal_context")
        return {key: value[-1] for key, value in trajectories.items()}, new_context, cost_volume

    init_apply = jax.jit(online_model_init)
    predict_apply = jax.jit(online_model_predict)
    completed = 0
    per_video_rows: list[dict] = []
    for example_index, segment, example in selected_examples(args.data_root, ids):
        evidence_path = evidence_root / f"{segment}.npz"
        history_path = history_root / f"{segment}.npz"
        started = time.time()
        frames, gt_tracks, gt_occluded, query_frames, query_points = decode_example(example)
        num_queries = len(query_points)
        if num_queries > args.qmax:
            raise RuntimeError(f"query count {num_queries} exceeds qmax={args.qmax}")
        dummy = np.zeros((1, args.qmax, 3), np.float32)
        features = init_apply(
            model_utils.preprocess_frames(frames[0][None, None]), jnp.asarray(dummy)
        )
        causal_state = tapir.construct_initial_causal_state(
            args.qmax, len(features.resolutions) - 1
        )

        reject_query: list[int] = []
        reject_frame: list[int] = []
        reject_candidate: list[np.ndarray] = []
        reject_margin: list[float] = []
        reject_evidence: list[np.ndarray] = []
        reject_visible: list[bool] = []
        reject_error: list[float] = []
        reject_outcome: list[int] = []
        history_query: list[int] = []
        history_frame: list[int] = []
        history_candidate: list[np.ndarray] = []
        history_margin: list[float] = []
        history_score: list[float] = []
        history_valid: list[bool] = []
        predicted_tracks = np.zeros_like(gt_tracks, dtype=np.float32)
        predicted_occluded = np.ones_like(gt_occluded, dtype=bool)

        for frame_index in range(len(frames)):
            for query_index in np.flatnonzero(query_frames == frame_index):
                point = query_points[query_index : query_index + 1]
                image = model_utils.preprocess_frames(frames[frame_index][None, None])
                new_features = init_apply(image, jnp.asarray(point[None]))
                features, causal_state = tapir.update_query_features(
                    features, new_features, np.asarray([query_index]), causal_state
                )
            image = model_utils.preprocess_frames(frames[frame_index][None, None])
            prediction, causal_state, cost_volume = predict_apply(image, features, causal_state)
            jax.block_until_ready(prediction["tracks"])
            tracks = np.asarray(prediction["tracks"])[0, :num_queries, 0]
            occlusion = np.asarray(prediction["occlusion"])[0, :num_queries, 0]
            expected = np.asarray(prediction["expected_dist"])[0, :num_queries, 0]
            costs = np.asarray(cost_volume)[0, 0, :num_queries]
            visible_score = (1.0 - sigmoid(occlusion)) * (1.0 - sigmoid(expected))
            predicted_tracks[:, frame_index] = tracks
            predicted_occluded[:, frame_index] = ~(visible_score > 0.5)
            evaluation = frame_index > query_frames
            for query_index in np.flatnonzero(evaluation):
                score = float(visible_score[query_index])
                valid = score > 0.5
                history_query.append(int(query_index))
                history_frame.append(frame_index)
                history_candidate.append(tracks[query_index].astype(np.float32))
                history_margin.append(score - 0.5)
                history_score.append(score)
                history_valid.append(valid)
                if valid:
                    continue
                error = float(np.linalg.norm(tracks[query_index] - gt_tracks[query_index, frame_index]))
                gt_visible = not bool(gt_occluded[query_index, frame_index])
                outcome = 0 if gt_visible and error <= 8.0 else (1 if gt_visible else 2)
                reject_query.append(int(query_index))
                reject_frame.append(frame_index)
                reject_candidate.append(tracks[query_index].astype(np.float32))
                reject_margin.append(score - 0.5)
                reject_evidence.append(costs[query_index].astype(np.float16))
                reject_visible.append(gt_visible)
                reject_error.append(error)
                reject_outcome.append(outcome)

        observed = {
            "native_reject": len(reject_outcome),
            "native_valid": int(np.sum(history_valid)),
            "C": int(np.sum(np.asarray(reject_outcome) == 0)),
            "L": int(np.sum(np.asarray(reject_outcome) == 1)),
            "A": int(np.sum(np.asarray(reject_outcome) == 2)),
        }
        native_metrics = tapvid_metrics_first_query(
            query_frames, gt_occluded, gt_tracks, predicted_occluded, predicted_tracks
        )
        native_counts = binary_counts(
            query_frames, gt_occluded, gt_tracks, predicted_occluded, predicted_tracks
        )
        per_video_row = {
            "example_index": example_index,
            "segment_name": segment,
            "queries": num_queries,
            "frames": len(frames),
            "eval_rows": len(history_query),
            **observed,
            **{f"native_{key}": value for key, value in native_metrics.items()},
            **{f"native_{key}": value for key, value in native_counts.items()},
            "native_F1": binary_f1(native_counts),
            "elapsed_sec": time.time() - started,
        }
        per_video_rows.append(per_video_row)

        np.savez_compressed(
            evidence_path,
            example_index=np.int32(example_index),
            segment_name=np.asarray(segment),
            query_index=np.asarray(reject_query, np.int16),
            frame_index=np.asarray(reject_frame, np.int16),
            candidate_xy=np.asarray(reject_candidate, np.float32),
            candidate_map_xy=np.asarray(reject_candidate, np.float32) / 8.0,
            native_margin=np.asarray(reject_margin, np.float32),
            task_evidence=np.asarray(reject_evidence, np.float16),
            gt_visible=np.asarray(reject_visible, bool),
            candidate_error=np.asarray(reject_error, np.float32),
            outcome=np.asarray(reject_outcome, np.int8),
        )
        np.savez_compressed(
            history_path,
            segment_name=np.asarray(segment),
            query_index=np.asarray(history_query, np.int16),
            frame_index=np.asarray(history_frame, np.int16),
            query_frame=query_frames[np.asarray(history_query, int)].astype(np.int16),
            candidate_xy=np.asarray(history_candidate, np.float32),
            native_margin=np.asarray(history_margin, np.float32),
            native_visible_score=np.asarray(history_score, np.float32),
            native_valid=np.asarray(history_valid, bool),
        )
        completed += 1
        print(
            json.dumps(
                {
                    "event": "complete_video",
                    "completed": completed,
                    "total": len(ids),
                    "segment": segment,
                    "rejects": len(reject_outcome),
                    "C": observed["C"],
                    "seconds": round(time.time() - started, 2),
                }
            ),
            flush=True,
        )

    if len(per_video_rows) != len(ids):
        raise RuntimeError(f"completed {len(per_video_rows)} rows for {len(ids)} selected IDs")
    with (args.output_root / "per_video.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(per_video_rows, key=lambda row: row["segment_name"]))
    summary = {
        "status": "complete",
        "videos": len(ids),
        "same_forward_evidence": True,
        "causal_history": True,
        "candidate_changed": False,
        "native_decision_changed": False,
        "native_metrics_source": "the same forward pass used for evidence export",
    }
    (args.output_root / "EXPORT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
