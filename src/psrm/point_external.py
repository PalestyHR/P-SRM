"""Apply a Design-frozen P-SRM to a separate point-tracking evaluation split.

The neural reader is refit on all Design rows by ``refit_r1a_r4.py``. This
entry fits the already-selected causal-history residual and optional native
margin fusion on Design only, selects the working point on Design OOF only,
and applies that fixed system once to the external split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from psrm.point_metrics import (
    RISK,
    bootstrap_metric_delta,
    bootstrap_video_delta,
    crossfit_logistic,
    f1,
    official_tap_transition,
    precision,
    scan_system,
)


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))).astype(np.float32)


def load_scores(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as source:
        result = pd.DataFrame(
            {
                "segment": source["segment"].astype(str),
                "query_index": source["query_index"].astype(int),
                "frame_index": source["frame_index"].astype(int),
                "native_margin": source["native_margin"].astype(float),
                "outcome": source["outcome"].astype(int),
                "candidate_error": source["candidate_error"].astype(float),
                "gt_visible": source["gt_visible"].astype(bool),
                "quality_logit": source["quality_logit"].astype(float),
                "quality_score": source["quality_score"].astype(float),
            }
        )
        if "fold" in source.files:
            result["fold"] = source["fold"].astype(int)
    if result.duplicated(["segment", "query_index", "frame_index"]).any():
        raise ValueError(f"duplicate score identity in {path}")
    return result


def load_s3(root: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as source:
            if "segment_name" not in source.files:
                continue
            segment = str(source["segment_name"])
            query = source["query_index"].astype(int)
            frame = source["frame_index"].astype(int)
            feature = source["s3_features"].astype(np.float32)
        piece = pd.DataFrame(
            {
                "segment": np.repeat(segment, len(query)),
                "query_index": query,
                "frame_index": frame,
                "s3": list(feature),
            }
        )
        pieces.append(piece)
    if not pieces:
        raise ValueError(f"no temporal features in {root}")
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(["segment", "query_index", "frame_index"]).any():
        raise ValueError(f"duplicate temporal identity in {root}")
    return result


def join_s3(score: pd.DataFrame, s3: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    joined = score.merge(
        s3,
        on=["segment", "query_index", "frame_index"],
        how="left",
        validate="one_to_one",
    )
    if joined["s3"].isna().any():
        raise ValueError("score/temporal identity mismatch")
    matrix = np.stack(joined.pop("s3").to_numpy()).astype(np.float32)
    return joined, matrix


def fit_balanced(matrix: np.ndarray, target: np.ndarray):
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        solver="lbfgs",
    ).fit(scaler.transform(matrix), target)
    return scaler, model


def evaluate_fixed(
    frame: pd.DataFrame,
    native_table: pd.DataFrame,
    score: np.ndarray,
    threshold: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["score"] = np.asarray(score, float)
    native = native_table.drop_duplicates("segment_name").set_index("segment_name").sort_index()
    names = native.index.to_numpy(str)
    group_index = {name: index for index, name in enumerate(names)}
    if not set(frame.segment).issubset(group_index):
        raise ValueError("evaluation scores contain a group absent from native metrics")
    accepted = frame.score.to_numpy(float) >= threshold
    accepted_frame = frame.loc[accepted]
    accepted_group = np.asarray([group_index[name] for name in accepted_frame.segment], np.int32)
    accepted_outcome = accepted_frame.outcome.to_numpy(int)
    accepted_error = accepted_frame.candidate_error.to_numpy(float)
    accepted_visible = accepted_frame.gt_visible.to_numpy(bool)
    n = len(names)
    counts = {}
    for code, label in ((0, "C"), (1, "L"), (2, "A")):
        counts[label] = np.bincount(
            accepted_group,
            weights=(accepted_outcome == code).astype(int),
            minlength=n,
        ).astype(int)
    within = np.stack(
        [
            np.bincount(
                accepted_group,
                weights=(accepted_visible & (accepted_error <= radius)).astype(int),
                minlength=n,
            ).astype(int)
            for radius in (1, 2, 4, 8, 16)
        ],
        axis=0,
    )
    tp0 = native.native_TP.to_numpy(float)
    tn0 = native.native_TN.to_numpy(float)
    fp0 = native.native_FP.to_numpy(float)
    fn0 = native.native_FN.to_numpy(float)
    base_f1 = f1(tp0, fp0, fn0)
    final_f1 = f1(tp0 + counts["C"], fp0 + counts["L"] + counts["A"], fn0 - counts["C"])
    official = official_tap_transition(
        native.reset_index(), counts["C"], counts["L"], counts["A"], within
    )
    per_video = pd.DataFrame(
        {
            "segment_name": names,
            "native_f1": base_f1,
            "psrm_f1": final_f1,
            "delta_f1": final_f1 - base_f1,
            "accepted_C": counts["C"],
            "accepted_L": counts["L"],
            "accepted_A": counts["A"],
            **official,
        }
    )
    per_video["delta_average_jaccard"] = per_video.psrm_average_jaccard - per_video.native_average_jaccard
    per_video["delta_occlusion_accuracy"] = per_video.psrm_occlusion_accuracy - per_video.native_occlusion_accuracy

    C = int(counts["C"].sum())
    L = int(counts["L"].sum())
    A = int(counts["A"].sum())
    native_tp, native_fp, native_fn = tp0.sum(), fp0.sum(), fn0.sum()
    final_tp, final_fp, final_fn = native_tp + C, native_fp + L + A, native_fn - C
    native_precision = float(precision(native_tp, native_fp))
    final_precision = float(precision(final_tp, final_fp))
    native_recall = float(native_tp / max(native_tp + native_fn, 1.0))
    final_recall = float(final_tp / max(final_tp + final_fn, 1.0))
    summary = {
        "threshold_frozen_from_design": float(threshold),
        "evaluation_videos": int(n),
        "accepted": int(C + L + A),
        "C": C,
        "L": L,
        "A": A,
        "native_precision": native_precision,
        "psrm_precision": final_precision,
        "native_recall": native_recall,
        "psrm_recall": final_recall,
        "native_pooled_f1": float(f1(native_tp, native_fp, native_fn)),
        "psrm_pooled_f1": float(f1(final_tp, final_fp, final_fn)),
        "delta_pooled_f1": float(f1(final_tp, final_fp, final_fn) - f1(native_tp, native_fp, native_fn)),
        "delta_precision": final_precision - native_precision,
        "added_invisible_rate": float(A / max(tn0.sum(), 1.0)),
        "negative_video_rate": float(np.mean(per_video.delta_f1 < 0)),
        "video_delta_f1_q10": float(np.quantile(per_video.delta_f1, 0.10)),
        "native_mean_average_jaccard": float(per_video.native_average_jaccard.mean()),
        "psrm_mean_average_jaccard": float(per_video.psrm_average_jaccard.mean()),
        "native_mean_occlusion_accuracy": float(per_video.native_occlusion_accuracy.mean()),
        "psrm_mean_occlusion_accuracy": float(per_video.psrm_occlusion_accuracy.mean()),
        **bootstrap_video_delta(per_video),
        **bootstrap_metric_delta(per_video, "delta_average_jaccard", "delta_average_jaccard"),
        **bootstrap_metric_delta(per_video, "delta_occlusion_accuracy", "delta_occlusion_accuracy"),
    }
    summary["risk_pass_on_external"] = bool(
        summary["added_invisible_rate"] <= RISK["max_added_invisible_rate"]
        and summary["delta_precision"] >= RISK["min_delta_precision"]
        and summary["negative_video_rate"] <= RISK["max_negative_video_rate"]
        and summary["video_delta_f1_q10"] >= RISK["min_video_delta_f1_q10"]
    )
    scored = frame.assign(accepted=accepted)
    return summary, per_video, scored
