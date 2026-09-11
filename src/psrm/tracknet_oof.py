"""Exact train-side system closure for TrackNet-family P-SRM OOF scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from psrm.tracknet_metrics import (
    load_sequences,
    metrics,
    select_threshold,
    summarize,
)


def load_oof(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as source:
        frame = pd.DataFrame(
            {
                "segment": source["segment"].astype(str),
                "query_index": source["query_index"].astype(int),
                "frame_index": source["frame_index"].astype(int),
                "native_margin": source["native_margin"].astype(float),
                "outcome": source["outcome"].astype(int),
                "fold": source["fold"].astype(int),
                "quality_logit": source["quality_logit"].astype(float),
                "quality_score": source["quality_score"].astype(float),
            }
        )
    if frame.empty or frame.duplicated(["segment", "query_index", "frame_index"]).any():
        raise ValueError("OOF score identities are empty or duplicated")
    if not np.isfinite(frame[["native_margin", "quality_logit", "quality_score"]]).all().all():
        raise ValueError("OOF scores contain non-finite values")
    frame["source_id"] = frame.segment + ":" + frame.frame_index.astype(str)
    frame["label"] = (frame.outcome == 0).astype(int)
    return frame


def crossfit(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    output = np.full(len(frame), np.nan, float)
    for fold in sorted(frame.fold.unique()):
        train = frame.fold.to_numpy() != fold
        holdout = ~train
        scaler = StandardScaler().fit(frame.loc[train, features])
        model = LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2000, random_state=0
        ).fit(scaler.transform(frame.loc[train, features]), frame.loc[train, "label"])
        output[holdout] = model.predict_proba(
            scaler.transform(frame.loc[holdout, features])
        )[:, 1]
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite cross-fitted fusion score")
    return output


def bootstrap_pooled_delta(per: pd.DataFrame, replicates: int = 10_000) -> dict[str, float]:
    rng = np.random.RandomState(0)
    values = per[["TP", "TN", "FP", "FN", "SRM_TP", "SRM_TN", "SRM_FP", "SRM_FN"]].to_numpy(float)
    samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        total = values[rng.randint(0, len(values), size=len(values))].sum(axis=0)
        native = metrics(*total[:4].astype(int))
        augmented = metrics(*total[4:].astype(int))
        samples[index] = augmented["F1"] - native["F1"]
    return {
        "delta_f1_ci95_low": float(np.quantile(samples, 0.025)),
        "delta_f1_ci95_high": float(np.quantile(samples, 0.975)),
        "delta_f1_probability_positive": float(np.mean(samples > 0)),
    }


def align_csv_data(csv_root: Path, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_sequences(csv_root)
    candidates = data.loc[~data.native_valid].reset_index(drop=True).copy()
    candidates["source_id"] = (
        candidates.sequence_id.astype(str) + ":" + candidates.frame_id.astype(str)
    )
    aligned = candidates[["source_id", "accept_label", "gt_visible"]].merge(
        oof, on="source_id", how="left", validate="one_to_one"
    )
    if aligned.isna().any().any():
        raise ValueError("CSV candidates and OOF scores do not align")
    expected = np.where(
        aligned.accept_label.astype(bool), 0, np.where(aligned.gt_visible.astype(bool), 1, 2)
    )
    if not np.array_equal(expected, aligned.outcome.to_numpy(int)):
        raise ValueError("CSV and OOF C/L/A labels disagree")
    return data, aligned


def synthesize_from_counts(counts_path: Path, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.read_csv(counts_path)
    group_column = "rally_id" if "rally_id" in counts else "segment_name"
    counts = counts.rename(columns={group_column: "sequence_id"})
    candidate_rows = []
    for row in oof.itertuples(index=False):
        candidate_rows.append(
            {
                "sequence_id": row.segment,
                "frame_id": row.frame_index,
                "native_valid": False,
                "gt_visible": row.outcome != 2,
                "native_error": np.nan,
                "accept_label": row.outcome == 0,
            }
        )
    data = pd.DataFrame(candidate_rows)
    native_rows = []
    for row in counts.itertuples(index=False):
        group = str(row.sequence_id)
        observed = data[data.sequence_id == group]
        if int((~observed.gt_visible).sum()) != int(row.TN):
            raise ValueError(f"TN/A mismatch for {group}")
        if int(observed.gt_visible.sum()) != int(row.FN):
            raise ValueError(f"FN/(C+L) mismatch for {group}")
        for index in range(int(row.TP)):
            native_rows.append(
                {
                    "sequence_id": group,
                    "frame_id": -(index + 1),
                    "native_valid": True,
                    "gt_visible": True,
                    "native_error": 0.0,
                    "accept_label": False,
                }
            )
        for index in range(int(row.FP)):
            native_rows.append(
                {
                    "sequence_id": group,
                    "frame_id": -(int(row.TP) + index + 1),
                    "native_valid": True,
                    "gt_visible": False,
                    "native_error": np.nan,
                    "accept_label": False,
                }
            )
    data = pd.concat([data, pd.DataFrame(native_rows)], ignore_index=True)
    aligned = oof.copy()
    aligned["accept_label"] = aligned.outcome == 0
    aligned["gt_visible"] = aligned.outcome != 2
    return data, aligned
