"""Train-only stacking and frozen external closure for TrackNet-family P-SRM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from psrm.tracknet_metrics import (
    load_sequences,
    select_threshold,
    summarize,
)
from psrm.tracknet_oof import bootstrap_pooled_delta, crossfit, load_oof


def load_model_scores(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as source:
        frame = pd.DataFrame(
            {
                "segment": source["segment"].astype(str),
                "query_index": source["query_index"].astype(int),
                "frame_index": source["frame_index"].astype(int),
                "native_margin": source["native_margin"].astype(float),
                "outcome": source["outcome"].astype(int),
                "quality_logit": source["quality_logit"].astype(float),
                "quality_score": source["quality_score"].astype(float),
            }
        )
    frame["source_id"] = frame.segment + ":" + frame.frame_index.astype(str)
    if frame.source_id.duplicated().any():
        raise ValueError("duplicate model-score identity")
    return frame


def load_s3(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as source:
            segment = str(source["segment_name"])
            query = source["query_index"].astype(int)
            frame = source["frame_index"].astype(int)
            feature = source["s3_features"].astype(np.float32)
        for index in range(len(frame)):
            rows.append(
                {
                    "segment": segment,
                    "query_index": int(query[index]),
                    "frame_index": int(frame[index]),
                    "s3": feature[index],
                }
            )
    result = pd.DataFrame(rows)
    if result.empty or result.duplicated(["segment", "query_index", "frame_index"]).any():
        raise ValueError("S3 reservoir is empty or contains duplicate identities")
    return result


def join_s3(score: pd.DataFrame, s3: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    joined = score.merge(
        s3,
        on=["segment", "query_index", "frame_index"],
        how="left",
        validate="one_to_one",
    )
    if joined.s3.isna().any():
        raise ValueError("score/S3 identity mismatch")
    matrix = np.stack(joined.s3.to_numpy())
    return joined.drop(columns="s3"), matrix


def fit_classifier(matrix: np.ndarray, label: np.ndarray):
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=2000, random_state=42
    ).fit(scaler.transform(matrix), label)
    return scaler, model


def align_to_candidates(data: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    candidates = data.loc[~data.native_valid].reset_index(drop=True).copy()
    candidates["source_id"] = (
        candidates.sequence_id.astype(str) + ":" + candidates.frame_id.astype(str)
    )
    joined = candidates[["source_id", "accept_label", "gt_visible", "sport"]].merge(
        score, on="source_id", how="left", validate="one_to_one"
    )
    if joined.isna().any().any():
        raise ValueError("evaluation CSV/model-score identity mismatch")
    expected = np.where(
        joined.accept_label.astype(bool), 0, np.where(joined.gt_visible.astype(bool), 1, 2)
    )
    if not np.array_equal(expected, joined.outcome.to_numpy(int)):
        raise ValueError("evaluation C/L/A mismatch")
    return joined
