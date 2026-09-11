"""Exact grouped system closure for a point-tracking P-SRM score file."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


RISK = {
    "max_added_invisible_rate": 0.03,
    "min_delta_precision": -0.005,
    "max_negative_video_rate": 0.30,
    "min_video_delta_f1_q10": -0.002,
}
TAP_THRESHOLDS = (1, 2, 4, 8, 16)


def sorted_linear_quantile(values: list[float], q: float) -> float:
    """NumPy-compatible linear quantile for an already sorted non-empty list."""

    if not values:
        raise ValueError("quantile requires at least one value")
    position = (len(values) - 1) * q
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def f1(tp, fp, fn):
    denominator = 2.0 * tp + fp + fn
    return np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(np.asarray(denominator, float)),
        where=denominator > 0,
    )


def precision(tp, fp):
    denominator = tp + fp
    return np.divide(
        tp,
        denominator,
        out=np.zeros_like(np.asarray(denominator, float)),
        where=denominator > 0,
    )


def load_oof(path: Path) -> pd.DataFrame:
    with np.load(path) as payload:
        frame = pd.DataFrame(
            {
                "segment_name": payload["segment"].astype(str),
                "query_index": payload["query_index"].astype(int),
                "frame_index": payload["frame_index"].astype(int),
                "native_margin": payload["native_margin"].astype(float),
                "outcome": payload["outcome"].astype(int),
                "candidate_error": payload["candidate_error"].astype(float),
                "gt_visible": payload["gt_visible"].astype(bool),
                "quality_score": payload["quality_score"].astype(float),
                "quality_logit": payload["quality_logit"].astype(float),
                "fold": payload["fold"].astype(int),
            }
        )
    if frame.empty or frame.segment_name.nunique() == 0:
        raise ValueError("OOF score file must contain at least one candidate-bearing group")
    if frame.duplicated(["segment_name", "query_index", "frame_index"]).any():
        raise ValueError("duplicate OOF row identity")
    if not np.isfinite(frame[["native_margin", "quality_score", "quality_logit"]]).all().all():
        raise ValueError("non-finite OOF score")
    return frame


def crossfit_logistic(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    output = np.full(len(frame), np.nan, float)
    y = (frame.outcome.to_numpy() == 0).astype(int)
    fold_values = frame.fold.to_numpy()
    for fold in sorted(frame.fold.unique()):
        train = fold_values != fold
        holdout = ~train
        scaler = StandardScaler().fit(frame.loc[train, features])
        model = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        )
        model.fit(scaler.transform(frame.loc[train, features]), y[train])
        output[holdout] = model.predict_proba(
            scaler.transform(frame.loc[holdout, features])
        )[:, 1]
    if not np.isfinite(output).all():
        raise RuntimeError("cross-fitted fusion score contains non-finite values")
    return output


def bootstrap_video_delta(per_video: pd.DataFrame, replicates: int = 10_000) -> dict:
    values = per_video.delta_f1.to_numpy(float)
    rng = np.random.RandomState(0)
    draws = rng.randint(0, len(values), size=(replicates, len(values)))
    samples = values[draws].mean(axis=1)
    return {
        "mean_video_delta_f1": float(values.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "probability_positive": float(np.mean(samples > 0)),
        "replicates": replicates,
    }


def bootstrap_metric_delta(
    per_video: pd.DataFrame,
    column: str,
    prefix: str,
    replicates: int = 10_000,
) -> dict:
    values = per_video[column].to_numpy(float)
    rng = np.random.RandomState(0)
    draws = rng.randint(0, len(values), size=(replicates, len(values)))
    samples = values[draws].mean(axis=1)
    return {
        f"mean_video_{prefix}": float(values.mean()),
        f"{prefix}_ci95_lower": float(np.quantile(samples, 0.025)),
        f"{prefix}_ci95_upper": float(np.quantile(samples, 0.975)),
        f"{prefix}_probability_positive": float(np.mean(samples > 0)),
    }


def official_tap_transition(
    native: pd.DataFrame,
    accepted_c: np.ndarray,
    accepted_l: np.ndarray,
    accepted_a: np.ndarray,
    accepted_within: np.ndarray,
) -> dict[str, np.ndarray]:
    """Update candidate-preserving TAP metrics from accepted rejection events.

    Online TAPIR always retains the same point coordinate. Re-admission changes
    the predicted occlusion state, hence Jaccard and Occlusion Accuracy, while
    Points Within Threshold remains identical to Native.
    """

    required = {
        "eval_rows",
        "native_valid",
        "C",
        "L",
        "native_average_jaccard",
        "native_average_pts_within_thresh",
        "native_occlusion_accuracy",
        *(f"native_jaccard_{radius}" for radius in TAP_THRESHOLDS),
    }
    missing = required.difference(native.columns)
    if missing:
        raise ValueError(f"native table missing official TAP fields: {sorted(missing)}")
    if accepted_within.shape != (len(TAP_THRESHOLDS), len(native)):
        raise ValueError("accepted_within has the wrong threshold/video shape")

    eval_rows = native.eval_rows.to_numpy(float)
    predicted_visible = native.native_valid.to_numpy(float)
    rejected_gt_visible = (native.C + native.L).to_numpy(float)
    native_oa = native.native_occlusion_accuracy.to_numpy(float)
    gt_visible = (
        native_oa * eval_rows
        - eval_rows
        + predicted_visible
        + 2.0 * rejected_gt_visible
    )
    accepted = accepted_c + accepted_l + accepted_a
    new_predicted_visible = predicted_visible + accepted
    new_jaccards = []
    for threshold_index, radius in enumerate(TAP_THRESHOLDS):
        native_jaccard = native[f"native_jaccard_{radius}"].to_numpy(float)
        native_tp = native_jaccard * (gt_visible + predicted_visible) / (
            1.0 + native_jaccard
        )
        new_tp = native_tp + accepted_within[threshold_index]
        denominator = gt_visible + new_predicted_visible - new_tp
        new_jaccards.append(
            np.divide(
                new_tp,
                denominator,
                out=np.zeros_like(new_tp),
                where=denominator > 0,
            )
        )
    psrm_average_jaccard = np.mean(np.stack(new_jaccards, axis=0), axis=0)
    psrm_occlusion_accuracy = native_oa + (
        accepted_c + accepted_l - accepted_a
    ) / eval_rows
    native_points = native.native_average_pts_within_thresh.to_numpy(float)
    return {
        "native_average_jaccard": native.native_average_jaccard.to_numpy(float),
        "psrm_average_jaccard": psrm_average_jaccard,
        "native_occlusion_accuracy": native_oa,
        "psrm_occlusion_accuracy": psrm_occlusion_accuracy,
        "native_average_pts_within_thresh": native_points,
        "psrm_average_pts_within_thresh": native_points.copy(),
    }


def scan_system(
    frame: pd.DataFrame,
    score_column: str,
    native_per_video: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    score = frame[score_column].to_numpy(float)
    order = np.argsort(-score, kind="mergesort")
    outcome = frame.outcome.to_numpy(int)[order]
    candidate_error = frame.candidate_error.to_numpy(float)[order]
    gt_visible = frame.gt_visible.to_numpy(bool)[order]
    segment = frame.segment_name.to_numpy(str)[order]
    score_sorted = score[order]

    native = native_per_video.drop_duplicates("segment_name").set_index("segment_name").sort_index()
    if not set(frame.segment_name).issubset(set(native.index)):
        raise ValueError("OOF scores contain groups absent from the native table")
    names = native.index.to_numpy(str)
    group_index = {name: index for index, name in enumerate(names)}
    group = np.fromiter((group_index[name] for name in segment), np.int32, len(segment))
    tp0 = native.native_TP.to_numpy(float)
    tn0 = native.native_TN.to_numpy(float)
    fp0 = native.native_FP.to_numpy(float)
    fn0 = native.native_FN.to_numpy(float)
    base_group_f1 = f1(tp0, fp0, fn0)
    base_precision = float(precision(tp0.sum(), fp0.sum()))
    base_f1 = float(f1(tp0.sum(), fp0.sum(), fn0.sum()))

    c_by = np.zeros(len(names), int)
    l_by = np.zeros(len(names), int)
    a_by = np.zeros(len(names), int)
    bad_by = np.zeros(len(names), int)
    delta_by = np.zeros(len(names), float)
    sorted_delta = [0.0] * len(names)
    C = L = A = 0
    curve = []
    selected = {
        "threshold": float("inf"),
        "accepted": 0,
        "C": 0,
        "L": 0,
        "A": 0,
        "precision": base_precision,
        "f1": base_f1,
        "delta_precision": 0.0,
        "delta_f1": 0.0,
        "added_invisible_rate": 0.0,
        "negative_video_rate": 0.0,
        "video_delta_f1_q10": 0.0,
        "risk_pass": True,
    }
    best_key = (0, 0, 0.0, float("inf"))
    best_accepted = 0
    evaluated_threshold_count = 0
    feasible_threshold_count = 1
    negative_video_count = 0
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score_sorted[j] == score_sorted[i]:
            j += 1
        block_group = group[i:j]
        block_outcome = outcome[i:j]
        block_error = candidate_error[i:j]
        block_visible = gt_visible[i:j]
        affected = np.unique(block_group)
        old = {index: delta_by[index] for index in affected}
        C += int(np.sum(block_outcome == 0))
        L += int(np.sum(block_outcome == 1))
        A += int(np.sum(block_outcome == 2))
        for index in affected:
            mask = block_group == index
            c_by[index] += int(np.sum(block_outcome[mask] == 0))
            l_by[index] += int(np.sum(block_outcome[mask] == 1))
            a_by[index] += int(np.sum(block_outcome[mask] == 2))
            bad_by[index] = l_by[index] + a_by[index]
            if old[index] < 0:
                negative_video_count -= 1
            sorted_delta.pop(bisect.bisect_left(sorted_delta, old[index]))
            current = float(
                f1(tp0[index] + c_by[index], fp0[index] + bad_by[index], fn0[index] - c_by[index])
                - base_group_f1[index]
            )
            delta_by[index] = current
            if current < 0:
                negative_video_count += 1
            bisect.insort(sorted_delta, current)
        current_precision = float(precision(tp0.sum() + C, fp0.sum() + L + A))
        current_f1 = float(f1(tp0.sum() + C, fp0.sum() + L + A, fn0.sum() - C))
        row = {
            "threshold": float(score_sorted[i]),
            "accepted": int(j),
            "C": C,
            "L": L,
            "A": A,
            "precision": current_precision,
            "f1": current_f1,
            "delta_precision": current_precision - base_precision,
            "delta_f1": current_f1 - base_f1,
            "added_invisible_rate": A / tn0.sum(),
            "negative_video_rate": negative_video_count / len(names),
            "video_delta_f1_q10": sorted_linear_quantile(sorted_delta, 0.10),
        }
        evaluated_threshold_count += 1
        row["risk_pass"] = bool(
            row["added_invisible_rate"] <= RISK["max_added_invisible_rate"]
            and row["delta_precision"] >= RISK["min_delta_precision"]
            and row["negative_video_rate"] <= RISK["max_negative_video_rate"]
            and row["video_delta_f1_q10"] >= RISK["min_video_delta_f1_q10"]
        )
        if row["risk_pass"]:
            feasible_threshold_count += 1
            key = (C, -A, row["delta_precision"], row["threshold"])
            if key > best_key:
                best_key, selected = key, row.copy()
                best_accepted = j
                row["is_new_best"] = True
            else:
                row["is_new_best"] = False
            # The exact scan evaluates every threshold, but the audit table only
            # retains feasible rows to avoid materializing >1M Python dicts.
            curve.append(row)
        i = j
    selected["evaluated_threshold_count"] = evaluated_threshold_count + 1
    selected["feasible_threshold_count"] = feasible_threshold_count

    accepted_group = group[:best_accepted]
    accepted_outcome = outcome[:best_accepted]
    accepted_error = candidate_error[:best_accepted]
    accepted_visible = gt_visible[:best_accepted]
    best_c = np.bincount(
        accepted_group,
        weights=(accepted_outcome == 0).astype(int),
        minlength=len(names),
    ).astype(int)
    best_l = np.bincount(
        accepted_group,
        weights=(accepted_outcome == 1).astype(int),
        minlength=len(names),
    ).astype(int)
    best_a = np.bincount(
        accepted_group,
        weights=(accepted_outcome == 2).astype(int),
        minlength=len(names),
    ).astype(int)
    best_bad = best_l + best_a
    best_within = np.stack(
        [
            np.bincount(
                accepted_group,
                weights=(accepted_visible & (accepted_error <= radius)).astype(int),
                minlength=len(names),
            ).astype(int)
            for radius in TAP_THRESHOLDS
        ],
        axis=0,
    )
    final_group_f1 = f1(tp0 + best_c, fp0 + best_bad, fn0 - best_c)
    official = official_tap_transition(
        native.reset_index(), best_c, best_l, best_a, best_within
    )
    per_video = pd.DataFrame(
        {
            "segment_name": names,
            "native_f1": base_group_f1,
            "psrm_f1": final_group_f1,
            "delta_f1": final_group_f1 - base_group_f1,
            "accepted_C": best_c,
            "accepted_L": best_l,
            "accepted_A": best_a,
            "accepted_error": best_bad,
            **official,
        }
    )
    per_video["delta_average_jaccard"] = (
        per_video.psrm_average_jaccard - per_video.native_average_jaccard
    )
    per_video["delta_occlusion_accuracy"] = (
        per_video.psrm_occlusion_accuracy - per_video.native_occlusion_accuracy
    )
    per_video["delta_average_pts_within_thresh"] = 0.0
    return selected, pd.DataFrame(curve), per_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--native-per-video", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame = load_oof(args.oof)
    native = pd.read_csv(args.native_per_video)
    fold_manifest = pd.read_csv(args.fold_manifest)
    segment_column = "segment_name" if "segment_name" in fold_manifest else "segment"
    if segment_column not in fold_manifest:
        raise ValueError("fold manifest must contain segment or segment_name")
    design_ids = set(fold_manifest[segment_column].astype(str))
    if not set(frame.segment_name).issubset(design_ids):
        raise ValueError("OOF scores contain a video outside the Design fold manifest")
    if not design_ids.issubset(set(native.segment_name.astype(str))):
        raise ValueError("native table does not cover every candidate-bearing manifest group")
    frame["score_margin"] = crossfit_logistic(frame, ["native_margin"])
    frame["score_b1_margin"] = crossfit_logistic(
        frame, ["quality_logit", "native_margin"]
    )
    systems = {
        "Margin": "score_margin",
        "P-SRM": "quality_score",
        "P-SRM+Margin": "score_b1_margin",
    }
    y = (frame.outcome == 0).astype(int).to_numpy()
    summaries = []
    for system, column in systems.items():
        selected, curve, per_video = scan_system(frame, column, native)
        selected.update(
            {
                "system": system,
                "average_precision": float(average_precision_score(y, frame[column])),
                "roc_auc": float(roc_auc_score(y, frame[column])),
                **bootstrap_video_delta(per_video),
                **bootstrap_metric_delta(
                    per_video, "delta_average_jaccard", "delta_average_jaccard"
                ),
                **bootstrap_metric_delta(
                    per_video, "delta_occlusion_accuracy", "delta_occlusion_accuracy"
                ),
                "native_mean_average_jaccard": float(
                    per_video.native_average_jaccard.mean()
                ),
                "psrm_mean_average_jaccard": float(
                    per_video.psrm_average_jaccard.mean()
                ),
                "native_mean_occlusion_accuracy": float(
                    per_video.native_occlusion_accuracy.mean()
                ),
                "psrm_mean_occlusion_accuracy": float(
                    per_video.psrm_occlusion_accuracy.mean()
                ),
                "mean_average_pts_within_thresh": float(
                    per_video.native_average_pts_within_thresh.mean()
                ),
            }
        )
        summaries.append(selected)
        safe = system.lower().replace("+", "_plus_")
        curve.to_csv(args.output_root / f"{safe}_risk_curve.csv", index=False)
        per_video.to_csv(args.output_root / f"{safe}_per_video.csv", index=False)
    frame.to_csv(args.output_root / "BLUEPRINT1_ROW_LEVEL_OOF_SCORES.csv", index=False)
    pd.DataFrame(summaries).to_csv(
        args.output_root / "BLUEPRINT1_DESIGN_SYSTEM_CLOSURE.csv", index=False
    )
    (args.output_root / "BLUEPRINT1_RISK_CONTRACT.json").write_text(
        json.dumps(RISK, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
