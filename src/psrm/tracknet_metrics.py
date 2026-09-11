"""TrackNet task metrics and source-only threshold calibration."""

from pathlib import Path
import numpy as np
import pandas as pd

BUDGET = {
    "added_invisible_activation_rate_max": 0.03,
    "delta_precision_min": -0.005,
    "negative_sequence_delta_f1_rate_max": 0.30,
    "sequence_delta_f1_q10_min": -0.002,
}

def load_sequences(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"No per-sequence results under {root}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

def metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Accuracy": (tp + tn) / total if total else 0.0,
        "Precision": precision, "Recall": recall, "F1": f1,
    }

def native_per_sequence(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sequence_id, frame in data.groupby("sequence_id", sort=True):
        tp = int((frame.native_valid & frame.gt_visible & (frame.native_error <= 4)).sum())
        tn = int((~frame.native_valid & ~frame.gt_visible).sum())
        fp = int((frame.native_valid & ((~frame.gt_visible) | (frame.native_error > 4))).sum())
        fn = int((~frame.native_valid & frame.gt_visible).sum())
        rows.append({"sequence_id": sequence_id, **metrics(tp, tn, fp, fn)})
    return pd.DataFrame(rows)

def summarize(data: pd.DataFrame, accepted: np.ndarray) -> tuple[dict, pd.DataFrame]:
    native_seq = native_per_sequence(data)
    candidates = data[~data.native_valid].copy()
    candidates["accepted"] = np.asarray(accepted, dtype=bool)
    selected = candidates[candidates.accepted]
    contributions = selected.groupby("sequence_id").apply(
        lambda x: pd.Series({
            "C": int(x.accept_label.sum()),
            "L": int((x.gt_visible & ~x.accept_label).sum()),
            "A": int((~x.gt_visible).sum()),
        }),
    ) if len(selected) else pd.DataFrame(columns=["C", "L", "A"])
    per = native_seq.merge(contributions, left_on="sequence_id", right_index=True, how="left").fillna({"C": 0, "L": 0, "A": 0})
    for name in ("C", "L", "A"):
        per[name] = per[name].astype(int)
    per["SRM_TP"] = per.TP + per.C
    per["SRM_TN"] = per.TN - per.A
    per["SRM_FP"] = per.FP + per.L + per.A
    per["SRM_FN"] = per.FN - per.C - per.L
    per["Native_F1"] = per.F1
    per["SRM_F1"] = [metrics(*row)["F1"] for row in per[["SRM_TP", "SRM_TN", "SRM_FP", "SRM_FN"]].to_numpy()]
    per["Delta_F1"] = per.SRM_F1 - per.Native_F1

    native = metrics(int(per.TP.sum()), int(per.TN.sum()), int(per.FP.sum()), int(per.FN.sum()))
    srm = metrics(int(per.SRM_TP.sum()), int(per.SRM_TN.sum()), int(per.SRM_FP.sum()), int(per.SRM_FN.sum()))
    c, l, a = int(per.C.sum()), int(per.L.sum()), int(per.A.sum())
    visible_rejects = int(((~data.native_valid) & data.gt_visible).sum())
    invisible_rejects = int(((~data.native_valid) & ~data.gt_visible).sum())
    accepted_count = c + l + a
    summary = {
        "native": native,
        "srm": srm,
        "delta": {key: srm[key] - native[key] for key in ("Accuracy", "Precision", "Recall", "F1")},
        "recovery": {
            "native_rejects": int((~data.native_valid).sum()),
            "visible_native_rejects": visible_rejects,
            "invisible_native_rejects": invisible_rejects,
            "correct_full_map_candidates": int(((~data.native_valid) & data.accept_label).sum()),
            "candidate_ceiling": int(((~data.native_valid) & data.accept_label).sum()) / visible_rejects if visible_rejects else 0.0,
            "accepted": accepted_count, "C": c, "L": l, "A": a,
            "Recovery_Precision": c / accepted_count if accepted_count else 0.0,
            "Recovery_Recall": c / visible_rejects if visible_rejects else 0.0,
            "Invisible_Activation_Rate": a / invisible_rejects if invisible_rejects else 0.0,
            "Acceptance_Coverage": accepted_count / int((~data.native_valid).sum()) if (~data.native_valid).sum() else 0.0,
            "Ceiling_Utilization": c / int(((~data.native_valid) & data.accept_label).sum()) if ((~data.native_valid) & data.accept_label).sum() else 0.0,
        },
        "stability": {
            "negative_sequence_rate": float((per.Delta_F1 < 0).mean()),
            "delta_f1_median": float(per.Delta_F1.median()),
            "delta_f1_q10": float(per.Delta_F1.quantile(0.1)),
            "sequence_count": int(len(per)),
        },
    }
    summary["budget_checks"] = {
        "added_invisible_activation_rate": summary["recovery"]["Invisible_Activation_Rate"] <= BUDGET["added_invisible_activation_rate_max"],
        "delta_precision": summary["delta"]["Precision"] >= BUDGET["delta_precision_min"],
        "negative_sequence_delta_f1_rate": summary["stability"]["negative_sequence_rate"] <= BUDGET["negative_sequence_delta_f1_rate_max"],
        "sequence_delta_f1_q10": summary["stability"]["delta_f1_q10"] >= BUDGET["sequence_delta_f1_q10_min"],
    }
    summary["budget_pass"] = bool(all(summary["budget_checks"].values()))
    return summary, per

def select_threshold(data: pd.DataFrame, scores: np.ndarray) -> tuple[float, dict, pd.DataFrame]:
    candidates = data[~data.native_valid].reset_index(drop=True)
    native = native_per_sequence(data).sort_values("sequence_id").reset_index(drop=True)
    sequence_to_index = {name: i for i, name in enumerate(native.sequence_id)}
    candidate_sequence = candidates.sequence_id.map(sequence_to_index).to_numpy(np.int64)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_sequence = candidate_sequence[order]
    sorted_c = candidates.accept_label.to_numpy(bool)[order]
    sorted_l = (candidates.gt_visible & ~candidates.accept_label).to_numpy(bool)[order]
    sorted_a = (~candidates.gt_visible).to_numpy(bool)[order]
    cumulative_c = np.zeros(len(native), dtype=np.int64)
    cumulative_l = np.zeros(len(native), dtype=np.int64)
    cumulative_a = np.zeros(len(native), dtype=np.int64)
    tp = native.TP.to_numpy(np.int64)
    tn = native.TN.to_numpy(np.int64)
    fp = native.FP.to_numpy(np.int64)
    fn = native.FN.to_numpy(np.int64)
    native_total = metrics(int(tp.sum()), int(tn.sum()), int(fp.sum()), int(fn.sum()))
    native_f1 = native.F1.to_numpy(float)
    visible_rejects = int(((~data.native_valid) & data.gt_visible).sum())
    invisible_rejects = int(((~data.native_valid) & ~data.gt_visible).sum())

    def record(threshold: float) -> dict:
        srm_tp = tp + cumulative_c
        srm_tn = tn - cumulative_a
        srm_fp = fp + cumulative_l + cumulative_a
        srm_fn = fn - cumulative_c - cumulative_l
        aggregate = metrics(int(srm_tp.sum()), int(srm_tn.sum()), int(srm_fp.sum()), int(srm_fn.sum()))
        p = np.divide(srm_tp, srm_tp + srm_fp, out=np.zeros_like(srm_tp, dtype=float), where=(srm_tp + srm_fp) > 0)
        r = np.divide(srm_tp, srm_tp + srm_fn, out=np.zeros_like(srm_tp, dtype=float), where=(srm_tp + srm_fn) > 0)
        f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
        delta = f1 - native_f1
        c, l, a = int(cumulative_c.sum()), int(cumulative_l.sum()), int(cumulative_a.sum())
        accepted = c + l + a
        invisible_rate = a / invisible_rejects if invisible_rejects else 0.0
        delta_precision = aggregate["Precision"] - native_total["Precision"]
        negative_rate = float((delta < 0).mean())
        q10 = float(np.quantile(delta, 0.1))
        budget_pass = (
            invisible_rate <= BUDGET["added_invisible_activation_rate_max"]
            and delta_precision >= BUDGET["delta_precision_min"]
            and negative_rate <= BUDGET["negative_sequence_delta_f1_rate_max"]
            and q10 >= BUDGET["sequence_delta_f1_q10_min"]
        )
        return {
            "threshold": float(threshold), "C": c, "L": l, "A": a,
            "Recovery_Precision": c / accepted if accepted else 0.0,
            "Recovery_Recall": c / visible_rejects if visible_rejects else 0.0,
            "Invisible_Activation_Rate": invisible_rate,
            "Delta_Precision": delta_precision,
            "Delta_F1": aggregate["F1"] - native_total["F1"],
            "Negative_Sequence_Rate": negative_rate,
            "Sequence_Delta_F1_Q10": q10,
            "Budget_Pass": budget_pass,
        }

    rows = [record(1.0)]
    cursor = 0
    while cursor < len(sorted_scores):
        score = float(sorted_scores[cursor])
        end = cursor + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[cursor]:
            end += 1
        for seq, value in zip(sorted_sequence[cursor:end], sorted_c[cursor:end]):
            cumulative_c[seq] += int(value)
        for seq, value in zip(sorted_sequence[cursor:end], sorted_l[cursor:end]):
            cumulative_l[seq] += int(value)
        for seq, value in zip(sorted_sequence[cursor:end], sorted_a[cursor:end]):
            cumulative_a[seq] += int(value)
        rows.append(record(score))
        cursor = end
    if not any(row["threshold"] == 0.0 for row in rows):
        rows.append(record(0.0))
    curve = pd.DataFrame(rows)
    feasible = curve[curve.Budget_Pass].copy()
    if feasible.empty:
        raise RuntimeError("INFEASIBLE UNDER FROZEN RISK BUDGET")
    selected = feasible.sort_values(
        ["C", "A", "Delta_Precision", "threshold"],
        ascending=[False, True, False, False],
    ).iloc[0]
    threshold = float(selected.threshold)
    result, _ = summarize(data, scores >= threshold)
    return threshold, result, curve
