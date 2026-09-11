"""KCF task-metric port of the existing exact P-SRM risk scanner.

The threshold loop, risk budgets and lexicographic selection rule are copied
from the point-metric calibration routine. The TAP-specific AJ/OA output
block is replaced by box overlap metrics. No TAP metrics are fabricated.
"""
import argparse
import bisect
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from psrm.point_metrics import (
    RISK, sorted_linear_quantile, f1, precision, load_oof, crossfit_logistic,
    bootstrap_video_delta,
)

def scan_system(
    frame: pd.DataFrame,
    score_column: str,
    native_per_video: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    score = frame[score_column].to_numpy(float)
    order = np.argsort(-score, kind="mergesort")
    outcome = frame.outcome.to_numpy(int)[order]
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
    if tn0.sum() <= 0:
        raise ValueError('native TN count is zero; the unchanged false-activation risk denominator is undefined')
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
    final_group_f1 = f1(tp0 + best_c, fp0 + best_bad, fn0 - best_c)
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
            "native_TP": tp0, "native_TN": tn0, "native_FP": fp0, "native_FN": fn0,
            "psrm_TP": tp0 + best_c, "psrm_TN": tn0 - best_a,
            "psrm_FP": fp0 + best_bad, "psrm_FN": fn0 - best_c,
        }
    )
    return selected, pd.DataFrame(curve), per_video

def evaluate(oof, corpus, output, resume=False):
    output.mkdir(parents=True, exist_ok=resume)
    data = load_oof(oof)
    frames = pd.read_csv(corpus/'frames.csv')
    native = pd.read_csv(corpus/'native_per_video.csv')
    metadata = json.loads((corpus/'corpus.json').read_text())
    lookup = frames.loc[frames.native_valid == 0, ['sequence', 'video_group', 'frame_index', 'candidate_iou']]
    data = data.rename(columns={'segment_name': 'sequence'})
    data = data.merge(lookup, on=['sequence', 'frame_index'], how='left', validate='one_to_one', indicator=True)
    if len(data) != len(lookup) or not (data['_merge'] == 'both').all():
        raise ValueError('OOF rows do not cover exactly the native rejection pool')
    data = data.drop(columns='_merge')
    data['segment_name'] = data.video_group
    data['score_margin'] = crossfit_logistic(data, ['native_margin'])
    data['score_psrm'] = crossfit_logistic(data, ['quality_logit', 'native_margin'])
    systems = {'native_margin': 'score_margin', 'spatial_temporal_quality': 'quality_score', 'psrm': 'score_psrm'}
    y = data.outcome.to_numpy(int) == 0
    summaries = []
    native_index = native.set_index('segment_name')
    for system, score_column in systems.items():
        selected, curve, per_video = scan_system(data, score_column, native)
        admitted = data[score_column].to_numpy(float) >= selected['threshold']
        data[f'accept_{system}'] = admitted
        selected.update({'system': system, 'label_source': metadata['label_source'],
            'evaluation': 'grouped OOF scores with OOF-selected development risk threshold',
            'reader_ap': float(average_precision_score(y, data[score_column])),
            'reader_roc_auc': float(roc_auc_score(y, data[score_column])),
            'native_precision': float(precision(native.native_TP.sum(), native.native_FP.sum())),
            'native_recall': float(precision(native.native_TP.sum(), native.native_FN.sum())),
            'native_f1': float(f1(native.native_TP.sum(), native.native_FP.sum(), native.native_FN.sum())),
            'recall': float(precision(per_video.psrm_TP.sum(), per_video.psrm_FN.sum())),
            **bootstrap_video_delta(per_video)})
        success_add = data.loc[admitted].assign(good=lambda t: t.candidate_iou >= 0.5).groupby('video_group').good.sum()
        names = per_video.segment_name
        base = native_index.loc[names, 'native_iou05_successes'].to_numpy(float)
        total = native_index.loc[names, 'eval_rows'].to_numpy(float)
        added = success_add.reindex(names, fill_value=0).to_numpy(float)
        per_video['native_iou05_success'] = base/total
        per_video['psrm_iou05_success'] = (base+added)/total
        per_video['delta_iou05_success'] = added/total
        selected['native_pooled_iou05_success'] = float(base.sum()/total.sum())
        selected['psrm_pooled_iou05_success'] = float((base+added).sum()/total.sum())
        selected['native_mean_video_iou05_success'] = float((base/total).mean())
        selected['psrm_mean_video_iou05_success'] = float(((base+added)/total).mean())
        rng = np.random.RandomState(0)
        draws = rng.randint(0, len(names), size=(10000, len(names)))
        boot = per_video.delta_iou05_success.to_numpy(float)[draws].mean(axis=1)
        selected['mean_video_iou05_delta_ci_lower'] = float(np.quantile(boot, 0.025))
        selected['mean_video_iou05_delta_ci_upper'] = float(np.quantile(boot, 0.975))
        curve.to_csv(output/f'{system}_risk_curve.csv', index=False)
        per_video.to_csv(output/f'{system}_per_video.csv', index=False)
        summaries.append(selected)
    data.to_csv(output/'oof_scores.csv', index=False)
    admission = data[['sequence','frame_index','accept_psrm']]
    final = frames.merge(admission, on=['sequence','frame_index'], how='left', validate='one_to_one')
    final['accept_psrm'] = final.accept_psrm.fillna(False).astype(bool)
    keep = final.native_valid.to_numpy(bool)
    accept = final.accept_psrm.to_numpy(bool)
    if np.any(keep & accept):
        raise ValueError('P-SRM admission reached a native-valid row')
    final['final_valid'] = keep | accept
    for key in ('x','y','w','h'):
        final[f'final_{key}'] = np.where(accept, final[f'candidate_{key}'], final[f'native_{key}'])
        if not np.array_equal(final.loc[keep, f'final_{key}'], final.loc[keep, f'native_{key}']):
            raise ValueError('native-valid output changed')
        if not np.array_equal(final.loc[accept, f'final_{key}'], final.loc[accept, f'candidate_{key}']):
            raise ValueError('admitted fixed candidate changed')
    final.to_csv(output/'final_outputs.csv', index=False)
    pd.DataFrame(summaries).to_csv(output/'system_results.csv', index=False)
    (output/'evaluation.json').write_text(json.dumps({
        'label_source': metadata['label_source'],
        'metric_definition': 'F1 uses visible fixed-box IoU >= 0.5; IoU05 success counts native-gated outputs against all annotated boxes',
        'official_otb_auc_reported': False,
        'threshold_selection': 'the unchanged development OOF scanner; not an independent test-set threshold evaluation',
        'risk': RISK, 'native_valid_output_identity': True, 'unchanged_admitted_candidates': True,
        'baseline_forwards_during_closure': 0,
    }, indent=2), encoding='utf-8')
    return summaries


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--oof', type=Path, required=True)
    p.add_argument('--corpus-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    print(json.dumps(evaluate(a.oof, a.corpus_root, a.output_root, resume=a.resume), indent=2))
