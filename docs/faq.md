# Frequently Asked Questions

Implementation notes for **P-SRM: Selective Recovery of Rejected Predictions in Visual Tracking**. Prepared on 19 September 2026. These notes supplement the manuscript and describe the released implementation; they do not change the reported experiments.

## 1. What can P-SRM recover, and what does it leave unchanged?

P-SRM verifies candidates in the original tracker's rejection set. A rejected candidate belongs to C if the target is visible and the candidate satisfies the task's localization criterion, L if the target is visible but mislocalized, and A if the target is invisible. With coordinates fixed, only C can become a correct recovery.

P-SRM changes the validity decision, not the candidate coordinates. Native acceptances remain accepted. Recovered outputs do not update the original tracker or become trusted historical acceptances. Ground-truth labels define supervision and offline metrics; they are not runtime inputs. See the [host interface](https://github.com/PalestyHR/P-SRM/blob/main/docs/interface.md).

## 2. How is the recovery threshold selected?

For the independent-evaluation point and TrackNet settings, selection uses labeled source-domain out-of-fold (OOF) scores. The calibration routine sorts scores in descending order and evaluates the acceptance sets obtained at distinct score values. Candidates with equal scores enter together. Recovery uses `score >= threshold`.

For each acceptance set, the routine computes correct recoveries, incorrect recoveries and changes to the combined native-plus-recovered outputs. It retains thresholds satisfying all four empirical conditions:

| Calibration quantity | Condition | Definition |
| --- | --- | --- |
| Added invisible activation rate | At most 0.03 | Newly accepted A candidates divided by the invisible native-rejected population. TrackNet counts those rejected frames directly; the point/KCF scanners use the native true-negative count. |
| Change in pooled precision | At least -0.005 | Precision of the combined output minus native precision, not precision of the recovery subset. This permits a decrease of at most 0.5 percentage points. |
| Fraction of sequences/videos with negative F1 change | At most 0.30 | Fraction with combined-output F1 strictly below native F1. |
| 10th percentile of sequence/video F1 changes | At least -0.002 | The 10th percentile must be at least -0.2 percentage points. |

Among feasible thresholds, selection first maximizes the number of C recoveries. Ties are resolved, in order, by fewer A recoveries, greater pooled-precision change, and a higher threshold. The scan includes a no-recovery reference; the point/KCF implementation represents it with an infinite threshold, while TrackNet records a threshold-1 reference before accumulating candidates. The scanners should be consulted for exact boundary handling.

These are source-calibration conditions, not guarantees for independent evaluation data. The source-selected threshold is kept fixed for evaluation, where the realized changes may differ. The conditions also do not impose a minimum precision on the recovery subset or require F1 to improve in every sequence.

Sources: [point scanner](https://github.com/PalestyHR/P-SRM/blob/main/src/psrm/point_metrics.py), [TrackNet scanner](https://github.com/PalestyHR/P-SRM/blob/main/src/psrm/tracknet_metrics.py), [KCF scanner](https://github.com/PalestyHR/P-SRM/blob/main/src/psrm/kcf_metrics.py).

## 3. How do OOF fitting, final refitting and evaluation fit together?

The quality network is trained with video-grouped folds. With quality networks fixed, the first readout combines the quality logit and causal history; the second incorporates the first readout's OOF logit and native margin. Readouts use standardized features and class-balanced logistic regression with L2 regularization. The released configuration specifies `C=0.1` and `max_iter=2000`; C is the inverse regularization strength.

Standardization statistics come from each readout's fitting partition and are applied unchanged to its held-out partition. The second readout's pooled OOF scores provide the threshold-calibration scores. For independent external evaluation, the quality model and readouts are fitted on the full source data according to the training pipeline, and the readout standardizers are refitted on that source data. Their parameters and the source-selected threshold are then frozen. The threshold is not retuned on external evaluation labels.

KCF/OTB2013 is a separate grouped-OOF protocol: the released evaluator pools fold predictions, fits the final readout by cross-fitting, and selects and reports the threshold on the OOF corpus. It is not an independent-test evaluation or an outer-fold assessment of threshold selection. The two Jogging targets share a video group. See [reproduction instructions](https://github.com/PalestyHR/P-SRM/blob/main/docs/reproduce.md) and the [KCF evaluator](https://github.com/PalestyHR/P-SRM/blob/main/src/psrm/evaluate_kcf.py).

## 4. Is M equivalent to changing the native rejection threshold?

M uses only the native scalar margin, passed through standardization and a one-dimensional logistic readout. For the two point-tracker controls checked for the manuscript, the final fitted mapping is increasing. Consequently, its final acceptance rule is equivalent to thresholding the native score while preserving all native acceptances.

The fitting procedure is still different from directly scanning raw native scores: the recovery threshold is selected on OOF logistic scores and then applied to the full-source refitted readout. Thus M is a native-score-only control; it does not establish that directly lowering every native threshold is ineffective. A threshold selected by this procedure may produce no additional acceptances.

Full additionally uses spatial quality evidence and history. The supported native interfaces are not all governed by a single scalar threshold: for example, TrackNetV1 includes intensity thresholding and a Hough-circle decoder. See [native decision evidence](https://github.com/PalestyHR/P-SRM/blob/main/docs/interface.md#native-decision-evidence).

## 5. What happens when little or no history is available?

History is causal and uses native outputs. With two accepted positions, a velocity estimate gives the current candidate's deviation from an extrapolated position. Available features are retained when history is shorter; unavailable terms are zero-filled before standardization. Availability flags distinguish zero, one and at least two previous acceptances. Recovery decisions are never fed back as native acceptances.

History feature coordinates follow the host convention: point trackers use the 256-by-256 model domain, TrackNet uses the 512-by-288 evaluation domain, and KCF uses native image coordinates. Use the matching model bundle and exporter rather than interchanging feature conventions. See the [interface documentation](https://github.com/PalestyHR/P-SRM/blob/main/docs/interface.md#causal-history).

## 6. What are the training defaults and the scope of the sensitivity experiment?

The released TAP-Net configuration uses ten initial-training epochs, batch size 256, learning rate 3e-4, weight decay 1e-4, 5% warm-up and auxiliary presence weight 0.5. Ranking refinement preserves the trained admission head and uses one epoch of LibAUC one-way KL training, learning rate 3e-6, weight decay 2e-4, ranking margin 0.6, Lambda 1 and moving-average coefficient 0.1. Consult each host's configuration for its complete recipe and sampler.

The added sensitivity experiment uses seed 3407 on TAP-Net/Kinetics and TrackNetV3/Shuttlecock. It varies one parameter at a time: auxiliary weight {0.25, 0.5, 1}, ranking margin {0.4, 0.6, 0.8}, and Lambda {0.5, 1, 2}. The common default is (0.5, 0.6, 1). Each variant follows the same grouped fitting and calibration rules, with its own affected models, readouts and selected threshold. Its default row is a same-batch measurement, not the main table's three-seed mean.

All tested settings retain AP_r and pooled-F1 gains over Native. TrackNetV3 changes little; TAP-Net's AP_r is more sensitive to auxiliary weight than to the tested ranking-margin and Lambda variations. This is local sensitivity evidence for two configurations and one seed.

Sources: [TAP-Net configuration](https://github.com/PalestyHR/P-SRM/blob/main/configs/tapnet.json) and [reproduction instructions](https://github.com/PalestyHR/P-SRM/blob/main/docs/reproduce.md).

## 7. Why can AP_r improve substantially while recovery or final F1 changes less?

AP_r evaluates the ordering of C versus L/A in the fixed rejection pool across score thresholds. The reported recovery count and final tracking metrics use one source-calibrated threshold. They answer different questions: whether correct candidates can be ranked above errors, and which candidates are actually admitted at the selected operating point.

The main table's correct-candidate recovery rate is recovered C divided by all C in the native rejection set. Recovery-subset precision is recovered C divided by all newly accepted C/L/A candidates. Neither is the precision of the combined tracking output. In TrackNet's raw summaries, `Ceiling_Utilization` corresponds to the former recovery rate; the separate legacy `Recovery_Recall` field uses visible native rejections as its denominator. Do not interchange those fields.

Final F1 can also depend on the task's event accounting. The released evaluators preserve the original task-specific conventions. Likewise, pooled F1 and average per-video F1 are different metrics; their bootstrap intervals should not be interchanged. See [metrics](https://github.com/PalestyHR/P-SRM/blob/main/docs/reproduce.md#metrics).

## 8. What is included in runtime?

Table 1 reports integration-level timing, not just the quality network's forward pass. TAP-Net's FPS is derived from clip-amortized end-to-end latency. Its measured increment is about 4.99 ms per video frame; separate stage-wise profiling measures about 3.94 ms/frame for evidence preparation and 1.08 ms/frame for core recovery. Separate stage measurements change synchronization and need not sum exactly to the end-to-end increment. Neural-tracker inference was timed on an RTX 4090; KCF's reported timing is on CPU.

## 9. Where do the labels come from?

The experiments use existing datasets and their supplied annotations, including TrackNetV3's corrected Shuttlecock Test labels. OTB2013 retains the original boxes and adds target-visibility labels. The supplement's initial labels were generated with Qwen3-VL-2B-Instruct, corrected, and human spot-checked on a subset; this is not a claim of exhaustive manual annotation. Dataset media are obtained from their original sources. See [data preparation](https://github.com/PalestyHR/P-SRM/blob/main/docs/data.md) and the [OTB supplement](https://github.com/PalestyHR/P-SRM/blob/main/annotations/otb2013/README.md).
