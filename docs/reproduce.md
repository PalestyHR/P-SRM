# Reproduction

Prepare the data using [data.md](data.md). Each host has a configuration under `configs/`, with its grouped split under `configs/splits/`.

## Prepared layout

```text
data/<host>/
  train/
    evidence/*.npz
    history/*.npz
    folds.csv
    native.csv              # point hosts
    csv/<sport>/*.csv        # TrackNet
  evaluation/
    evidence/*.npz
    history/*.npz
    folds.csv
    native.csv              # point hosts
    csv/<sport>/*.csv        # TrackNet
```

Evidence NPZs contain `segment_name`, `query_index`, `frame_index`, `dense_evidence`, `candidate_metadata`, `native_margin`, `outcome`, `candidate_error`, and `gt_visible`. History NPZs contain the same row identities and `s3_features`. Outcomes are C=0 (visible, correctly localized), L=1 (visible, incorrectly localized), A=2 (invisible). Targets and native metrics are used offline; they do not enter runtime model inputs.

`folds.csv` records the group ID, fold, rejected-row count, positive count and packet filename. Row counts are computed from your exported evidence. The published splits retain only identities and fold assignments. Video groups, including both Jogging targets for KCF, stay together.

## Main configurations

For any point or TrackNet configuration:

```bash
python -m psrm.reproduce evaluate --config configs/tracknetv3_shuttlecock.json --data data/tracknetv3_shuttlecock/evaluation --weights weights/tracknetv3_shuttlecock --output outputs/tracknetv3_shuttlecock --device cuda
```

To retrain with the selected recipe:

```bash
python -m psrm.reproduce train --config configs/tracknetv3_shuttlecock.json --data data/tracknetv3_shuttlecock --output runs/tracknetv3_shuttlecock/seed_42 --seed 42 --device cuda
```

Repeat the training command with seeds `3407` and `8008` in separate output directories for the repeated-seed settings. TAPNext++ uses its recorded seed-42 setting. Never select a seed using evaluation performance.

The pipeline runs ten epochs of presence-supervised quality training, one epoch of LibAUC one-way Task-KL with the task-trained admission head preserved, grouped causal readout fitting, full-source refit, source-only final readout fitting and OOF threshold calibration. It does not fit on evaluation labels.

The numerical recipe is recorded in each configuration. `RareClassDualSampler` is used only by TrackNetV1/RacketVision, matching that experiment: it retains upstream batch quotas, epoch length and stable sample IDs, and cyclically refills a class smaller than its quota. Other configurations use the official LibAUC 1.4.0 sampler. This is not a replacement loss or a change to class proportions.

## KCF / OTB2013

KCF is grouped OOF rather than a full-source model applied to an independent test split. Prepare its corpus under `data/kcf/train`, then score using the five supplied fold models:

```bash
python -m psrm.evaluate_kcf --data data/kcf/train --weights weights/kcf --output outputs/kcf --device cpu
```

This follows the original development-OOF readout fitting and threshold-selection protocol. For retraining:

```bash
python -m psrm.reproduce train --config configs/kcf.json --data data/kcf --output runs/kcf/seed_42 --seed 42 --device cuda
```

## Readout ablations

After a point-host training run, compare Native, M, Q, Q+H, Q+M, Full and M+H using the saved source OOF and evaluation neural scores:

```bash
python -m psrm.ablate_readouts --data data/tapnet --run runs/tapnet/seed_42 --seed 42 --output outputs/tapnet_readouts
```

Use the corresponding Online TAPIR paths for the other point host. Every learned readout and working threshold is fitted on source data. Neural weights are reused across these readout variants.

For the presence-supervision and Task-KL training controls, run the full pipeline in separate output directories with --auxiliary none and/or --task-kl off:

```bash
python -m psrm.reproduce train --config configs/tapnet.json --data data/tapnet --output runs/tapnet/no_aux_no_kl --seed 42 --auxiliary none --task-kl off --device cuda
```

The defaults are --auxiliary presence --task-kl on (Full). Each control trains its own fold and full-source models, fits its own source readouts, and calibrates its threshold on source OOF scores.

## Metrics

APᵣ is average precision over the fixed native-rejection pool, with C as positive and L/A as negative. It is a ranking metric, distinct from the final tracking F1. Final metrics combine the unchanged native acceptances with selected recoveries.

Preserve the task-specific correctness criterion and accounting: point tracking uses its point tolerance and native TAP-Vid conventions; TrackNet uses a 4-pixel tolerance in 512×288; KCF uses visibility and fixed-box IoU ≥ 0.5. The implementation keeps their original FN accounting separate. KCF's supplemental gated overlap result is not the official OTB success-curve AUC.

Point/KCF intervals are paired video-group bootstrap intervals. TrackNet also reports pooled F1 differences with a paired sequence bootstrap. Mean-video F1 and pooled F1 intervals have different estimands and should not be interchanged.

Threshold calibration preserves the paper's risk budgets and tie ordering. Do not recalibrate on independent evaluation data. Floating-point/library changes and native video decoding can affect scores near a threshold; use the recorded reference environments when comparing numerical values.
