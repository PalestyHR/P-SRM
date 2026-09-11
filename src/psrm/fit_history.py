"""Leakage-free outer-fold residual integration for R1a+R4 with S2/S3.

The validated R1a+R4 checkpoints are read-only.  For each frozen outer fold,
the corresponding anchor model (trained without that holdout fold) is used to
score the outer-train and outer-holdout rows.  A low-capacity affine residual
readout is then fitted only on the outer-train rows and evaluated once on the
outer holdout.  Native margin is intentionally excluded here and remains in
the existing system-closure fusion stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from psrm.data import iter_video_batches, load_fold_manifest
from psrm.model import PointQualitySidecar, PointQualitySidecarConfig


SYSTEM_FEATURES = {
    "ANCHOR_AFFINE": (),
    "ANCHOR_PLUS_S3": ("s3",),
}


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float64)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output.astype(np.float32)


def load_anchor_model(checkpoint: Path, device: torch.device) -> PointQualitySidecar:
    final_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = PointQualitySidecarConfig(**final_payload["config"])
    model = PointQualitySidecar(config)
    model.load_state_dict(final_payload["model"], strict=True)
    model.eval()
    return model.to(device)


def score_anchor(model, records, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    fields = {name: [] for name in ("segment", "query_index", "frame_index", "quality_logit")}
    with torch.no_grad():
        for batch in iter_video_batches(
            records, batch_size, seed=0, epoch=0, shuffle=False
        ):
            dense = torch.from_numpy(batch["dense"]).to(device)
            metadata = torch.from_numpy(batch["metadata"]).to(device)
            fields["quality_logit"].append(
                model(dense, metadata).cpu().numpy().astype(np.float32)
            )
            fields["segment"].append(batch["segment"].astype(str))
            fields["query_index"].append(batch["query_index"].astype(np.int16))
            fields["frame_index"].append(batch["frame_index"].astype(np.int16))
    return {name: np.concatenate(parts) for name, parts in fields.items()}


def load_feature_reservoir(records, s3_root: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "segment",
            "query_index",
            "frame_index",
            "native_margin",
            "outcome",
            "candidate_error",
            "gt_visible",
            "fold",
            "s3",
        )
    }

    def align_rows(
        reference_query: np.ndarray,
        reference_frame: np.ndarray,
        source_query: np.ndarray,
        source_frame: np.ndarray,
        *,
        segment: str,
        source_name: str,
    ) -> np.ndarray:
        reference_keys = list(zip(reference_query.tolist(), reference_frame.tolist()))
        source_keys = list(zip(source_query.tolist(), source_frame.tolist()))
        if len(set(reference_keys)) != len(reference_keys):
            raise ValueError(f"duplicate reference identity: {segment}")
        if len(set(source_keys)) != len(source_keys):
            raise ValueError(f"duplicate {source_name} identity: {segment}")
        source_index = {key: index for index, key in enumerate(source_keys)}
        if set(reference_keys) != set(source_keys):
            raise ValueError(f"{source_name}/reference identity-set mismatch: {segment}")
        return np.asarray([source_index[key] for key in reference_keys], dtype=np.int64)

    s3_paths: dict[str, Path] = {}
    for candidate_path in sorted(s3_root.glob("*.npz")):
        with np.load(candidate_path, allow_pickle=False) as candidate_payload:
            if "segment_name" not in candidate_payload.files:
                continue
            candidate_segment = str(candidate_payload["segment_name"])
        if candidate_segment in s3_paths:
            raise ValueError(f"duplicate S3 segment: {candidate_segment}")
        s3_paths[candidate_segment] = candidate_path

    for record in records:
        if record.segment not in s3_paths:
            raise FileNotFoundError(f"missing S3 segment: {record.segment}")
        s3_path = s3_paths[record.segment]
        with np.load(record.path) as evidence, np.load(s3_path) as s3:
            query = evidence["query_index"].astype(np.int32)
            frame = evidence["frame_index"].astype(np.int32)
            if len(query) != record.rows:
                raise ValueError(f"manifest row mismatch: {record.segment}")
            if not np.isfinite(s3["s3_features"]).all():
                raise ValueError(f"non-finite residual feature: {record.segment}")
            s3_order = align_rows(
                query,
                frame,
                s3["query_index"].astype(np.int32),
                s3["frame_index"].astype(np.int32),
                segment=record.segment,
                source_name="S3",
            )
            fields["segment"].append(np.repeat(record.segment, record.rows))
            fields["query_index"].append(query)
            fields["frame_index"].append(frame)
            fields["native_margin"].append(evidence["native_margin"].astype(np.float32))
            fields["outcome"].append(evidence["outcome"].astype(np.int8))
            fields["candidate_error"].append(evidence["candidate_error"].astype(np.float32))
            fields["gt_visible"].append(evidence["gt_visible"].astype(bool))
            fields["fold"].append(np.full(record.rows, record.fold, np.int8))
            fields["s3"].append(s3["s3_features"][s3_order].astype(np.float32))
    combined = {name: np.concatenate(parts) for name, parts in fields.items()}
    return combined


def aligned_anchor_score(
    score_cache: Path,
    checkpoint: Path,
    records,
    reservoir: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if score_cache.exists():
        with np.load(score_cache) as payload:
            scored = {name: payload[name] for name in payload.files}
    else:
        model = load_anchor_model(checkpoint, device)
        scored = score_anchor(model, records, batch_size, device)
        np.savez(score_cache, **scored)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for name in ("segment", "query_index", "frame_index"):
        if not np.array_equal(scored[name], reservoir[name]):
            raise ValueError(f"anchor/reservoir alignment mismatch in {name}")
    logit = scored["quality_logit"].astype(np.float32)
    if not np.isfinite(logit).all():
        raise ValueError("non-finite anchor logit")
    return logit


def design_matrix(anchor: np.ndarray, reservoir: dict[str, np.ndarray], sources) -> np.ndarray:
    columns = [anchor.reshape(-1, 1)]
    columns.extend(reservoir[source] for source in sources)
    return np.concatenate(columns, axis=1).astype(np.float32)


def fit_fold_readouts(
    fold: int,
    anchor: np.ndarray,
    reservoir: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    folds = reservoir["fold"]
    target = (reservoir["outcome"] == 0).astype(np.int8)
    train = folds != fold
    holdout = folds == fold
    predictions: dict[str, np.ndarray] = {"ANCHOR": anchor[holdout].astype(np.float32)}
    receipts: dict[str, dict] = {}
    for name, sources in SYSTEM_FEATURES.items():
        matrix = design_matrix(anchor, reservoir, sources)
        scaler = StandardScaler().fit(matrix[train])
        classifier = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            random_state=42 + fold,
            solver="lbfgs",
        )
        classifier.fit(scaler.transform(matrix[train]), target[train])
        predictions[name] = classifier.decision_function(
            scaler.transform(matrix[holdout])
        ).astype(np.float32)
        receipts[name] = {
            "fold": fold,
            "sources": ["anchor", *sources],
            "train_rows": int(train.sum()),
            "holdout_rows": int(holdout.sum()),
            "coefficient": classifier.coef_[0].astype(float).tolist(),
            "intercept": float(classifier.intercept_[0]),
            "iterations": int(classifier.n_iter_[0]),
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
        }
    return predictions, receipts


def save_system(
    name: str,
    logits_by_fold: list[np.ndarray],
    reservoir: dict[str, np.ndarray],
    output_root: Path,
) -> dict:
    # Fold concatenation reproduces the established P-SRM OOF ordering.
    order = np.concatenate([np.flatnonzero(reservoir["fold"] == fold) for fold in range(5)])
    logit = np.concatenate(logits_by_fold).astype(np.float32)
    score = sigmoid(logit)
    target = reservoir["outcome"][order] == 0
    destination = output_root / name
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination / "oof_scores_all.npz",
        segment=reservoir["segment"][order],
        query_index=reservoir["query_index"][order],
        frame_index=reservoir["frame_index"][order],
        native_margin=reservoir["native_margin"][order],
        outcome=reservoir["outcome"][order],
        candidate_error=reservoir["candidate_error"][order],
        gt_visible=reservoir["gt_visible"][order],
        fold=reservoir["fold"][order],
        quality_logit=logit,
        quality_score=score,
    )
    top = max(1, int(round(0.01 * len(score))))
    top_outcome = reservoir["outcome"][order][np.argsort(-score, kind="mergesort")[:top]]
    return {
        "system": name,
        "rows": int(len(score)),
        "average_precision": float(average_precision_score(target, score)),
        "roc_auc": float(roc_auc_score(target, score)),
        "top1_C_fraction": float(np.mean(top_outcome == 0)),
        "top1_L_fraction": float(np.mean(top_outcome == 1)),
        "top1_A_fraction": float(np.mean(top_outcome == 2)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--s3-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_root = args.output_root / "anchor_score_cache"
    cache_root.mkdir(exist_ok=True)
    records = load_fold_manifest(args.fold_manifest, args.evidence_root)
    reservoir = load_feature_reservoir(records, args.s3_root)
    device = torch.device(args.device)
    all_predictions = {name: [] for name in ("ANCHOR", *SYSTEM_FEATURES)}
    receipts: list[dict] = []
    for fold in range(5):
        checkpoint = args.anchor_root / f"seed_{args.seed}" / f"fold_{fold}" / "final_model.pt"
        anchor = aligned_anchor_score(
            cache_root / f"anchor_fold_{fold}_all_rows.npz",
            checkpoint,
            records,
            reservoir,
            args.batch_size,
            device,
        )
        predictions, fold_receipts = fit_fold_readouts(fold, anchor, reservoir)
        for name, values in predictions.items():
            all_predictions[name].append(values)
        receipts.extend(fold_receipts.values())
        print(json.dumps({"event": "fold_completed", "fold": fold}), flush=True)
    summaries = [
        save_system(name, logits, reservoir, args.output_root)
        for name, logits in all_predictions.items()
    ]
    (args.output_root / "readout_receipts.json").write_text(
        json.dumps(receipts, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "reader_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "systems": summaries}), flush=True)


if __name__ == "__main__":
    main()
