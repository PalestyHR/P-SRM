"""Grouped-OOF training entry for P-SRM on Kinetics Design only."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import (
    fold_local_pos_weight,
    fold_local_visibility_pos_weight,
    iter_video_batches,
    load_fold_manifest,
)
from .model import PointQualitySidecar, PointQualitySidecarConfig


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def learning_rate_at_step(
    step: int, total_steps: int, warmup_fraction: float, base_lr: float
) -> float:
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))
    if step < warmup_steps:
        return base_lr * float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def localization_quality_target(
    candidate_error: torch.Tensor, quality_scale: float
) -> torch.Tensor:
    """Map point error to continuous quality, with quality 0.5 at the task tolerance."""

    if quality_scale <= 0.0:
        raise ValueError("quality_scale must be positive")
    return torch.exp(-math.log(2.0) * torch.square(candidate_error / quality_scale))


def binary_metrics(y: np.ndarray, score: np.ndarray, prefix: str) -> dict[str, float]:
    y = np.asarray(y, bool)
    score = np.asarray(score, float)
    if y.size == 0 or np.unique(y).size < 2:
        return {}
    return {
        f"{prefix}_average_precision": float(average_precision_score(y, score)),
        f"{prefix}_roc_auc": float(roc_auc_score(y, score)),
    }


def train_fold(args: argparse.Namespace, fold: int) -> Path:
    records = load_fold_manifest(args.fold_manifest, args.evidence_root)
    outer_train = [record for record in records if record.fold != fold]
    outer_holdout = [record for record in records if record.fold == fold]
    fold_root = args.output_root / f"seed_{args.seed}" / f"fold_{fold}"
    fold_root.mkdir(parents=True, exist_ok=True)
    score_path = fold_root / "oof_scores.npz"
    if score_path.exists() and not args.overwrite:
        print(
            json.dumps(
                {"event": "fold_reused", "aux_mode": args.aux_mode, "seed": args.seed, "fold": fold}
            ),
            flush=True,
        )
        return score_path

    seed_everything(args.seed + fold)
    device = torch.device(args.device)
    config = PointQualitySidecarConfig(
        conv_channels=args.conv_channels,
        hidden_width=args.hidden_width,
        dropout=args.dropout,
        activation=args.activation,
        leaky_relu_slope=args.leaky_relu_slope,
        aux_mode=args.aux_mode,
    )
    model = PointQualitySidecar(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    pos_weight = torch.tensor([fold_local_pos_weight(outer_train)], device=device)
    admission_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    presence_pos_weight = None
    presence_criterion = None
    if args.aux_mode in {"presence", "combined"}:
        presence_pos_weight = torch.tensor(
            [fold_local_visibility_pos_weight(outer_train)], device=device
        )
        presence_criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=presence_pos_weight
        )
    steps_per_epoch = sum(math.ceil(record.rows / args.batch_size) for record in outer_train)
    schedule_epochs = args.schedule_epochs or args.epochs
    total_steps = steps_per_epoch * schedule_epochs
    global_step = 0
    history = []
    model.train()
    print(
        json.dumps(
            {
                "event": "fold_started",
                "aux_mode": args.aux_mode,
                "seed": args.seed,
                "fold": fold,
                "outer_train_rows": int(sum(record.rows for record in outer_train)),
                "outer_holdout_rows": int(sum(record.rows for record in outer_holdout)),
            }
        ),
        flush=True,
    )
    for epoch in range(args.epochs):
        loss_sums = {
            "total": 0.0,
            "admission": 0.0,
            "presence": 0.0,
            "localization_quality": 0.0,
        }
        rows_seen = 0
        for batch in iter_video_batches(
            outer_train,
            args.batch_size,
            seed=args.seed + fold,
            epoch=epoch,
            shuffle=True,
        ):
            lr = learning_rate_at_step(
                global_step, total_steps, args.warmup_fraction, args.learning_rate
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            dense = torch.from_numpy(batch["dense"]).to(device)
            metadata = torch.from_numpy(batch["metadata"]).to(device)
            label = torch.from_numpy(batch["label"]).to(device)
            visible = torch.from_numpy(batch["gt_visible"].astype(np.float32)).to(device)
            candidate_error = torch.from_numpy(batch["candidate_error"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model.forward_with_aux(dense, metadata)
            admission_loss = admission_criterion(outputs["admission"], label)
            presence_loss = admission_loss.new_zeros(())
            if presence_criterion is not None:
                presence_loss = presence_criterion(outputs["presence"], visible)
            localization_loss = admission_loss.new_zeros(())
            visible_mask = visible.bool()
            if args.aux_mode in {"quality", "combined"} and visible_mask.any():
                quality_target = localization_quality_target(
                    candidate_error[visible_mask], args.quality_scale
                )
                localization_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs["localization_quality"][visible_mask], quality_target
                )
            loss = (
                admission_loss
                + args.presence_weight * presence_loss
                + args.quality_weight * localization_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            optimizer.step()
            count = int(label.numel())
            loss_sums["total"] += float(loss.detach()) * count
            loss_sums["admission"] += float(admission_loss.detach()) * count
            loss_sums["presence"] += float(presence_loss.detach()) * count
            loss_sums["localization_quality"] += float(localization_loss.detach()) * count
            rows_seen += count
            global_step += 1
        history.append(
            {
                "epoch": epoch + 1,
                "loss": loss_sums["total"] / rows_seen,
                "admission_loss": loss_sums["admission"] / rows_seen,
                "presence_loss": loss_sums["presence"] / rows_seen,
                "localization_quality_loss": (
                    loss_sums["localization_quality"] / rows_seen
                ),
            }
        )
        print(
            json.dumps(
                {
                    "event": "epoch_completed",
                    "aux_mode": args.aux_mode,
                    "seed": args.seed,
                    "fold": fold,
                    **history[-1],
                }
            ),
            flush=True,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "config": config.__dict__,
                "epoch": epoch + 1,
                "seed": args.seed,
                "fold": fold,
            },
            fold_root / "final_epoch.pt",
        )

    source_fields = [
        "segment",
        "query_index",
        "frame_index",
        "native_margin",
        "outcome",
        "candidate_error",
        "gt_visible",
        "fold",
    ]
    fields: dict[str, list[np.ndarray]] = {
        name: [] for name in source_fields + ["quality_logit", "quality_score"]
    }
    if args.aux_mode in {"presence", "combined"}:
        fields["presence_logit"] = []
        fields["presence_score"] = []
    if args.aux_mode in {"quality", "combined"}:
        fields["localization_quality_logit"] = []
        fields["localization_quality_score"] = []
    model.eval()
    with torch.no_grad():
        for batch in iter_video_batches(
            outer_holdout,
            args.batch_size,
            seed=args.seed + fold,
            epoch=0,
            shuffle=False,
        ):
            dense = torch.from_numpy(batch["dense"]).to(device)
            metadata = torch.from_numpy(batch["metadata"]).to(device)
            outputs = model.forward_with_aux(dense, metadata)
            logit = outputs["admission"].cpu().numpy().astype(np.float32)
            fields["quality_logit"].append(logit)
            fields["quality_score"].append((1.0 / (1.0 + np.exp(-logit))).astype(np.float32))
            if "presence" in outputs:
                value = outputs["presence"].cpu().numpy().astype(np.float32)
                fields["presence_logit"].append(value)
                fields["presence_score"].append((1.0 / (1.0 + np.exp(-value))).astype(np.float32))
            if "localization_quality" in outputs:
                value = outputs["localization_quality"].cpu().numpy().astype(np.float32)
                fields["localization_quality_logit"].append(value)
                fields["localization_quality_score"].append(
                    (1.0 / (1.0 + np.exp(-value))).astype(np.float32)
                )
            for name in source_fields:
                fields[name].append(batch[name])
    combined = {name: np.concatenate(parts) for name, parts in fields.items()}
    np.savez(score_path, **combined)
    y = combined["outcome"] == 0
    metrics = {
        "fold": fold,
        "seed": args.seed,
        "rows": int(y.size),
        "positives": int(y.sum()),
        "average_precision": float(average_precision_score(y, combined["quality_score"])),
        "roc_auc": float(roc_auc_score(y, combined["quality_score"])),
        "parameter_count": model.parameter_count,
        "pos_weight": float(pos_weight.item()),
        "presence_pos_weight": (
            float(presence_pos_weight.item()) if presence_pos_weight is not None else None
        ),
        "aux_mode": args.aux_mode,
        "presence_weight": args.presence_weight,
        "quality_weight": args.quality_weight,
        "quality_scale": args.quality_scale,
        "training_epochs": args.epochs,
        "schedule_epochs": schedule_epochs,
        "history": history,
    }
    c_mask = combined["outcome"] == 0
    l_mask = combined["outcome"] == 1
    a_mask = combined["outcome"] == 2
    metrics.update(
        binary_metrics(
            c_mask[c_mask | l_mask],
            combined["quality_score"][c_mask | l_mask],
            "c_vs_l",
        )
    )
    metrics.update(
        binary_metrics(
            c_mask[c_mask | a_mask],
            combined["quality_score"][c_mask | a_mask],
            "c_vs_a",
        )
    )
    if "presence_score" in combined:
        metrics.update(
            binary_metrics(combined["gt_visible"], combined["presence_score"], "presence")
        )
    if "localization_quality_score" in combined:
        visible_rows = combined["gt_visible"].astype(bool)
        metrics.update(
            binary_metrics(
                c_mask[visible_rows],
                combined["localization_quality_score"][visible_rows],
                "localization_c_vs_l",
            )
        )
    (fold_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "fold_completed",
                "aux_mode": args.aux_mode,
                "seed": args.seed,
                "fold": fold,
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
            }
        ),
        flush=True,
    )
    return score_path


def combine_seed(args: argparse.Namespace) -> Path:
    paths = [args.output_root / f"seed_{args.seed}" / f"fold_{fold}" / "oof_scores.npz" for fold in range(5)]
    payloads = [np.load(path) for path in paths]
    keys = payloads[0].files
    combined = {key: np.concatenate([payload[key] for payload in payloads]) for key in keys}
    for payload in payloads:
        payload.close()
    output = args.output_root / f"seed_{args.seed}" / "oof_scores_all.npz"
    np.savez(output, **combined)
    y = combined["outcome"] == 0
    report = {
        "seed": args.seed,
        "rows": int(y.size),
        "videos": int(np.unique(combined["segment"]).size),
        "positives": int(y.sum()),
        "average_precision": float(average_precision_score(y, combined["quality_score"])),
        "roc_auc": float(roc_auc_score(y, combined["quality_score"])),
    }
    (output.parent / "oof_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[42, 3407, 8008], default=42)
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5), default=list(range(5)))
    parser.add_argument("--conv-channels", type=int, choices=[64, 128], default=64)
    parser.add_argument("--hidden-width", type=int, choices=[256, 512], default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", choices=["relu", "leaky_relu"], default="relu")
    parser.add_argument("--leaky-relu-slope", type=float, default=0.01)
    parser.add_argument(
        "--aux-mode",
        choices=["none", "presence"],
        default="none",
    )
    parser.add_argument("--presence-weight", type=float, default=0.5)
    parser.add_argument("--quality-weight", type=float, default=0.5)
    parser.add_argument("--quality-scale", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, choices=[1e-4, 3e-4], default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--schedule-epochs",
        type=int,
        default=None,
        help="cosine-schedule horizon; defaults to the training epoch count",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.epochs <= 0
        or (args.schedule_epochs is not None and args.schedule_epochs < args.epochs)
        or not 0.0 <= args.warmup_fraction < 1.0
        or args.presence_weight < 0.0
        or args.quality_weight < 0.0
        or args.quality_scale <= 0.0
    ):
        raise ValueError("invalid epoch or warmup configuration")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        train_fold(args, fold)
    if set(args.folds) == set(range(5)):
        combine_seed(args)


if __name__ == "__main__":
    main()
