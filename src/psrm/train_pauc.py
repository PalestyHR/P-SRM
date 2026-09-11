"""Faithful P-SRM task port of LibAUC one-way partial-AUC optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from libauc.losses import pAUCLoss
from libauc.optimizers import SOPAs

from .data import iter_video_batches, load_fold_manifest
from .indexed_data import IndexedEvidenceDataset, collate_indexed_evidence
from .libauc_compat import import_official_dual_sampler
from .model import PointQualitySidecar, PointQualitySidecarConfig
from .train_bce import seed_everything


DualSampler = import_official_dual_sampler()


def load_pretrained_model(
    checkpoint: Path,
    device: torch.device,
    head_policy: str = "preserve",
) -> PointQualitySidecar:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = PointQualitySidecarConfig(**payload["config"])
    model = PointQualitySidecar(config)
    model.load_state_dict(payload["model"], strict=True)
    if head_policy != "preserve":
        raise ValueError("The paper recipe preserves the task-trained admission head")
    return model.to(device)


def build_method(args, model, data_len: int, positives: int, device):
    criterion = pAUCLoss(
        mode="1w", data_len=data_len, gamma=args.gamma,
        margin=args.margin, Lambda=args.Lambda, device=device,
    )
    optimizer = SOPAs(
        model.parameters(), lr=args.learning_rate, mode="adam",
        weight_decay=args.weight_decay, device=device,
    )
    return criterion, optimizer


def evaluate(model, records, batch_size: int, device):
    source_fields = [
        "segment", "query_index", "frame_index", "native_margin", "outcome",
        "candidate_error", "gt_visible", "fold",
    ]
    fields = {name: [] for name in source_fields + ["quality_logit", "quality_score"]}
    model.eval()
    with torch.no_grad():
        for batch in iter_video_batches(
            records, batch_size, seed=0, epoch=0, shuffle=False
        ):
            dense = torch.from_numpy(batch["dense"]).to(device)
            metadata = torch.from_numpy(batch["metadata"]).to(device)
            logit = model(dense, metadata).cpu().numpy().astype(np.float32)
            fields["quality_logit"].append(logit)
            fields["quality_score"].append(
                (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)
            )
            for name in source_fields:
                fields[name].append(batch[name])
    return {name: np.concatenate(parts) for name, parts in fields.items()}


def train_fold(args, fold: int) -> Path:
    records = load_fold_manifest(args.fold_manifest, args.evidence_root)
    outer_train = [record for record in records if record.fold != fold]
    outer_holdout = [record for record in records if record.fold == fold]
    fold_root = args.output_root / f"seed_{args.seed}" / f"fold_{fold}"
    fold_root.mkdir(parents=True, exist_ok=True)
    output = fold_root / "oof_scores.npz"
    if output.exists() and not args.overwrite:
        print(json.dumps({"event": "fold_reused", "fold": fold}), flush=True)
        return output

    seed_everything(args.seed + fold)
    device = torch.device(args.device)
    checkpoint = args.pretrained_root / f"seed_{args.seed}" / f"fold_{fold}" / "final_epoch.pt"
    model = load_pretrained_model(checkpoint, device, args.head_policy)
    dataset = IndexedEvidenceDataset(outer_train)
    sampler_class = DualSampler
    if getattr(args, "sampler", "official") == "rare-class":
        from .rare_sampler import RareClassDualSampler
        sampler_class = RareClassDualSampler
    sampler = sampler_class(
        dataset,
        batch_size=args.batch_size,
        labels=dataset.labels,
        sampling_rate=args.sampling_rate,
        random_seed=args.seed + fold,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_indexed_evidence,
        pin_memory=False,
    )
    criterion, optimizer = build_method(
        args, model, len(dataset), int(dataset.labels.sum()), device
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        rows = 0
        positive_rows = 0
        negative_rows = 0
        for step, (dense, metadata, labels, indices) in enumerate(loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            dense = dense.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            indices = indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            probability = torch.sigmoid(model(dense, metadata))
            loss = criterion(probability, labels, indices)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite pAUC loss")
            loss.backward()
            optimizer.step()
            count = int(labels.numel())
            total_loss += float(loss.detach()) * count
            rows += count
            positive_rows += int(labels.sum().item())
            negative_rows += count - int(labels.sum().item())
        receipt = {
            "epoch": epoch + 1,
            "loss": total_loss / rows,
            "rows": rows,
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
        }
        history.append(receipt)
        print(json.dumps({"event": "epoch_completed", "fold": fold, **receipt}), flush=True)

    combined = evaluate(model, outer_holdout, args.eval_batch_size, device)
    np.savez(output, **combined)
    target = combined["outcome"] == 0
    metrics = {
        "fold": fold,
        "method": args.method,
        "rows": int(target.size),
        "positives": int(target.sum()),
        "average_precision": float(average_precision_score(target, combined["quality_score"])),
        "roc_auc": float(roc_auc_score(target, combined["quality_score"])),
        "history": history,
        "dataset_receipt": dataset.receipt.__dict__,
        "libauc_distribution_version": "1.4.0",
        "libauc_sampler_class": f"{DualSampler.__module__}.{DualSampler.__name__}",
        "pretrained_checkpoint": str(checkpoint),
        "config": vars(args),
    }
    metrics["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in metrics["config"].items()}
    (fold_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "config": model.config.__dict__, "metrics": metrics}, fold_root / "final_model.pt")
    print(json.dumps({"event": "fold_completed", "fold": fold, "ap": metrics["average_precision"], "roc_auc": metrics["roc_auc"]}), flush=True)
    return output


def combine(args) -> None:
    paths = [args.output_root / f"seed_{args.seed}" / f"fold_{fold}" / "oof_scores.npz" for fold in range(5)]
    payloads = [np.load(path) for path in paths]
    combined = {key: np.concatenate([payload[key] for payload in payloads]) for key in payloads[0].files}
    for payload in payloads:
        payload.close()
    np.savez(args.output_root / f"seed_{args.seed}" / "oof_scores_all.npz", **combined)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", choices=["kl"], default="kl")
    parser.add_argument(
        "--head-policy",
        choices=["preserve"],
        default="preserve",
        help=(
            "preserve keeps the task-trained admission head"

        ),
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--sampling-rate", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--margin", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--Lambda", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampler", choices=["official", "rare-class"], default="official")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.sampling_rate < 1.0 or args.epochs <= 0:
        raise ValueError("invalid sampling rate or epoch count")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        train_fold(args, fold)
    if set(args.folds) == {0, 1, 2, 3, 4}:
        combine(args)


if __name__ == "__main__":
    main()
