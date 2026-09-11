"""Full-train refit of the frozen R1a plus faithful LibAUC Task-KL recipe.

This entry is used only after grouped-OOF model selection.  It fits on the full
training partition and scores a separately supplied evaluation partition.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from libauc.losses import pAUCLoss
from libauc.optimizers import SOPAs

from psrm.data import (
    fold_local_pos_weight,
    fold_local_visibility_pos_weight,
    iter_video_batches,
    load_fold_manifest,
)
from psrm.indexed_data import IndexedEvidenceDataset, collate_indexed_evidence
from psrm.libauc_compat import import_official_dual_sampler
from psrm.model import PointQualitySidecar, PointQualitySidecarConfig
from psrm.train_bce import learning_rate_at_step, seed_everything


DualSampler = import_official_dual_sampler()


def train_r1a(args, model, records, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    admission = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([fold_local_pos_weight(records)], device=device)
    )
    presence = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([fold_local_visibility_pos_weight(records)], device=device)
    )
    steps_per_epoch = sum(math.ceil(record.rows / args.batch_size) for record in records)
    total_steps = 10 * steps_per_epoch
    step = 0
    history = []
    for epoch in range(10):
        model.train()
        total = admission_total = presence_total = 0.0
        rows = 0
        for batch in iter_video_batches(
            records, args.batch_size, seed=args.seed, epoch=epoch, shuffle=True
        ):
            rate = learning_rate_at_step(step, total_steps, 0.05, 3e-4)
            for group in optimizer.param_groups:
                group["lr"] = rate
            dense = torch.from_numpy(batch["dense"]).to(device)
            metadata = torch.from_numpy(batch["metadata"]).to(device)
            label = torch.from_numpy(batch["label"]).to(device)
            visible = torch.from_numpy(batch["gt_visible"].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_with_aux(dense, metadata)
            loss_admission = admission(output["admission"], label)
            loss_presence = presence(output["presence"], visible) if args.aux_mode == "presence" else torch.zeros((), device=device)
            loss = loss_admission + 0.5 * loss_presence
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite R1a loss")
            loss.backward()
            optimizer.step()
            count = int(label.numel())
            total += float(loss.detach()) * count
            admission_total += float(loss_admission.detach()) * count
            presence_total += float(loss_presence.detach()) * count
            rows += count
            step += 1
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total / rows,
                "admission_loss": admission_total / rows,
                "presence_loss": presence_total / rows,
            }
        )
        print(json.dumps({"event": "r1a_epoch", **history[-1]}), flush=True)
    return history


def train_r4(args, model, records, device):
    dataset = IndexedEvidenceDataset(records)
    sampler_class = DualSampler
    if getattr(args, "sampler", "official") == "rare-class":
        from .rare_sampler import RareClassDualSampler
        sampler_class = RareClassDualSampler
    sampler = sampler_class(
        dataset,
        batch_size=args.batch_size,
        labels=dataset.labels,
        sampling_rate=0.5,
        random_seed=args.seed,
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
    criterion = pAUCLoss(
        mode="1w",
        data_len=len(dataset),
        gamma=0.1,
        margin=0.6,
        Lambda=1.0,
        device=device,
    )
    optimizer = SOPAs(
        model.parameters(),
        lr=3e-6,
        mode="adam",
        weight_decay=2e-4,
        device=device,
    )
    model.train()
    total = 0.0
    rows = 0
    positive = negative = 0
    for dense, metadata, label, index in loader:
        dense = dense.to(device)
        metadata = metadata.to(device)
        label = label.to(device)
        index = index.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(torch.sigmoid(model(dense, metadata)), label, index)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Task-KL loss")
        loss.backward()
        optimizer.step()
        count = int(label.numel())
        total += float(loss.detach()) * count
        rows += count
        positive += int(label.sum().item())
        negative += count - int(label.sum().item())
    receipt = {
        "epoch": 1,
        "loss": total / rows,
        "rows": rows,
        "positive_rows": positive,
        "negative_rows": negative,
    }
    print(json.dumps({"event": "r4_epoch", **receipt}), flush=True)
    return receipt


def score(model, records, batch_size, device):
    fields = {
        name: []
        for name in (
            "segment",
            "query_index",
            "frame_index",
            "native_margin",
            "outcome",
            "candidate_error",
            "gt_visible",
            "quality_logit",
            "quality_score",
        )
    }
    model.eval()
    with torch.no_grad():
        for batch in iter_video_batches(records, batch_size, seed=0, epoch=0, shuffle=False):
            dense = torch.from_numpy(batch["dense"]).to(device)
            metadata = torch.from_numpy(batch["metadata"]).to(device)
            logit = model(dense, metadata).cpu().numpy().astype(np.float32)
            fields["quality_logit"].append(logit)
            fields["quality_score"].append(
                (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)
            )
            for name in fields:
                if name not in {"quality_logit", "quality_score"}:
                    fields[name].append(batch[name])
    return {name: np.concatenate(values) for name, values in fields.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-evidence-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-evidence-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aux-mode", choices=["none","presence"], default="presence")
    parser.add_argument("--skip-kl", action="store_true")
    parser.add_argument("--sampler", choices=["official", "rare-class"], default="official")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_records = load_fold_manifest(args.train_manifest, args.train_evidence_root)
    from .prepare_dataset import evaluation_records
    evaluation_records = evaluation_records(args.evaluation_evidence_root)
    config = PointQualitySidecarConfig(
        conv_channels=64,
        hidden_width=256,
        dropout=0.0,
        activation="relu",
        aux_mode=args.aux_mode,
    )
    model = PointQualitySidecar(config).to(device)
    r1a_history = train_r1a(args, model, train_records, device)
    r4_receipt = {} if args.skip_kl else train_r4(args, model, train_records, device)
    result = score(model, evaluation_records, args.eval_batch_size, device)
    np.savez(args.output_root / "evaluation_scores.npz", **result)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "seed": args.seed,
            "r1a_history": r1a_history,
            "r4_receipt": r4_receipt,
        },
        args.output_root / "quality.pt",
    )
    (args.output_root / "refit_receipt.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "train_rows": int(sum(record.rows for record in train_records)),
                "evaluation_rows": int(sum(record.rows for record in evaluation_records)),
                "parameter_count": model.parameter_count,
                "r1a": {
                    "epochs": 10,
                    "learning_rate": 3e-4,
                    "weight_decay": 1e-4,
                    "warmup_fraction": 0.05,
                    "presence_weight": 0.5 if args.aux_mode == "presence" else 0.0,
                    "aux_mode": args.aux_mode,
                },
                "r4": {
                    "method": "LibAUC one-way Task-KL",
                    "enabled": not args.skip_kl,
                    "epochs": 1,
                    "learning_rate": 3e-6,
                    "weight_decay": 2e-4,
                    "sampling_rate": 0.5,
                    "gamma": 0.1,
                    "margin": 0.6,
                    "Lambda": 1.0,
                    "head_policy": "preserve",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
