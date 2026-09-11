"""Stable-index P-SRM evidence dataset for official LibAUC pAUC training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import VideoRecord


@dataclass(frozen=True)
class MaterializationReceipt:
    rows: int
    positives: int
    videos: int
    first_segment: str
    last_segment: str


class IndexedEvidenceDataset(Dataset):
    """Materialize legal same-forward evidence while preserving stable row IDs."""

    def __init__(self, records: Sequence[VideoRecord]) -> None:
        if not records:
            raise ValueError("records must be non-empty")
        total = sum(record.rows for record in records)
        self.dense_evidence = np.empty((total, 3, 32, 32), dtype=np.float32)
        self.candidate_metadata = np.empty((total, 4), dtype=np.float32)
        self.labels = np.empty(total, dtype=np.int64)
        cursor = 0
        for record in records:
            with np.load(record.path) as payload:
                n = int(payload["outcome"].size)
                if n != record.rows or str(payload["segment_name"]) != record.segment:
                    raise ValueError(f"fold-manifest mismatch for {record.segment}")
                if n == 0:
                    continue
                stop = cursor + n
                self.dense_evidence[cursor:stop] = payload["dense_evidence"]
                self.candidate_metadata[cursor:stop] = payload["candidate_metadata"]
                self.labels[cursor:stop] = (payload["outcome"] == 0).astype(np.int64)
                cursor = stop
        if cursor != total or np.unique(self.labels).size != 2:
            raise RuntimeError("materialized dataset is incomplete or single-class")
        self.targets = self.labels
        self.receipt = MaterializationReceipt(
            rows=total,
            positives=int(self.labels.sum()),
            videos=len(records),
            first_segment=records[0].segment,
            last_segment=records[-1].segment,
        )

    def __len__(self) -> int:
        return int(self.labels.size)

    def __getitem__(self, index: int):
        index = int(index)
        return (
            self.dense_evidence[index],
            self.candidate_metadata[index],
            self.labels[index],
            index,
        )


def collate_indexed_evidence(batch):
    dense, metadata, labels, indices = zip(*batch)
    return (
        torch.from_numpy(np.stack(dense)),
        torch.from_numpy(np.stack(metadata)),
        torch.as_tensor(labels, dtype=torch.float32),
        torch.as_tensor(indices, dtype=torch.long),
    )
