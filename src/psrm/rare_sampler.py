"""LibAUC 1.4.0 DualSampler compatibility for a class smaller than its quota.

The installed sampler explicitly leaves this case as a TODO. Keep its original
initialization, epoch length, quota and ordinary iterator. In the unsupported
case only, consume shuffled class pools cyclically until each quota is filled.
Stable dataset indices are preserved, including intentional repeated samples.
"""
import json

import numpy as np

from psrm.libauc_compat import import_official_dual_sampler

OfficialDualSampler = import_official_dual_sampler()


class RareClassDualSampler(OfficialDualSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pos_len == 0 or self.neg_len == 0:
            raise ValueError("Both classes must exist; no synthetic labels are allowed")
        self.rare_class = self.pos_len <= self.num_pos or self.neg_len <= self.num_neg
        if self.rare_class:
            print(json.dumps({
                "event": "rare_class_sampler_compat",
                "upstream": "LibAUC 1.4.0 DualSampler",
                "positives": int(self.pos_len), "negatives": int(self.neg_len),
                "positive_quota": int(self.num_pos), "negative_quota": int(self.num_neg),
                "batches": int(self.num_batches),
                "behavior": "cyclic class-pool refill with stable original sample IDs",
            }), flush=True)

    @staticmethod
    def _take(pool, pointer, count):
        parts = []
        remaining = count
        while remaining:
            if pointer == len(pool):
                np.random.shuffle(pool)
                pointer = 0
            take = min(remaining, len(pool) - pointer)
            parts.append(pool[pointer:pointer + take].copy())
            pointer += take
            remaining -= take
        return np.concatenate(parts), pointer

    def __iter__(self):
        if not self.rare_class:
            return super().__iter__()
        self.sampled = np.zeros(self.num_batches * self.batch_size, dtype=np.int64)
        for batch in range(self.num_batches):
            start = batch * self.batch_size
            positive, self.pos_ptr = self._take(self.pos_indices, self.pos_ptr, self.num_pos)
            negative, self.neg_ptr = self._take(self.neg_indices, self.neg_ptr, self.num_neg)
            self.sampled[start:start + self.num_pos] = positive
            self.sampled[start + self.num_pos:start + self.batch_size] = negative
        return iter(self.sampled)
