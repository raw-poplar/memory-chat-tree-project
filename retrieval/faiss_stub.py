"""Minimal vector index wrapper with FAISS-compatible API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class VectorIndex:
    dim: int
    storage: Dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, node_ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        for node_id, vec in zip(node_ids, vectors):
            self.storage[node_id] = np.asarray(vec, dtype=np.float32)

    def search(self, query: Sequence[float], topk: int) -> List[Tuple[str, float]]:
        if not self.storage:
            return []
        q = np.asarray(query, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        scores: List[Tuple[str, float]] = []
        for node_id, vec in self.storage.items():
            v = vec / (np.linalg.norm(vec) + 1e-8)
            scores.append((node_id, float(np.dot(q, v))))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:topk]

    def remove(self, node_ids: Sequence[str]) -> None:
        for node_id in node_ids:
            self.storage.pop(node_id, None)
