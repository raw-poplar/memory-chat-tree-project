"""HNSW vector index wrapper with optional persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class HNSWConfig:
    dim: int = 768
    space: str = "cosine"
    max_elements: int = 500000
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 64
    persist_path: str | None = None
    model: str | None = None


@dataclass
class HNSWIndex:
    config: HNSWConfig
    _index: any = field(init=False)
    _id_to_label: Dict[str, int] = field(default_factory=dict)
    _label_to_id: Dict[int, str] = field(default_factory=dict)
    _next_label: int = 0

    def __post_init__(self) -> None:
        try:
            import hnswlib  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError("hnswlib 未安装，无法使用 HNSWIndex。请先 pip install hnswlib。") from exc

        self._lib = hnswlib
        self._index = self._lib.Index(space=self.config.space, dim=self.config.dim)
        persist_path = self.config.persist_path
        if persist_path and Path(persist_path).exists():
            meta = self._read_meta(persist_path)
            if meta and self._meta_compatible(meta):
                try:
                    self._load(persist_path, meta)
                    return
                except Exception:
                    pass
        self._init_empty()

    # Public API -----------------------------------------------------------
    def add(self, node_ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        if not node_ids:
            return
        labels: List[int] = []
        new_vecs: List[np.ndarray] = []
        for node_id, vec in zip(node_ids, vectors):
            if node_id in self._id_to_label:
                continue  # skip duplicates
            label = self._next_label
            self._next_label += 1
            self._id_to_label[node_id] = label
            self._label_to_id[label] = node_id
            labels.append(label)
            new_vecs.append(np.asarray(vec, dtype=np.float32))
        if not labels:
            return
        self._index.add_items(np.vstack(new_vecs), labels)

    def search(self, query: Sequence[float], topk: int) -> List[Tuple[str, float]]:
        if self._next_label == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        labels, distances = self._index.knn_query(q, k=min(topk, self._next_label))
        results: List[Tuple[str, float]] = []
        for label, dist in zip(labels[0], distances[0]):
            node_id = self._label_to_id.get(int(label))
            if node_id is None:
                continue
            # hnswlib returns distance; cosine space returns 1 - cosine_sim
            score = 1.0 - float(dist)
            results.append((node_id, score))
        return results

    def save(self) -> None:
        if not self.config.persist_path:
            return
        path = Path(self.config.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(path))
        meta = {
            "id_to_label": self._id_to_label,
            "next_label": self._next_label,
            "dim": self.config.dim,
            "space": self.config.space,
            "model": self.config.model,
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Internal -------------------------------------------------------------
    def _meta_path(self, persist_path: str) -> Path:
        path = Path(persist_path)
        return path.with_suffix(path.suffix + ".meta.json")

    def _read_meta(self, persist_path: str) -> Optional[Dict[str, object]]:
        meta_path = self._meta_path(persist_path)
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _meta_compatible(self, meta: Dict[str, object]) -> bool:
        if meta.get("dim") != self.config.dim:
            return False
        if meta.get("space") != self.config.space:
            return False
        expected_model = self.config.model
        if expected_model is not None and meta.get("model") != expected_model:
            return False
        return True

    def _init_empty(self) -> None:
        self._index.init_index(
            max_elements=self.config.max_elements,
            ef_construction=self.config.ef_construction,
            M=self.config.m,
        )
        self._index.set_ef(self.config.ef_search)
        self._id_to_label = {}
        self._label_to_id = {}
        self._next_label = 0

    def _load(self, persist_path: str, meta: Dict[str, object]) -> None:
        self._index.load_index(persist_path)
        raw_map = meta.get("id_to_label") or {}
        if isinstance(raw_map, dict):
            self._id_to_label = {str(k): int(v) for k, v in raw_map.items()}
        else:
            self._id_to_label = {}
        self._label_to_id = {int(v): k for k, v in self._id_to_label.items()}
        self._next_label = int(meta.get("next_label", len(self._label_to_id)))
        self._index.set_ef(self.config.ef_search)

