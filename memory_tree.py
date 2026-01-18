from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class MemoryNode:
    node_id: str
    role: str  # root | leaf | summary
    title: str
    text: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "title": self.title,
            "text": self.text,
            "parent_id": self.parent_id,
            "children": list(self.children or []),
            "created_at": float(self.created_at),
            "metadata": dict(self.metadata or {}),
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "MemoryNode":
        return MemoryNode(
            node_id=str(payload.get("node_id") or payload.get("id") or ""),
            role=str(payload.get("role") or "leaf"),
            title=str(payload.get("title") or ""),
            text=str(payload.get("text") or payload.get("content") or payload.get("summary") or ""),
            parent_id=(str(payload["parent_id"]) if payload.get("parent_id") else None),
            children=[str(x) for x in (payload.get("children") or [])],
            created_at=float(payload.get("created_at") or time.time()),
            metadata=dict(payload.get("metadata") or {}),
        )


class MemoryTree:
    """A minimal hierarchical memory tree.

    - Parent node = abstraction (summary) of its children.
    - Child node = detail/expansion under its parent.
    - Siblings = parallel items under the same parent.
    """

    def __init__(self, *, root_id: str = "root", nodes: Optional[Dict[str, MemoryNode]] = None) -> None:
        self.root_id = root_id
        self.nodes: Dict[str, MemoryNode] = nodes or {}
        if self.root_id not in self.nodes:
            self.nodes[self.root_id] = MemoryNode(
                node_id=self.root_id,
                role="root",
                title="root",
                text="",
                parent_id=None,
                children=[],
            )

    # ------------------------------- IO -------------------------------- #
    @classmethod
    def load(cls, path: Path) -> "MemoryTree":
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        root_id = str(payload.get("root_id") or "root")
        nodes_payload = payload.get("nodes") or {}
        nodes: Dict[str, MemoryNode] = {}
        if isinstance(nodes_payload, dict):
            for node_id, node_data in nodes_payload.items():
                if not isinstance(node_data, dict):
                    continue
                node = MemoryNode.from_dict({**node_data, "node_id": node_data.get("node_id") or node_id})
                if node.node_id:
                    nodes[node.node_id] = node
        elif isinstance(nodes_payload, list):
            for item in nodes_payload:
                if not isinstance(item, dict):
                    continue
                node = MemoryNode.from_dict(item)
                if node.node_id:
                    nodes[node.node_id] = node
        tree = cls(root_id=root_id, nodes=nodes)
        tree._repair_root_children()
        return tree

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "root_id": self.root_id,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def reset(self) -> None:
        root = self.nodes.get(self.root_id)
        self.nodes = {self.root_id: root or MemoryNode(node_id=self.root_id, role="root", title="root", text="")}
        self.nodes[self.root_id].children = []

    # ------------------------------ Helpers ----------------------------- #
    def _root(self) -> MemoryNode:
        return self.nodes[self.root_id]

    def _repair_root_children(self) -> None:
        root = self._root()
        repaired: List[str] = []
        for node_id in list(root.children or []):
            if node_id in self.nodes and node_id != self.root_id:
                repaired.append(node_id)
        root.children = repaired

    def _default_leaf_title(self, user_text: str) -> str:
        t = (user_text or "").strip().replace("\n", " ")
        if not t:
            return "turn"
        if len(t) <= 32:
            return t
        return t[:32] + "..."

    # ---------------------------- Operations ---------------------------- #
    def add_leaf(
        self,
        user_text: str,
        assistant_text: str,
        *,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        node_id = _new_id("leaf")
        title_final = (title or "").strip() or self._default_leaf_title(user_text)
        leaf_text = f"User: {(user_text or '').strip()}\nAssistant: {(assistant_text or '').strip()}".strip()
        node = MemoryNode(
            node_id=node_id,
            role="leaf",
            title=title_final,
            text=leaf_text,
            parent_id=self.root_id,
            children=[],
            metadata=dict(metadata or {}),
        )
        self.nodes[node_id] = node
        self._root().children.append(node_id)
        return node_id

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_root_children(self) -> int:
        return len(self._root().children or [])

    def protected_leaf_ids(self, protect_last_leaves: int) -> set[str]:
        if protect_last_leaves <= 0:
            return set()
        protected: set[str] = set()
        for node_id in reversed(self._root().children or []):
            node = self.nodes.get(node_id)
            if not node:
                continue
            if node.role == "leaf":
                protected.add(node_id)
                if len(protected) >= protect_last_leaves:
                    break
        return protected

    def iter_memory_ids(
        self,
        *,
        protect_last_leaves: int,
        max_items: Optional[int] = None,
    ) -> List[str]:
        root_children = list(self._root().children or [])
        protected = self.protected_leaf_ids(protect_last_leaves)
        candidates = [nid for nid in root_children if nid not in protected and nid != self.root_id]
        if max_items is None:
            return candidates
        if max_items <= 0:
            return []

        summary_ids = [nid for nid in candidates if (self.nodes.get(nid) and self.nodes[nid].role == "summary")]
        leaf_ids = [nid for nid in candidates if (self.nodes.get(nid) and self.nodes[nid].role == "leaf")]
        keep: set[str] = set(summary_ids)
        remaining = max(max_items - len(summary_ids), 0)
        if remaining > 0:
            keep.update(leaf_ids[-remaining:])
        return [nid for nid in root_children if nid in keep]

    def render_node_for_prompt(self, node_id: str) -> Optional[str]:
        node = self.nodes.get(node_id)
        if not node:
            return None
        text = (node.text or "").strip()
        if not text:
            return None
        if node.role == "summary":
            title = (node.title or "").strip()
            if title:
                return f"{title}: {text}"
            return text
        return text

    def render_nodes_for_summary(self, node_ids: Sequence[str]) -> str:
        parts: List[str] = []
        for nid in node_ids:
            node = self.nodes.get(nid)
            if not node:
                continue
            title = (node.title or "").strip()
            text = (node.text or "").strip()
            if not text:
                continue
            if node.role == "summary" and title:
                parts.append(f"[{title}] {text}")
            else:
                parts.append(text)
        return "\n\n".join(parts).strip()

    def compact_oldest(
        self,
        *,
        summarizer: Callable[[str], str],
        chunk_size: int,
        protect_last_leaves: int,
        title: Optional[str] = None,
    ) -> bool:
        if chunk_size <= 1:
            chunk_size = 2
        root = self._root()
        protected = self.protected_leaf_ids(protect_last_leaves)
        candidates: List[str] = []
        for nid in root.children:
            if nid in protected or nid == self.root_id:
                continue
            if nid in self.nodes:
                candidates.append(nid)
            if len(candidates) >= chunk_size:
                break
        if len(candidates) < 2:
            return False

        combined = self.render_nodes_for_summary(candidates)
        if not combined:
            return False
        summary_text = (summarizer(combined) or "").strip()
        if not summary_text:
            return False

        summary_id = _new_id("sum")
        summary_title = (title or "").strip() or f"摘要@{time.strftime('%Y-%m-%d %H:%M:%S')}"
        summary_node = MemoryNode(
            node_id=summary_id,
            role="summary",
            title=summary_title,
            text=summary_text,
            parent_id=self.root_id,
            children=list(candidates),
        )
        self.nodes[summary_id] = summary_node
        for child_id in candidates:
            child = self.nodes.get(child_id)
            if child:
                child.parent_id = summary_id

        # Replace the candidates in root.children with the new summary node.
        indices = [root.children.index(cid) for cid in candidates if cid in root.children]
        insert_at = min(indices) if indices else 0
        for cid in candidates:
            if cid in root.children:
                root.children.remove(cid)
        root.children.insert(insert_at, summary_id)
        return True

    def walk(self, start_id: Optional[str] = None) -> Iterable[MemoryNode]:
        start = start_id or self.root_id
        stack = [start]
        seen: set[str] = set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            node = self.nodes.get(nid)
            if not node:
                continue
            yield node
            for child_id in reversed(node.children or []):
                stack.append(child_id)
