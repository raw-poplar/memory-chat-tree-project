from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root on sys.path so this script can be run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_DIR = PROJECT_ROOT / "model"

from memory_tree import MemoryTree  # noqa: E402
from retrieval import HNSWConfig, HNSWIndex, VectorIndex  # noqa: E402
from utils import MemoryBlock, build_prompt, normalize_mem_slot  # noqa: E402
from utils.prompting import (  # noqa: E402
    format_history_block,
    format_sys_block,
    format_user_block,
    get_protocol,
    protocol_special_tokens,
)

warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set.*",
    category=FutureWarning,
)

DEFAULT_STRUCTURED_MEM_HINT = (
    "提示：下面的记忆块是外部记忆检索模块返回的结构化输入（可能包含噪声），请按需参考；不要把标签当作对话内容复述。"
)

STATE_SUMMARY_SYSTEM = (
    "你是一个长期记忆整理器。把输入的对话片段提炼成后续检索有用的“用户事实/偏好/任务状态”。"
    "只保留可从用户信息中可靠推断的内容，不要编造；不要复述对话原文；忽略与长期记忆无关的闲聊/跑题内容。"
    "避免记录敏感隐私（如地址/电话/证件号等）；如不可避免，用 <PRIVATE> 替代。"
    "只输出固定格式的结构化记忆，不要输出额外解释。"
)

HELP_TEXT = """命令：
- /help：显示本帮助
- /exit, /quit：退出
- /reset：清空当前会话 [HIST]（记忆树仍保留）
- /mem_stats：查看记忆树与索引状态
- /mem_clear：清空记忆树与检索索引（慎用，会删除本地落盘文件）
- /mem_tail [n]：查看最近 n 条记忆节点（默认 10）
- /mem_show <id>：查看某条记忆节点的完整内容
- /mem_search <kw>：按关键词搜索记忆节点（最多显示 50 条）
- /show_mem：查看上一轮检索到并注入的记忆（含相似度分数）
- /show_prompt：查看上一轮实际喂给模型的输入（slot prompt 或渲染后的 chat template）
- /debug [on|off]：开关每轮调试输出（也可用 --debug 默认开启）
- /persona <text>：设置 persona（写入系统提示词）
- /instruction <text>：设置 instruction（写入系统提示词）
"""


def _try_reconfigure_stdio_utf8() -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _resolve_model_dir_arg(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if p.exists() and p.is_dir() and not (p / "config.json").exists():
        try:
            subdirs = [c for c in p.iterdir() if c.is_dir()]
        except Exception:
            subdirs = []
        candidates = [c for c in subdirs if (c / "config.json").exists()]
        if len(candidates) == 1:
            return candidates[0]
    return p


def _infer_context_window(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, fallback: int) -> int:
    candidates: List[int] = []
    for attr in ("n_positions", "max_position_embeddings"):
        v = getattr(getattr(model, "config", None), attr, None)
        if isinstance(v, int) and 8 <= v <= 100000:
            candidates.append(v)
    tmax = getattr(tokenizer, "model_max_length", None)
    if isinstance(tmax, int) and 8 <= tmax <= 100000:
        candidates.append(tmax)
    if candidates:
        return min(candidates)
    return int(fallback)


def _infer_embedding_dim(model: AutoModelForCausalLM, fallback: int = 768) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return int(fallback)


def _l2_normalize(vec: List[float]) -> List[float]:
    if not vec:
        return vec
    norm = math.sqrt(sum((float(x) * float(x)) for x in vec))
    if not norm or norm <= 0:
        return vec
    inv = 1.0 / (norm + 1e-12)
    return [float(x) * inv for x in vec]


def _count_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    if not text:
        return 0
    try:
        return max(len(tokenizer.encode(text, add_special_tokens=False)), 0)
    except Exception:
        return max(len(text.split()), 0)


def _truncate_to_tokens(tokenizer: AutoTokenizer, text: str, max_tokens: int) -> str:
    if not text or max_tokens <= 0:
        return ""
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    except Exception:
        return " ".join(text.split()[:max_tokens]).strip()


def _truncate_to_tokens_tail(tokenizer: AutoTokenizer, text: str, max_tokens: int) -> str:
    """Truncate by keeping the *last* max_tokens tokens (useful for queries where recent text matters)."""
    if not text or max_tokens <= 0:
        return ""
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids = token_ids[-max_tokens:]
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    except Exception:
        return " ".join(text.split()[-max_tokens:]).strip()


def _collapse_cjk_spaces(text: str) -> str:
    # Mirrors scripts/chat.py behaviour.
    if not text:
        return text
    import re

    text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)
    return text


def _strip_think(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if "<think>" not in text and "</think>" not in text:
        return text
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.replace("<think>", "").replace("</think>", "").strip()


def _load_model(model_dir: Path, device: str) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    model_kwargs = {}
    if (model_dir / "pytorch_model.bin").exists():
        model_kwargs["use_safetensors"] = False
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir))

    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    return tokenizer, model, torch_device


def _ensure_protocol_tokens(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, protocol) -> None:
    """Ensure slot-tag protocol tokens are not mapped to [UNK] for BERT-like tokenizers."""
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is None:
        return
    try:
        probe = (
            f"{protocol.sys_open} x {protocol.sys_close}\n"
            f"{protocol.hist_open}\n"
            f"{protocol.speaker_user_prefix} x\n"
            f"{protocol.speaker_assistant_prefix} x\n"
            f"{protocol.hist_close}\n"
            f"{protocol.mem_open('session')} x {protocol.mem_close}\n"
            f"{protocol.user_open} x {protocol.user_close}"
        )
        ids = tokenizer.encode(probe, add_special_tokens=False)
    except Exception:
        return
    if unk_id not in ids:
        return
    try:
        tokens = protocol_special_tokens(protocol)
        added = tokenizer.add_special_tokens({"additional_special_tokens": list(tokens)})
    except Exception:
        return
    if added:
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            return
        try:
            unk_vec = model.get_input_embeddings().weight[unk_id].detach().clone()
            with torch.no_grad():
                for tok in tokens:
                    tid = tokenizer.convert_tokens_to_ids(tok)
                    if isinstance(tid, int) and tid >= 0:
                        model.get_input_embeddings().weight[tid].copy_(unk_vec)
        except Exception:
            pass
        print(f"[info] tokenizer had [UNK] for slot tags; added {added} protocol tokens.")


def _generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompt: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=False)
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": float(repetition_penalty),
    }
    if do_sample:
        gen_kwargs.update(
            {
                "temperature": max(float(temperature), 1e-5),
                "top_p": float(top_p),
                "top_k": int(top_k),
            }
        )
    else:
        # Override any non-default sampling settings shipped in generation_config.json to avoid warnings.
        gen_kwargs.update({"temperature": 1.0, "top_p": 1.0, "top_k": 50})

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    decoded = tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return (decoded or "").strip()


def _render_user_assistant_prompt(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            lines.append(content)
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    lines.append("Assistant:")
    return "\n".join(lines).strip()


def _encode_chat_messages(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    *,
    chat_template: str,
    enable_thinking: bool,
) -> tuple[Dict[str, torch.Tensor], int]:
    use_chat_template: bool
    if chat_template == "on":
        use_chat_template = True
    elif chat_template == "off":
        use_chat_template = False
    else:
        use_chat_template = bool(getattr(tokenizer, "chat_template", None))

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=bool(enable_thinking)
            )
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.unsqueeze(0) if input_ids.ndim == 1 else input_ids
                attn = torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attn}, int(input_ids.shape[1])
        except TypeError:
            pass
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=bool(enable_thinking)
            )
            enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            return enc, int(enc["input_ids"].shape[1])
        except Exception:
            pass

    prompt_text = _render_user_assistant_prompt(messages)
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    return enc, int(enc["input_ids"].shape[1])


def _generate_from_enc(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    enc: Dict[str, torch.Tensor],
    input_len: int,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": float(repetition_penalty),
    }
    if do_sample:
        gen_kwargs.update(
            {
                "temperature": max(float(temperature), 1e-5),
                "top_p": float(top_p),
                "top_k": int(top_k),
            }
        )
    else:
        # Override any non-default sampling settings shipped in generation_config.json to avoid warnings.
        gen_kwargs.update({"temperature": 1.0, "top_p": 1.0, "top_k": 50})

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    decoded = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return (decoded or "").strip()


def main() -> None:
    _try_reconfigure_stdio_utf8()
    parser = argparse.ArgumentParser(description="Chat with a persistent hierarchical memory tree.")
    parser.add_argument(
        "--model_dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to a HF save_pretrained model dir (default: ./model; if ./model has a single child dir with config.json, use that).",
    )
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda or specific torch device string")
    parser.add_argument(
        "--context_window",
        type=int,
        default=0,
        help="Total context window tokens (prompt + generation). 0=auto infer from model/tokenizer.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate per turn")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--history_turns", type=int, default=8, help="How many (user,assistant) turns in [HIST]")
    parser.add_argument(
        "--memory_dir",
        default=str(Path(__file__).resolve().parent / "local_memory"),
        help="Directory to store the persistent memory tree JSON.",
    )
    parser.add_argument("--persona", default="", help="Persona text appended in [SYS] (optional)")
    parser.add_argument("--instruction", default="", help="Instruction appended in [SYS] (optional)")
    parser.add_argument(
        "--mem_slot",
        default="session",
        help="Memory slot name used for injected memory blocks (train/session/longterm).",
    )
    parser.add_argument(
        "--mem_max_items",
        type=int,
        default=24,
        help="Max number of memory items to inject each turn (summaries always prioritized).",
    )
    parser.add_argument(
        "--mem_select",
        choices=["time", "vector", "hybrid"],
        default="vector",
        help="How to choose memory blocks: time=chronological, vector=similarity, hybrid=summary+similarity.",
    )
    parser.add_argument(
        "--index_backend",
        choices=["auto", "hnsw", "flat"],
        default="auto",
        help="Vector index backend. auto prefers hnswlib if installed.",
    )
    parser.add_argument(
        "--index_path",
        default="",
        help="Path for HNSW index file (default: <memory_dir>/memory_hnsw.bin).",
    )
    parser.add_argument(
        "--vector_cache_path",
        default="",
        help="Path for vector cache JSON (default: <memory_dir>/memory_vectors.json).",
    )
    parser.add_argument("--embed_max_tokens", type=int, default=256, help="Max tokens when extracting embeddings.")
    parser.add_argument(
        "--retrieval_query_turns",
        type=int,
        default=2,
        help="How many recent history turns to include in retrieval query.",
    )
    parser.add_argument(
        "--retrieval_query_tokens",
        type=int,
        default=96,
        help="Max tokens of retrieval query text before embedding.",
    )
    parser.add_argument(
        "--retrieval_topk",
        type=int,
        default=0,
        help="Override mem_max_items for vector retrieval (0=use mem_max_items).",
    )
    parser.add_argument(
        "--retrieval_min_score",
        type=float,
        default=0.0,
        help="Drop retrieved memories with cosine score below this threshold.",
    )
    parser.add_argument(
        "--retrieval_rerank",
        choices=["off", "keyword", "token_overlap"],
        default="keyword",
        help=(
            "Optional lightweight rerank on top of vector similarity: "
            "keyword boosts candidates that contain alnum keywords from the query; "
            "token_overlap uses token-id set overlap."
        ),
    )
    parser.add_argument(
        "--retrieval_rerank_weight",
        type=float,
        default=0.35,
        help="Rerank bonus weight (added to cosine score when ranking).",
    )
    parser.add_argument(
        "--retrieval_rerank_doc_tokens",
        type=int,
        default=128,
        help="Max tokens per candidate text when computing token_overlap rerank.",
    )
    parser.add_argument(
        "--hybrid_summary_keep",
        type=int,
        default=4,
        help="When mem_select=hybrid: also include this many most-recent summary nodes.",
    )
    parser.add_argument(
        "--expand_children",
        type=int,
        default=0,
        help="When a retrieved memory node is a summary, also inject up to N of its child nodes (details).",
    )
    parser.add_argument(
        "--prompt_format",
        choices=["auto", "slot", "chat"],
        default="auto",
        help="Prompt format. auto uses chat-template messages when available; slot uses [SYS]/[MEM]/[USER] tags.",
    )
    parser.add_argument(
        "--chat_template",
        choices=["auto", "on", "off"],
        default="auto",
        help="When prompt_format=chat/auto: use tokenizer chat template if available (auto), force on, or disable.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="When using chat templates that support it (e.g. Qwen3), allow <think> reasoning output (default: off).",
    )
    parser.add_argument(
        "--slot_tag_lang",
        choices=["en", "zh"],
        default="en",
        help="When prompt_format=slot: use English ([SYS]/[MEM]/[USER]) or Chinese ([系统]/[记忆]/[用户]) slot tags.",
    )
    parser.add_argument(
        "--mem_structured_hint",
        action="store_true",
        help="Append a short instruction that injected memory blocks are structured external inputs (may be noisy).",
    )
    parser.add_argument(
        "--pack_mode",
        choices=["legacy", "slot"],
        default="slot",
        help="Prompt packing mode. slot=enforce per-slot budgets.",
    )
    parser.add_argument("--budget_user_ratio", type=float, default=0.30, help="User budget ratio (slot mode).")
    parser.add_argument("--budget_history_ratio", type=float, default=0.30, help="History budget ratio (slot mode).")
    parser.add_argument("--min_slot_tokens", type=int, default=64, help="Minimum tokens per slot (slot mode).")
    parser.add_argument(
        "--summary_mode",
        choices=["truncate", "llm", "state"],
        default="truncate",
        help="How to build parent summary nodes when over budget: truncate=clip, llm=compress, state=user facts/preferences/tasks.",
    )
    parser.add_argument(
        "--summary_tokens",
        type=int,
        default=128,
        help="Target token length for each parent summary node.",
    )
    parser.add_argument(
        "--summary_input_tokens",
        type=int,
        default=768,
        help="Max tokens of source text fed into the summarizer (truncate beyond this).",
    )
    parser.add_argument(
        "--compact_chunk_size",
        type=int,
        default=4,
        help="How many sibling nodes to merge into a parent summary at once.",
    )
    parser.add_argument(
        "--keep_tokenizer_spaces",
        action="store_true",
        help="Do not collapse spaces between Chinese characters in output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print prompt/retrieval diagnostics each turn (and enable /show_prompt, /show_mem).",
    )
    parser.add_argument(
        "--debug_max_chars",
        type=int,
        default=800,
        help="Max chars to print for debug blocks (0=no truncation).",
    )
    args = parser.parse_args()

    protocol = get_protocol(str(args.slot_tag_lang))

    model_dir = _resolve_model_dir_arg(str(args.model_dir))
    if not model_dir.exists():
        print(f"[error] model_dir not found: {model_dir}")
        print(f"[hint] Put a HF save_pretrained model in: {DEFAULT_MODEL_DIR}")
        print('[hint] Or pass: --model_dir "<path-to-model>"')
        raise SystemExit(2)
    tokenizer, model, device = _load_model(model_dir, args.device)

    inferred_window = _infer_context_window(tokenizer, model, fallback=2048)
    context_window = int(args.context_window) if int(args.context_window or 0) > 0 else inferred_window
    prompt_budget = max(context_window - int(args.max_new_tokens), 64)

    memory_dir = Path(args.memory_dir).resolve()
    memory_path = memory_dir / "memory_tree.json"
    tree = MemoryTree.load(memory_path)

    history: List[Tuple[str, str]] = []
    pending_commit: Optional[Tuple[str, str]] = None

    mem_slot = normalize_mem_slot(args.mem_slot)

    embed_dim = _infer_embedding_dim(model, fallback=768)
    model_fingerprint = str(model_dir)
    vector_cache_path = (
        Path(args.vector_cache_path).resolve() if str(args.vector_cache_path).strip() else (memory_dir / "memory_vectors.json")
    )
    index_path = Path(args.index_path).resolve() if str(args.index_path).strip() else (memory_dir / "memory_hnsw.bin")

    def _load_vector_cache() -> Dict[str, List[float]]:
        if not vector_cache_path.exists():
            return {}
        try:
            payload = json.loads(vector_cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if isinstance(payload, dict) and "vectors" in payload:
            if payload.get("dim") != embed_dim or payload.get("model") != model_fingerprint:
                return {}
            raw = payload.get("vectors") or {}
        elif isinstance(payload, dict):
            raw = payload
        else:
            return {}
        vectors: Dict[str, List[float]] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if not k or not isinstance(v, list):
                    continue
                if embed_dim and len(v) != embed_dim:
                    continue
                try:
                    vectors[str(k)] = [float(x) for x in v]
                except Exception:
                    continue
        return vectors

    def _save_vector_cache(vectors: Dict[str, List[float]]) -> None:
        vector_cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "model": model_fingerprint, "dim": embed_dim, "vectors": vectors}
        tmp = vector_cache_path.with_suffix(vector_cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(vector_cache_path)

    vectors: Dict[str, List[float]] = _load_vector_cache()

    def _get_embedding(text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return []
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max(int(args.embed_max_tokens), 8),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = getattr(outputs, "hidden_states", None)
        if not hidden:
            return []
        last = hidden[-1][0]  # [seq_len, hidden_size]
        mask = inputs.get("attention_mask")
        if mask is not None:
            m = mask[0].unsqueeze(-1).float()
            pooled = (last * m).sum(dim=0) / m.sum(dim=0).clamp_min(1.0)
        else:
            pooled = last.mean(dim=0)
        vec = pooled.detach().float().cpu().tolist()
        return _l2_normalize([float(x) for x in vec])

    index_backend_used: str
    if args.index_backend in {"auto", "hnsw"}:
        try:
            hnsw_cfg = HNSWConfig(
                dim=embed_dim,
                space="cosine",
                persist_path=str(index_path),
                model=model_fingerprint,
            )
            vec_index: object = HNSWIndex(hnsw_cfg)
            index_backend_used = "hnsw"
        except Exception:
            if args.index_backend == "hnsw":
                raise
            vec_index = VectorIndex(dim=embed_dim)
            index_backend_used = "flat"
    else:
        vec_index = VectorIndex(dim=embed_dim)
        index_backend_used = "flat"

    def _index_add(node_ids: List[str], vecs: List[List[float]]) -> None:
        if not node_ids:
            return
        try:
            vec_index.add(node_ids, vecs)  # type: ignore[attr-defined]
        except Exception:
            return
        if index_backend_used == "hnsw":
            try:
                vec_index.save()  # type: ignore[attr-defined]
            except Exception:
                pass

    def _ensure_indexed(node_ids: List[str]) -> None:
        missing: List[str] = []
        for nid in node_ids:
            if not nid or nid == tree.root_id:
                continue
            if nid not in vectors:
                missing.append(nid)
        if not missing:
            return
        new_ids: List[str] = []
        new_vecs: List[List[float]] = []
        for nid in missing:
            text = tree.render_node_for_prompt(nid) or ""
            vec = _get_embedding(text)
            if not vec or (embed_dim and len(vec) != embed_dim):
                continue
            vectors[nid] = vec
            new_ids.append(nid)
            new_vecs.append(vec)
        if new_ids:
            _index_add(new_ids, new_vecs)
            _save_vector_cache(vectors)

    # Bootstrap index from cache (and lazily fill missing vectors later).
    if vectors:
        _index_add(list(vectors.keys()), list(vectors.values()))

    def _should_use_chat_prompt() -> bool:
        if args.prompt_format == "slot":
            return False
        if args.prompt_format == "chat":
            return True
        if args.chat_template == "off":
            return False
        return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")

    use_chat_prompt = _should_use_chat_prompt()
    if not use_chat_prompt:
        _ensure_protocol_tokens(tokenizer, model, protocol)

    debug_enabled = bool(args.debug)
    debug_max_chars = int(args.debug_max_chars)
    last_debug: Dict[str, Any] = {}

    def _dbg_trunc(text: str) -> str:
        t = (text or "").rstrip()
        if debug_max_chars <= 0:
            return t
        if len(t) <= debug_max_chars:
            return t
        return t[:debug_max_chars].rstrip() + "\n... (truncated)"

    def _dbg_print_last_prompt() -> None:
        if not last_debug:
            print("[debug] no prompt captured yet.")
            return
        mode = str(last_debug.get("prompt_format") or "")
        if mode == "slot":
            tok = last_debug.get("prompt_tokens")
            print(f"[debug] prompt_format=slot  prompt_tokens={tok}")
            print(_dbg_trunc(str(last_debug.get("prompt") or "")))
            return
        if mode == "chat":
            tok = last_debug.get("input_tokens")
            print(f"[debug] prompt_format=chat  input_tokens={tok}")
            messages = last_debug.get("messages") or []
            if isinstance(messages, list):
                for m in messages:
                    if not isinstance(m, dict):
                        continue
                    r = str(m.get("role") or "")
                    c = str(m.get("content") or "")
                    print(f"[debug] {r}: {_dbg_trunc(c)}")
            prompt_text = last_debug.get("prompt_text")
            if isinstance(prompt_text, str) and prompt_text.strip():
                print("[debug] rendered_prompt:")
                print(_dbg_trunc(prompt_text))
            return
        print(f"[debug] unknown prompt_format={mode!r}")

    def _dbg_print_last_mem() -> None:
        if not last_debug:
            print("[debug] no retrieval captured yet.")
            return
        q = str(last_debug.get("query_text") or "").strip()
        if q:
            print("[debug] retrieval_query:")
            print(_dbg_trunc(q))
        mem_items = last_debug.get("mem_items") or []
        if not mem_items:
            print("[debug] retrieved_memories: (none)")
            return
        print("[debug] retrieved_memories:")
        if isinstance(mem_items, list):
            for it in mem_items:
                if not isinstance(it, dict):
                    continue
                nid = str(it.get("id") or "")
                role = str(it.get("role") or "")
                title = str(it.get("title") or "")
                score = it.get("score")
                score_str = f"{float(score):.3f}" if isinstance(score, (int, float)) else ""
                preview = str(it.get("preview") or "")
                meta = f"id={nid} role={role}"
                if title:
                    meta += f" title={title}"
                if score_str:
                    meta += f" score={score_str}"
                print(f"[debug] - {meta}\n{_dbg_trunc(preview)}")

    def _set_last_debug(payload: Dict[str, Any]) -> None:
        last_debug.clear()
        last_debug.update(payload)
        if debug_enabled:
            _dbg_print_last_mem()
            _dbg_print_last_prompt()

    def _instruction_with_mem_hint(base_instruction: str, *, has_mem: bool) -> str:
        base = (base_instruction or "").strip()
        if not args.mem_structured_hint or not has_mem:
            return base
        hint = DEFAULT_STRUCTURED_MEM_HINT
        if hint in base:
            return base
        return (base + "\n" + hint).strip() if base else hint

    def _summary_title() -> str:
        prefix = "用户状态" if str(args.summary_mode) == "state" else "摘要"
        return f"{prefix}@{time.strftime('%Y-%m-%d %H:%M:%S')}"

    def _drop_assistant_from_memory_dump(text: str) -> str:
        """Best-effort: keep user parts from leaf 'User/Assistant' dumps to reduce noise for state summaries."""
        t = (text or "").strip()
        if not t:
            return ""
        chunks = [c.strip() for c in t.split("\n\n") if c and c.strip()]
        cleaned: List[str] = []
        for chunk in chunks:
            if "\nAssistant:" in chunk and chunk.lstrip().startswith("User:"):
                cleaned.append(chunk.split("\nAssistant:", 1)[0].strip())
                continue
            if "\n助手:" in chunk and chunk.lstrip().startswith("用户:"):
                cleaned.append(chunk.split("\n助手:", 1)[0].strip())
                continue
            cleaned.append(chunk.strip())
        return "\n\n".join([c for c in cleaned if c]).strip()

    def summarize_text(source: str) -> str:
        source_raw = (source or "").strip()
        if not source_raw:
            return ""
        if str(args.summary_mode) == "state":
            source_prepared = _drop_assistant_from_memory_dump(source_raw)
        else:
            source_prepared = source_raw
        source_prepared = _truncate_to_tokens(tokenizer, source_prepared, int(args.summary_input_tokens))
        if args.summary_mode == "truncate":
            return _truncate_to_tokens(tokenizer, source_prepared, int(args.summary_tokens))

        if str(args.summary_mode) == "state":
            summary_system = STATE_SUMMARY_SYSTEM
            summary_user_prompt = (
                "请从下面的对话片段中，提取长期有用的用户记忆，并按固定格式输出：\n"
                "用户事实：<...或无>\n"
                "用户偏好：<...或无>\n"
                "任务/目标：<...或无>\n"
                "任务状态：<...或无>\n"
                "约束/边界：<...或无>\n"
                "待办：<...或无>\n\n"
                f"对话片段：\n{source_prepared}\n\n"
                f"要求：尽量不超过 {int(args.summary_tokens)} tokens。"
            )
        else:
            summary_system = "你是一个摘要器。只输出摘要文本本身，不要输出多余解释。"
            summary_user_prompt = (
                f"请将以下内容压缩为简洁摘要，保留关键实体、目标、约束；不要编造。\n\n{source_prepared}\n\n"
                f"要求：尽量不超过 {int(args.summary_tokens)} tokens。"
            )

        if use_chat_prompt:
            messages = [
                {"role": "system", "content": summary_system},
                {"role": "user", "content": summary_user_prompt},
            ]
            enc, input_len = _encode_chat_messages(
                tokenizer, messages, chat_template=str(args.chat_template), enable_thinking=False
            )
            reply = _generate_from_enc(
                model,
                tokenizer,
                device,
                enc,
                input_len,
                max_new_tokens=max(int(args.summary_tokens), 32),
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            )
        else:
            prompt = build_prompt(
                summary_user_prompt,
                persona=summary_system,
                instruction="",
                memories=[],
                history_pairs=[],
                max_history_turns=0,
                protocol=protocol,
            )
            reply = _generate(
                model,
                tokenizer,
                device,
                prompt,
                max_new_tokens=max(int(args.summary_tokens), 32),
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            )
        reply = (reply or "").strip()
        if not reply:
            return _truncate_to_tokens(tokenizer, source_prepared, int(args.summary_tokens))
        return _truncate_to_tokens(tokenizer, reply, int(args.summary_tokens))

    def commit_pending_if_any() -> None:
        nonlocal pending_commit
        if not pending_commit:
            return
        u, a = pending_commit
        pending_commit = None
        leaf_id = tree.add_leaf(u, a)
        tree.save(memory_path)
        if leaf_id:
            _ensure_indexed([leaf_id])

    def _build_retrieval_query(user_text: str) -> str:
        turns = max(int(args.retrieval_query_turns), 0)
        parts: List[str] = []
        recent = history[-turns:] if turns > 0 else []
        for u, a in recent:
            u = (u or "").strip()
            a = (a or "").strip()
            if u:
                parts.append(f"User: {u}")
            if a:
                parts.append(f"Assistant: {a}")
        parts.append(f"User: {(user_text or '').strip()}")
        combined = "\n".join(parts).strip()
        # Keep the tail so the current user input isn't truncated away by long recent turns.
        return _truncate_to_tokens_tail(tokenizer, combined, int(args.retrieval_query_tokens))

    def _retrieve_ranked_memory_ids_from_query_vec(
        allowed_ids: List[str],
        query_vec: List[float],
        *,
        query_text: str,
        topk: int,
    ) -> List[Tuple[str, float]]:
        if topk <= 0 or not allowed_ids or not query_vec:
            return []
        _ensure_indexed(allowed_ids)

        allowed_set = set(allowed_ids)
        min_score = float(args.retrieval_min_score)
        rerank_mode = str(args.retrieval_rerank or "off")
        rerank_weight = float(args.retrieval_rerank_weight)
        rerank_doc_tokens = int(args.retrieval_rerank_doc_tokens)

        def _extract_keywords(text: str) -> List[str]:
            import re

            raw = set(re.findall(r"[A-Za-z0-9_]{4,}", text or ""))
            stop = {
                "user",
                "assistant",
                "system",
                "mem",
                "memory",
                "hist",
                "history",
                "prompt",
                "slot",
            }
            return sorted(
                [
                    k
                    for k in raw
                    if any(c.isalpha() for c in k) and (k.lower() not in stop)
                ]
            )

        def _token_id_set(text: str, *, max_tokens: int) -> set[int]:
            text = (text or "").strip()
            if not text or max_tokens <= 0:
                return set()
            try:
                token_ids = tokenizer.encode(text, add_special_tokens=False)[: max_tokens]
                return set(int(x) for x in token_ids if isinstance(x, int))
            except Exception:
                return set()

        def _rerank(candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
            nonlocal last_debug
            if rerank_mode == "off" or not candidates:
                return candidates
            if rerank_weight <= 0:
                return candidates

            if rerank_mode == "keyword":
                kws = _extract_keywords(query_text)
                if not kws:
                    return candidates
                reranked: List[Tuple[str, float, float, int]] = []
                for nid, score in candidates:
                    doc = tree.render_node_for_prompt(nid) or ""
                    hits = sum((1 for k in kws if k in doc))
                    bonus = rerank_weight * (float(hits) / float(len(kws)))
                    reranked.append((nid, score, score + bonus, hits))
                reranked.sort(key=lambda x: x[2], reverse=True)
                if debug_enabled:
                    last_debug["rerank_mode"] = rerank_mode
                    last_debug["rerank_keywords"] = kws[:32]
                    last_debug["rerank_weight"] = rerank_weight
                    last_debug["rerank_top"] = [
                        {"id": nid, "base": float(s), "hits": int(h), "rank": i + 1}
                        for i, (nid, s, _, h) in enumerate(reranked[: min(len(reranked), 10)])
                    ]
                return [(nid, score) for nid, score, _, _ in reranked]

            if rerank_mode == "token_overlap":
                qset = _token_id_set(query_text, max_tokens=max(int(args.retrieval_query_tokens), 8))
                if not qset:
                    return candidates
                doc_max = max(rerank_doc_tokens, 8)
                reranked2: List[Tuple[str, float, float, float]] = []
                for nid, score in candidates:
                    doc = tree.render_node_for_prompt(nid) or ""
                    dset = _token_id_set(doc, max_tokens=doc_max)
                    overlap = 0.0
                    if dset:
                        overlap = len(qset & dset) / float(max(len(qset), 1))
                    bonus = rerank_weight * float(overlap)
                    reranked2.append((nid, score, score + bonus, overlap))
                reranked2.sort(key=lambda x: x[2], reverse=True)
                if debug_enabled:
                    last_debug["rerank_mode"] = rerank_mode
                    last_debug["rerank_weight"] = rerank_weight
                    last_debug["rerank_top"] = [
                        {"id": nid, "base": float(s), "overlap": float(o), "rank": i + 1}
                        for i, (nid, s, _, o) in enumerate(reranked2[: min(len(reranked2), 10)])
                    ]
                return [(nid, score) for nid, score, _, _ in reranked2]

            return candidates

        if index_backend_used == "hnsw":
            oversample = min(max(topk * 8, topk), 256)
            try:
                raw = vec_index.search(query_vec, oversample)  # type: ignore[attr-defined]
            except Exception:
                raw = []
            results: List[Tuple[str, float]] = []
            for nid, score in raw:
                if nid not in allowed_set:
                    continue
                score_f = float(score)
                if score_f < min_score:
                    continue
                results.append((nid, score_f))
            results = _rerank(results)
            return results[:topk]

        # Flat fallback: score only allowed_ids (cheaper than scanning everything).
        q = query_vec
        scored: List[Tuple[str, float]] = []
        for nid in allowed_ids:
            v = vectors.get(nid)
            if not v:
                continue
            score = sum((qi * vi) for qi, vi in zip(q, v))
            if float(score) >= min_score:
                scored.append((nid, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        oversample = min(max(topk * 8, topk), 256)
        candidates = scored[:oversample]
        candidates = _rerank(candidates)
        return candidates[:topk]

    def _expand_summary_children(
        mem_ids: List[str],
        query_vec: List[float],
        *,
        total_limit: int,
        expand_only: Optional[set[str]] = None,
    ) -> List[str]:
        child_k = max(int(args.expand_children), 0)
        if child_k <= 0 or not mem_ids or total_limit <= 0:
            return mem_ids[: max(total_limit, 0)]
        expanded: List[str] = []

        def _child_candidates(summary_id: str) -> List[str]:
            node = tree.nodes.get(summary_id)
            if not node or node.role != "summary":
                return []
            raw = [cid for cid in (node.children or []) if cid and cid != tree.root_id and cid in tree.nodes]
            if not raw:
                return []
            leaf_first = [cid for cid in raw if getattr(tree.nodes.get(cid), "role", "") == "leaf"]
            other = [cid for cid in raw if cid not in set(leaf_first)]
            return leaf_first + other

        for nid in mem_ids:
            if len(expanded) >= total_limit:
                break
            if nid not in expanded:
                expanded.append(nid)
            if len(expanded) >= total_limit:
                continue
            node = tree.nodes.get(nid)
            if not node or node.role != "summary":
                continue
            if expand_only is not None and nid not in expand_only:
                continue

            candidates = _child_candidates(nid)
            if not candidates:
                continue
            _ensure_indexed(candidates)
            scored: List[Tuple[str, float]] = []
            for cid in candidates:
                v = vectors.get(cid)
                if not v:
                    continue
                score = sum((qi * vi) for qi, vi in zip(query_vec, v))
                scored.append((cid, float(score)))
            scored.sort(key=lambda x: x[1], reverse=True)
            for cid, _ in scored[:child_k]:
                if len(expanded) >= total_limit:
                    break
                if cid not in expanded:
                    expanded.append(cid)

        return expanded[: max(total_limit, 0)]

    def _build_slot_packed_prompt(
        user_text: str,
        *,
        history_turns: int,
        memories: List[MemoryBlock],
    ) -> str:
        min_slot = max(int(args.min_slot_tokens), 16)

        sys_instruction = _instruction_with_mem_hint(args.instruction, has_mem=bool(memories))
        sys_line = format_sys_block(args.persona, sys_instruction, protocol=protocol)
        if sys_line:
            prefix = f"{protocol.sys_open} "
            suffix = f" {protocol.sys_close}"
            sys_max = max(prompt_budget - min_slot, 0)
            if _count_tokens(tokenizer, sys_line) > sys_max and sys_line.startswith(prefix) and sys_line.endswith(suffix):
                content = sys_line[len(prefix) : -len(suffix)].strip()
                overhead = _count_tokens(tokenizer, f"{prefix}x{suffix}") - _count_tokens(tokenizer, "x")
                allowed = max(sys_max - overhead, 0)
                truncated = _truncate_to_tokens(tokenizer, content, allowed).strip()
                sys_line = f"{prefix}{truncated}{suffix}" if truncated else None

        sys_tokens = _count_tokens(tokenizer, sys_line) if sys_line else 0
        available = max(prompt_budget - sys_tokens, 0)

        user_ratio = max(min(float(args.budget_user_ratio), 0.95), 0.0)
        hist_ratio = max(min(float(args.budget_history_ratio), 0.95), 0.0)
        if user_ratio + hist_ratio > 0.95:
            scale = 0.95 / max(user_ratio + hist_ratio, 1e-8)
            user_ratio *= scale
            hist_ratio *= scale

        user_budget = min(max(int(available * user_ratio), min_slot if available >= min_slot else available), available)
        remaining = max(available - user_budget, 0)
        hist_budget = min(max(int(available * hist_ratio), min_slot if remaining >= min_slot else remaining), remaining)
        mem_budget = max(available - user_budget - hist_budget, 0)

        # USER block (truncate inside budget).
        user_overhead = _count_tokens(tokenizer, f"{protocol.user_open} x {protocol.user_close}") - _count_tokens(tokenizer, "x")
        allowed_user = max(user_budget - user_overhead, 0)
        user_text_fit = _truncate_to_tokens(tokenizer, user_text, allowed_user)
        user_line = format_user_block(user_text_fit, protocol=protocol)
        user_tokens = _count_tokens(tokenizer, user_line) if user_line else 0
        mem_budget += max(user_budget - user_tokens, 0)

        # HIST block (keep newest pairs first, drop older when overflow).
        pairs = history[-max(history_turns, 0) :] if history_turns > 0 else []

        def _pairs_to_history_lines(pairs_in: List[Tuple[str, str]]) -> List[str]:
            messages: List[Tuple[str, str]] = []
            for u, a in pairs_in:
                u = (u or "").strip()
                a = (a or "").strip()
                if u:
                    messages.append(("user", u))
                if a:
                    messages.append(("assistant", a))
            return format_history_block(messages, protocol=protocol)

        selected_rev: List[Tuple[str, str]] = []
        for u, a in reversed(pairs):
            candidate_pairs = list(reversed(selected_rev + [(u, a)]))
            cand_lines = _pairs_to_history_lines(candidate_pairs)
            cand_text = "\n".join(cand_lines).strip()
            if not cand_text:
                selected_rev.append((u, a))
                continue
            if _count_tokens(tokenizer, cand_text) <= hist_budget:
                selected_rev.append((u, a))
        selected_pairs = list(reversed(selected_rev))
        hist_lines = _pairs_to_history_lines(selected_pairs) if selected_pairs else []
        hist_text = "\n".join(hist_lines).strip()
        hist_tokens = _count_tokens(tokenizer, hist_text) if hist_text else 0
        mem_budget += max(hist_budget - hist_tokens, 0)

        # MEM blocks (pack until mem_budget; truncate last item if needed).
        def _format_mem_line(slot: str, text: str) -> str:
            return f"{protocol.mem_open(slot)} {text} {protocol.mem_close}"

        mem_overhead = _count_tokens(tokenizer, _format_mem_line(mem_slot, "x")) - _count_tokens(tokenizer, "x")
        mem_lines: List[str] = []
        remaining_mem = max(int(mem_budget), 0)
        for mem in memories:
            text = (mem.text or "").strip()
            if not text or remaining_mem <= 0:
                continue
            line = _format_mem_line(mem.slot, text)
            tok = _count_tokens(tokenizer, line)
            if tok <= remaining_mem:
                mem_lines.append(line)
                remaining_mem -= tok
                continue

            allowed_mem_text = max(remaining_mem - mem_overhead, 0)
            if allowed_mem_text <= 0:
                break
            truncated = _truncate_to_tokens(tokenizer, text, allowed_mem_text).strip()
            if not truncated:
                break
            line2 = _format_mem_line(mem.slot, truncated)
            tok2 = _count_tokens(tokenizer, line2)
            if 0 < tok2 <= remaining_mem:
                mem_lines.append(line2)
            break

        # Assemble canonical order: SYS -> MEM -> HIST -> USER
        lines: List[str] = []
        if sys_line:
            lines.append(sys_line)
        lines.extend(mem_lines)
        lines.extend(hist_lines)
        if user_line:
            lines.append(user_line)
        prompt = "\n".join(lines).strip()
        if _count_tokens(tokenizer, prompt) <= prompt_budget:
            return prompt

        # Final guard: trim memory until fit.
        while mem_lines and _count_tokens(tokenizer, prompt) > prompt_budget:
            mem_lines.pop()
            lines = []
            if sys_line:
                lines.append(sys_line)
            lines.extend(mem_lines)
            lines.extend(hist_lines)
            if user_line:
                lines.append(user_line)
            prompt = "\n".join(lines).strip()
        return prompt

    def _build_chat_messages(
        user_text: str,
        *,
        history_turns: int,
        memory_texts: List[str],
    ) -> List[Dict[str, str]]:
        sys_parts: List[str] = []
        persona = (args.persona or "").strip()
        instruction = (args.instruction or "").strip()
        if persona:
            sys_parts.append(persona)
        if instruction:
            sys_parts.append(instruction)
        if memory_texts:
            if args.mem_structured_hint:
                sys_parts.append("提示：下面的记忆列表是外部记忆检索模块返回的结构化输入（可能包含噪声），请按需参考。")
            header = "以下是检索到的相关记忆（可能有噪声，仅供参考）："
            if args.mem_structured_hint:
                header = "以下是检索到的相关记忆（结构化输入，可能有噪声，仅供参考）："
            mem_block = header + "\n" + "\n".join(
                f"- {t.strip()}" for t in memory_texts if t and t.strip()
            )
            sys_parts.append(mem_block.strip())

        messages: List[Dict[str, str]] = []
        if sys_parts:
            messages.append({"role": "system", "content": "\n\n".join(sys_parts).strip()})
        pairs = history[-max(history_turns, 0) :] if history_turns > 0 else []
        for u, a in pairs:
            u = (u or "").strip()
            a = (a or "").strip()
            if u:
                messages.append({"role": "user", "content": u})
            if a:
                messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": (user_text or "").strip()})
        return messages

    def build_fitted_prompt(user_text: str) -> str:
        history_turns = max(int(args.history_turns), 0)
        mem_max_items = max(int(args.mem_max_items), 0) if args.mem_max_items is not None else None

        while True:
            score_map: Dict[str, float] = {}
            query_text: str = ""
            if args.mem_select == "time":
                mem_ids = tree.iter_memory_ids(protect_last_leaves=history_turns, max_items=mem_max_items)
            else:
                candidates = tree.iter_memory_ids(protect_last_leaves=history_turns, max_items=None)
                query_text = _build_retrieval_query(user_text)
                query_vec = _get_embedding(query_text)
                if not query_vec:
                    mem_ids = tree.iter_memory_ids(protect_last_leaves=history_turns, max_items=mem_max_items)
                    ranked = []
                else:
                    if args.mem_select == "hybrid":
                        keep = max(int(args.hybrid_summary_keep), 0)
                        summary_ids = [
                            nid
                            for nid in candidates
                            if (tree.nodes.get(nid) is not None and tree.nodes[nid].role == "summary")
                        ]
                        baseline = summary_ids[-keep:] if keep > 0 else []
                        if mem_max_items is not None:
                            baseline = baseline[-min(len(baseline), mem_max_items) :]
                        remaining = max(int(mem_max_items or 0) - len(baseline), 0)
                        topk = int(args.retrieval_topk) if int(args.retrieval_topk or 0) > 0 else remaining
                        topk = min(max(topk, 0), remaining)
                        ranked = _retrieve_ranked_memory_ids_from_query_vec(
                            candidates, query_vec, query_text=query_text, topk=topk
                        )
                        score_map = {nid: float(score) for nid, score in ranked}
                        retrieved = [nid for nid, _ in ranked]
                        mem_ids = list(dict.fromkeys(baseline + retrieved))
                    else:
                        topk = int(args.retrieval_topk) if int(args.retrieval_topk or 0) > 0 else int(mem_max_items or 0)
                        ranked = _retrieve_ranked_memory_ids_from_query_vec(
                            candidates, query_vec, query_text=query_text, topk=topk
                        )
                        score_map = {nid: float(score) for nid, score in ranked}
                        mem_ids = [nid for nid, _ in ranked]
                if not mem_ids:
                    mem_ids = tree.iter_memory_ids(protect_last_leaves=history_turns, max_items=mem_max_items)
                if query_vec and mem_max_items is not None and mem_max_items > 0:
                        mem_ids = _expand_summary_children(
                            mem_ids,
                            query_vec,
                            total_limit=int(mem_max_items),
                            expand_only=set(score_map.keys()) if score_map else None,
                        )
            mem_texts: List[str] = []
            mem_items: List[Dict[str, Any]] = []
            for nid in mem_ids:
                t = tree.render_node_for_prompt(nid)
                if not t:
                    continue
                role = getattr(tree.nodes.get(nid), "role", "")
                title = getattr(tree.nodes.get(nid), "title", "")
                mem_items.append(
                    {
                        "id": nid,
                        "role": role,
                        "title": title,
                        "score": float(score_map[nid]) if nid in score_map else None,
                        "preview": t,
                    }
                )
                if nid in score_map:
                    mem_texts.append(f"id={nid} role={role} score={score_map[nid]:.3f} {t}")
                else:
                    mem_texts.append(f"id={nid} role={role} {t}")
            memories = [MemoryBlock(slot=mem_slot, text=t) for t in mem_texts]
            sys_instruction = _instruction_with_mem_hint(args.instruction, has_mem=bool(memories))
            if args.pack_mode == "legacy":
                prompt = build_prompt(
                    user_text,
                    memories=memories,
                    persona=args.persona,
                    instruction=sys_instruction,
                    history_pairs=history,
                    max_history_turns=history_turns,
                    protocol=protocol,
                )
            else:
                prompt = _build_slot_packed_prompt(user_text, history_turns=history_turns, memories=memories)
            if _count_tokens(tokenizer, prompt) <= prompt_budget:
                _set_last_debug(
                    {
                        "prompt_format": "slot",
                        "user_text": user_text,
                        "query_text": query_text,
                        "mem_items": mem_items,
                        "prompt": prompt,
                        "prompt_tokens": _count_tokens(tokenizer, prompt),
                    }
                )
                return prompt

            compacted = tree.compact_oldest(
                summarizer=summarize_text,
                chunk_size=int(args.compact_chunk_size),
                protect_last_leaves=history_turns,
                title=_summary_title(),
            )
            if compacted:
                tree.save(memory_path)
                continue
            if history_turns > 0:
                history_turns -= 1
                continue
            if mem_max_items is None:
                mem_max_items = max(len(mem_ids) - 1, 0)
                continue
            if mem_max_items > 0:
                mem_max_items -= 1
                continue

            # Last resort: truncate user input.
            user_text = _truncate_to_tokens(tokenizer, user_text, max(prompt_budget // 2, 16))

    def build_fitted_chat_enc(user_text: str) -> tuple[Dict[str, torch.Tensor], int]:
        history_turns = max(int(args.history_turns), 0)
        mem_max_items = max(int(args.mem_max_items), 0) if args.mem_max_items is not None else None

        while True:
            score_map: Dict[str, float] = {}
            query_vec: List[float] = []
            query_text: str = ""
            if args.mem_select == "time":
                mem_ids = tree.iter_memory_ids(protect_last_leaves=history_turns, max_items=mem_max_items)
            else:
                candidates = tree.iter_memory_ids(protect_last_leaves=history_turns, max_items=None)
                query_text = _build_retrieval_query(user_text)
                query_vec = _get_embedding(query_text)
                if not query_vec:
                    mem_ids = tree.iter_memory_ids(protect_last_leaves=history_turns, max_items=mem_max_items)
                else:
                    if args.mem_select == "hybrid":
                        keep = max(int(args.hybrid_summary_keep), 0)
                        summary_ids = [
                            nid
                            for nid in candidates
                            if (tree.nodes.get(nid) is not None and tree.nodes[nid].role == "summary")
                        ]
                        baseline = summary_ids[-keep:] if keep > 0 else []
                        if mem_max_items is not None:
                            baseline = baseline[-min(len(baseline), mem_max_items) :]
                        remaining = max(int(mem_max_items or 0) - len(baseline), 0)
                        topk = int(args.retrieval_topk) if int(args.retrieval_topk or 0) > 0 else remaining
                        topk = min(max(topk, 0), remaining)
                        ranked = _retrieve_ranked_memory_ids_from_query_vec(
                            candidates, query_vec, query_text=query_text, topk=topk
                        )
                        score_map = {nid: float(score) for nid, score in ranked}
                        retrieved = [nid for nid, _ in ranked]
                        mem_ids = list(dict.fromkeys(baseline + retrieved))
                    else:
                        topk = int(args.retrieval_topk) if int(args.retrieval_topk or 0) > 0 else int(mem_max_items or 0)
                        ranked = _retrieve_ranked_memory_ids_from_query_vec(
                            candidates, query_vec, query_text=query_text, topk=topk
                        )
                        score_map = {nid: float(score) for nid, score in ranked}
                        mem_ids = [nid for nid, _ in ranked]
                if not mem_ids:
                    mem_ids = tree.iter_memory_ids(protect_last_leaves=history_turns, max_items=mem_max_items)
                if query_vec and mem_max_items is not None and mem_max_items > 0:
                        mem_ids = _expand_summary_children(
                            mem_ids,
                            query_vec,
                            total_limit=int(mem_max_items),
                            expand_only=set(score_map.keys()) if score_map else None,
                        )

            memory_texts: List[str] = []
            mem_items: List[Dict[str, Any]] = []
            for nid in mem_ids:
                t = tree.render_node_for_prompt(nid)
                if not t:
                    continue
                role = getattr(tree.nodes.get(nid), "role", "")
                title = getattr(tree.nodes.get(nid), "title", "")
                mem_items.append(
                    {
                        "id": nid,
                        "role": role,
                        "title": title,
                        "score": float(score_map[nid]) if nid in score_map else None,
                        "preview": t,
                    }
                )
                if nid in score_map:
                    memory_texts.append(f"(score={score_map[nid]:.3f}) {t}")
                else:
                    memory_texts.append(t)

            messages = _build_chat_messages(user_text, history_turns=history_turns, memory_texts=memory_texts)
            enc, input_len = _encode_chat_messages(
                tokenizer,
                messages,
                chat_template=str(args.chat_template),
                enable_thinking=bool(args.enable_thinking),
            )
            if int(enc["input_ids"].shape[1]) <= int(prompt_budget):
                prompt_text: str = ""
                if hasattr(tokenizer, "apply_chat_template"):
                    try:
                        prompt_text = str(
                            tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=bool(args.enable_thinking),
                            )
                        )
                    except Exception:
                        prompt_text = ""
                _set_last_debug(
                    {
                        "prompt_format": "chat",
                        "user_text": user_text,
                        "query_text": query_text,
                        "mem_items": mem_items,
                        "messages": messages,
                        "prompt_text": prompt_text,
                        "input_tokens": int(enc["input_ids"].shape[1]),
                    }
                )
                return enc, input_len

            compacted = tree.compact_oldest(
                summarizer=summarize_text,
                chunk_size=int(args.compact_chunk_size),
                protect_last_leaves=history_turns,
                title=_summary_title(),
            )
            if compacted:
                tree.save(memory_path)
                continue
            if history_turns > 0:
                history_turns -= 1
                continue
            if mem_max_items is None:
                mem_max_items = max(len(mem_ids) - 1, 0)
                continue
            if mem_max_items > 0:
                mem_max_items -= 1
                continue

            user_text = _truncate_to_tokens(tokenizer, user_text, max(prompt_budget // 2, 16))

    print(
        "进入对话模式。\n"
        "输入 /help 查看命令说明。\n"
        f"prompt_budget={prompt_budget}  context_window={context_window}  summary_mode={args.summary_mode}\n"
        f"mem_select={args.mem_select}  pack_mode={args.pack_mode}  prompt_format={'chat' if use_chat_prompt else 'slot'}  "
        f"slot_tag_lang={args.slot_tag_lang}  mem_hint={'on' if args.mem_structured_hint else 'off'}  "
        f"thinking={'on' if args.enable_thinking else 'off'}  "
        f"index_backend={index_backend_used}"
    )
    while True:
        try:
            user_text = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not user_text:
            continue

        # At the start of each round: commit the previous round into the memory tree.
        commit_pending_if_any()

        if user_text.startswith("/"):
            cmd, *rest = user_text.split(" ", 1)
            arg = rest[0] if rest else ""
            if cmd in {"/exit", "/quit"}:
                break
            if cmd == "/help":
                print(HELP_TEXT)
                continue
            if cmd == "/reset":
                history.clear()
                print("已清空当前会话 [HIST]（记忆树仍保留）。")
                continue
            if cmd == "/debug":
                v = (arg or "").strip().lower()
                if v in {"on", "1", "true", "yes"}:
                    debug_enabled = True
                elif v in {"off", "0", "false", "no"}:
                    debug_enabled = False
                print(f"debug={'on' if debug_enabled else 'off'}  debug_max_chars={debug_max_chars}")
                continue
            if cmd == "/show_prompt":
                _dbg_print_last_prompt()
                continue
            if cmd == "/show_mem":
                _dbg_print_last_mem()
                continue
            if cmd == "/mem_tail":
                n = 10
                try:
                    if arg.strip():
                        n = int(arg.strip())
                except Exception:
                    n = 10
                root_children = list(getattr(tree.nodes.get(tree.root_id), "children", []) or [])
                tail_ids = list(reversed(root_children[-max(n, 0) :])) if n and n > 0 else []
                if not tail_ids:
                    print("(empty)")
                    continue
                for nid in tail_ids:
                    node = tree.nodes.get(nid)
                    if not node:
                        continue
                    preview = (node.text or "").strip().splitlines()[0] if (node.text or "").strip() else ""
                    print(f"{nid} role={node.role} title={node.title} created_at={node.created_at:.0f} {preview}")
                continue
            if cmd == "/mem_show":
                nid = arg.strip()
                node = tree.nodes.get(nid) if nid else None
                if not node:
                    print("用法：/mem_show <node_id>")
                    continue
                print(f"id={node.node_id} role={node.role} title={node.title} created_at={node.created_at:.0f}")
                print(f"parent_id={node.parent_id} children={list(node.children or [])}")
                print(node.text or "")
                continue
            if cmd == "/mem_search":
                kw = arg.strip()
                if not kw:
                    print("用法：/mem_search <keyword>")
                    continue
                hits: List[str] = []
                for nid, node in tree.nodes.items():
                    if nid == tree.root_id:
                        continue
                    hay = f"{node.title}\n{node.text}"
                    if kw in hay:
                        hits.append(nid)
                if not hits:
                    print("(no matches)")
                    continue
                for nid in hits[-50:]:
                    node = tree.nodes.get(nid)
                    if not node:
                        continue
                    print(f"{nid} role={node.role} title={node.title}")
                continue
            if cmd == "/mem_clear":
                tree.reset()
                tree.save(memory_path)
                vectors.clear()
                last_debug.clear()
                for p in (
                    vector_cache_path,
                    index_path,
                    index_path.with_suffix(index_path.suffix + ".meta.json"),
                ):
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
                    except Exception:
                        pass
                if index_backend_used == "hnsw":
                    try:
                        vec_index = HNSWIndex(
                            HNSWConfig(
                                dim=embed_dim,
                                space="cosine",
                                persist_path=str(index_path),
                                model=model_fingerprint,
                            )
                        )
                    except Exception:
                        vec_index = VectorIndex(dim=embed_dim)
                        index_backend_used = "flat"
                else:
                    vec_index = VectorIndex(dim=embed_dim)
                print("已清空记忆树与检索索引。")
                continue
            if cmd == "/mem_stats":
                print(f"memory_path={memory_path}")
                print(f"nodes={tree.num_nodes()}  root_children={tree.num_root_children()}")
                print(f"vector_cache_path={vector_cache_path}  vectors_cached={len(vectors)}")
                print(f"index_backend={index_backend_used}  index_path={index_path}")
                continue
            if cmd == "/persona":
                args.persona = arg.strip()
                print("已更新 persona。")
                continue
            if cmd == "/instruction":
                args.instruction = arg.strip()
                print("已更新 instruction。")
                continue
            print(f"未知命令：{cmd}（用 /help 查看）")
            continue

        if use_chat_prompt:
            enc, input_len = build_fitted_chat_enc(user_text)
            reply = _generate_from_enc(
                model,
                tokenizer,
                device,
                enc,
                input_len,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        else:
            prompt = build_fitted_prompt(user_text)
            reply = _generate(
                model,
                tokenizer,
                device,
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        if not args.enable_thinking:
            reply = _strip_think(reply)
        if not args.keep_tokenizer_spaces:
            reply = _collapse_cjk_spaces(reply)
        print(f"Assistant> {reply}")
        history.append((user_text, reply))
        pending_commit = (user_text, reply)

    # Ensure we don't lose the last turn if the user exits immediately after a reply.
    commit_pending_if_any()
    tree.save(memory_path)


if __name__ == "__main__":
    main()
