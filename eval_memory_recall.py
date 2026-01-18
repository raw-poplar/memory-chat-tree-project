from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root on sys.path so this script can be run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chat_tree import (  # noqa: E402
    _encode_chat_messages,
    _generate_from_enc,
    _infer_context_window,
    _l2_normalize,
    _strip_think,
    _truncate_to_tokens,
    _truncate_to_tokens_tail,
)
from memory_tree import MemoryTree  # noqa: E402


@dataclass(frozen=True)
class EvalItem:
    idx: int
    name: str
    question: str
    expected: str
    leaf_id: str


@dataclass(frozen=True)
class SemanticSeed:
    name: str
    memory_user: str
    question: str
    expected: str


def _semantic_seeds() -> List[SemanticSeed]:
    # Intentionally avoid "ID-style" keys like K0_xxx; these are meant to be closer to semantic recall.
    return [
        SemanticSeed(
            name="city",
            memory_user="我现在常住在杭州。",
            question="我住在哪个城市？请只输出城市名。",
            expected="杭州",
        ),
        SemanticSeed(
            name="birthday",
            memory_user="我的生日是 7 月 19 日。",
            question="我的生日是哪一天？请只输出日期。",
            expected="7 月 19 日",
        ),
        SemanticSeed(
            name="theme",
            memory_user="我更喜欢深色模式，亮色太刺眼。",
            question="我更偏好什么界面主题？请只输出答案。",
            expected="深色模式",
        ),
        SemanticSeed(
            name="coffee",
            memory_user="我最常点的咖啡是无糖冰美式。",
            question="我平时最爱点什么咖啡？请只输出答案。",
            expected="无糖冰美式",
        ),
        SemanticSeed(
            name="allergy",
            memory_user="我对花生严重过敏。",
            question="我有什么食物过敏？请只输出过敏源。",
            expected="花生",
        ),
        SemanticSeed(
            name="no_cilantro",
            memory_user="我不吃香菜。",
            question="我有什么忌口？请只输出忌口食材。",
            expected="香菜",
        ),
        SemanticSeed(
            name="pet_name",
            memory_user="我养了一只猫，名字叫豆豆。",
            question="我家猫叫什么名字？请只输出名字。",
            expected="豆豆",
        ),
        SemanticSeed(
            name="sport",
            memory_user="我最喜欢的运动是羽毛球。",
            question="我最喜欢的运动是什么？请只输出答案。",
            expected="羽毛球",
        ),
        SemanticSeed(
            name="color",
            memory_user="我最喜欢的颜色是墨绿色。",
            question="我最喜欢什么颜色？请只输出颜色。",
            expected="墨绿色",
        ),
        SemanticSeed(
            name="music",
            memory_user="我经常听 lo-fi hip hop。",
            question="我常听什么类型的音乐？请只输出类型。",
            expected="lo-fi hip hop",
        ),
        SemanticSeed(
            name="movie",
            memory_user="我最喜欢的电影是《星际穿越》。",
            question="我最喜欢的电影是哪部？请只输出片名。",
            expected="《星际穿越》",
        ),
        SemanticSeed(
            name="fruit",
            memory_user="我最爱吃的水果是芒果。",
            question="我最喜欢吃什么水果？请只输出水果名。",
            expected="芒果",
        ),
        SemanticSeed(
            name="os",
            memory_user="我电脑的主力操作系统是 Windows 11。",
            question="我电脑主要用什么操作系统？请只输出系统名。",
            expected="Windows 11",
        ),
        SemanticSeed(
            name="language",
            memory_user="我最常写的编程语言是 Python。",
            question="我主要用什么编程语言？请只输出语言名。",
            expected="Python",
        ),
        SemanticSeed(
            name="email",
            memory_user="我最常用的邮箱是 yang.sheep@example.com。",
            question="我常用的邮箱地址是什么？请只输出邮箱。",
            expected="yang.sheep@example.com",
        ),
        SemanticSeed(
            name="commute",
            memory_user="我通勤通常坐地铁 2 号线。",
            question="我平时怎么通勤？请只输出答案。",
            expected="地铁 2 号线",
        ),
        SemanticSeed(
            name="sleep",
            memory_user="我一般晚上 11 点半以后才睡。",
            question="我通常几点睡觉？请只输出时间。",
            expected="11 点半以后",
        ),
        SemanticSeed(
            name="brightness",
            memory_user="我看书时会把屏幕亮度调到 30% 左右。",
            question="我阅读时习惯把屏幕亮度调到多少？请只输出数值。",
            expected="30%",
        ),
        SemanticSeed(
            name="meeting_av",
            memory_user="开线上会议时我会打开摄像头，但麦克风默认静音。",
            question="我开线上会议时通常怎么设置音视频？请只输出答案。",
            expected="打开摄像头",
        ),
        SemanticSeed(
            name="task_focus",
            memory_user="我当前的优先事项是把记忆树项目先跑通再谈优化。",
            question="我当前最优先要完成的事情是什么？请只输出答案。",
            expected="把记忆树项目先跑通",
        ),
    ]


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


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_torch_dtype(device: torch.device, dtype: str) -> Optional[torch.dtype]:
    dtype_norm = (dtype or "auto").strip().lower()
    if dtype_norm == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    if dtype_norm == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if dtype_norm == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if dtype_norm == "fp32":
        return torch.float32
    return None


def _load_model(
    model_dir: Path,
    device: torch.device,
    dtype: Optional[torch.dtype],
) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
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
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs.pop("dtype", None)
            model_kwargs["torch_dtype"] = dtype
            model = AutoModelForCausalLM.from_pretrained(str(model_dir), **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(str(model_dir), **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _infer_embedding_dim(model: AutoModelForCausalLM, fallback: int = 768) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return int(fallback)


def _get_embedding(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    text: str,
    *,
    embed_max_tokens: int,
) -> List[float]:
    text = (text or "").strip()
    if not text:
        return []
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max(int(embed_max_tokens), 8),
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


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum((ai * bi) for ai, bi in zip(a, b)))


def _rand_hex(n: int) -> str:
    n = max(int(n), 0)
    if n <= 0:
        return ""
    alphabet = "0123456789ABCDEF"
    return "".join(random.choice(alphabet) for _ in range(n))


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
    return sorted([k for k in raw if any(c.isalpha() for c in k) and (k.lower() not in stop)])


def _token_id_set(tokenizer: AutoTokenizer, text: str, *, max_tokens: int) -> set[int]:
    text = (text or "").strip()
    if not text or max_tokens <= 0:
        return set()
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)[: max_tokens]
        return set(int(x) for x in token_ids if isinstance(x, int))
    except Exception:
        return set()


def _build_retrieval_query(
    tokenizer: AutoTokenizer,
    history_pairs: List[Tuple[str, str]],
    user_text: str,
    *,
    retrieval_query_turns: int,
    retrieval_query_tokens: int,
) -> str:
    turns = max(int(retrieval_query_turns), 0)
    parts: List[str] = []
    recent = history_pairs[-turns:] if turns > 0 else []
    for u, a in recent:
        u = (u or "").strip()
        a = (a or "").strip()
        if u:
            parts.append(f"User: {u}")
        if a:
            parts.append(f"Assistant: {a}")
    parts.append(f"User: {(user_text or '').strip()}")
    combined = "\n".join(parts).strip()
    return _truncate_to_tokens_tail(tokenizer, combined, int(retrieval_query_tokens))


def _make_filler(turn_idx: int, *, tokens: int) -> Tuple[str, str]:
    # Use low-entropy filler so retrieval queries are not dominated by random tokens.
    words = ["无关内容"] * max(int(tokens), 8)
    user = f"无关内容（{turn_idx}）： " + " ".join(words)
    assistant = "好的。"
    return user, assistant


def _approx_tokens(tokenizer: AutoTokenizer, pairs: List[Tuple[str, str]]) -> int:
    total = 0
    for u, a in pairs:
        if u:
            total += len(tokenizer.encode(u, add_special_tokens=False))
        if a:
            total += len(tokenizer.encode(a, add_special_tokens=False))
    return int(total)


def _build_trimmed_messages(
    tokenizer: AutoTokenizer,
    *,
    system_text: str,
    history_pairs: List[Tuple[str, str]],
    user_text: str,
    max_history_turns: int,
    prompt_budget: int,
    chat_template: str,
) -> Tuple[Dict[str, torch.Tensor], int, int]:
    turns = max(int(max_history_turns), 0)
    pairs = history_pairs[-turns:] if turns > 0 else []

    def _messages_from_pairs(p: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_text.strip():
            messages.append({"role": "system", "content": system_text.strip()})
        for u, a in p:
            u = (u or "").strip()
            a = (a or "").strip()
            if u:
                messages.append({"role": "user", "content": u})
            if a:
                messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": (user_text or "").strip()})
        return messages

    # Drop earliest turns until fit.
    while True:
        messages = _messages_from_pairs(pairs)
        enc, input_len = _encode_chat_messages(tokenizer, messages, chat_template=chat_template, enable_thinking=False)
        input_tokens = int(enc["input_ids"].shape[1])
        if input_tokens <= int(prompt_budget):
            return enc, input_len, input_tokens
        if pairs:
            pairs = pairs[1:]
            continue
        # Last resort: truncate user input.
        shortened = _truncate_to_tokens(tokenizer, user_text, max(max(prompt_budget // 2, 16), 16))
        user_text = shortened


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory retrieval recall beyond context window.")
    parser.add_argument(
        "--task",
        choices=["kv", "semantic"],
        default="kv",
        help="Evaluation task: kv=ID-style key/value; semantic=paraphrased user facts without ID keys.",
    )
    parser.add_argument(
        "--model_dir",
        default=str(PROJECT_ROOT / "model"),
        help="HF save_pretrained dir (default: ./model).",
    )
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda or specific torch device string")
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Model dtype (auto=fp16/bf16 on CUDA, fp32 on CPU).",
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=10240,
        help="Simulated context window tokens for the evaluation (default: 10240).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate per query.")
    parser.add_argument("--history_turns", type=int, default=8, help="History turns to include in prompt.")
    parser.add_argument(
        "--retrieval_query_turns",
        type=int,
        default=0,
        help="History turns used in retrieval query (0=only current question).",
    )
    parser.add_argument("--retrieval_query_tokens", type=int, default=96, help="Max tokens of retrieval query text.")
    parser.add_argument("--topk", type=int, default=5, help="Vector retrieval top-k.")
    parser.add_argument("--min_score", type=float, default=0.0, help="Drop retrieved memories below this cosine score.")
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
    parser.add_argument("--num_facts", type=int, default=20, help="How many facts to write and query.")
    parser.add_argument("--code_len", type=int, default=16, help="Random code length (hex).")
    parser.add_argument("--filler_turns", type=int, default=80, help="How many filler turns after facts.")
    parser.add_argument("--filler_tokens", type=int, default=120, help="Approx tokens per filler user turn.")
    parser.add_argument(
        "--filler_tokens_tail",
        type=int,
        default=24,
        help="Use smaller filler for the last few turns (kept in prompt/query) to reduce GPU memory use.",
    )
    parser.add_argument("--embed_max_tokens", type=int, default=64, help="Max tokens when extracting embeddings.")
    parser.add_argument(
        "--mem_preview_tokens",
        type=int,
        default=96,
        help="Max tokens per injected memory line (to avoid oversized prompts).",
    )
    parser.add_argument(
        "--eval_generation",
        action="store_true",
        help="Also run generation to test whether answers become correct with memory injection.",
    )
    parser.add_argument("--eval_generation_n", type=int, default=10, help="How many queries to run generation on.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--debug_queries", type=int, default=0, help="Print top retrieval results for first N queries.")
    parser.add_argument(
        "--chat_template",
        choices=["auto", "on", "off"],
        default="auto",
        help="Tokenizer chat template mode for prompt encoding.",
    )
    args = parser.parse_args()

    random.seed(int(args.seed))

    model_dir = _resolve_model_dir_arg(str(args.model_dir))
    torch_device = _resolve_device(str(args.device))
    torch_dtype = _resolve_torch_dtype(torch_device, str(args.dtype))
    tokenizer, model, device = _load_model(model_dir, torch_device, torch_dtype)

    inferred_window = _infer_context_window(tokenizer, model, fallback=2048)
    context_window = int(args.context_window) if int(args.context_window or 0) > 0 else inferred_window
    prompt_budget = max(context_window - int(args.max_new_tokens), 64)

    embed_dim = _infer_embedding_dim(model, fallback=768)
    dtype_name = str(torch_dtype).replace("torch.", "") if torch_dtype is not None else "auto"
    print(
        f"[info] model_dir={model_dir}  device={device}  dtype={dtype_name}  embed_dim={embed_dim}\n"
        f"[info] eval_context_window={context_window}  prompt_budget={prompt_budget}  max_new_tokens={int(args.max_new_tokens)}"
    )

    # Build a long conversation: facts first, then enough filler to exceed context.
    tree = MemoryTree()
    history_pairs: List[Tuple[str, str]] = []
    items: List[EvalItem] = []

    assistant_ack = "好的，我记住了。"
    task = str(args.task or "kv")
    if task == "semantic":
        seeds = _semantic_seeds()
        if not seeds:
            print("[error] no semantic seeds available.")
            raise SystemExit(1)
        random.shuffle(seeds)
        n = max(int(args.num_facts), 0)
        selected = seeds[: min(n, len(seeds))]
        for i, s in enumerate(selected):
            user = s.memory_user
            leaf_id = tree.add_leaf(user, assistant_ack, title=f"sem_{s.name}")
            history_pairs.append((user, assistant_ack))
            items.append(EvalItem(idx=i, name=s.name, question=s.question, expected=s.expected, leaf_id=leaf_id))
    else:
        for i in range(int(args.num_facts)):
            code = _rand_hex(max(int(args.code_len), 8))
            key = f"K{i}_{_rand_hex(8)}"
            user = f"请记住：键 {key} 的值是 {code}。"
            leaf_id = tree.add_leaf(user, assistant_ack, title=f"kv_{key}")
            history_pairs.append((user, assistant_ack))
            question = f"键 {key} 的值是什么？请只输出值本身（不要输出多余解释）。"
            items.append(EvalItem(idx=i, name=key, question=question, expected=code, leaf_id=leaf_id))

    for j in range(int(args.filler_turns)):
        tail_turns = max(int(args.history_turns), int(args.retrieval_query_turns), 1)
        tok = int(args.filler_tokens_tail) if j >= int(args.filler_turns) - tail_turns else int(args.filler_tokens)
        u, a = _make_filler(j, tokens=tok)
        tree.add_leaf(u, a, title=f"filler_{j}")
        history_pairs.append((u, a))

    approx_hist_tokens = _approx_tokens(tokenizer, history_pairs)
    print(f"[info] built turns={len(history_pairs)}  approx_history_tokens={approx_hist_tokens}")
    if approx_hist_tokens <= int(context_window):
        print("[warn] history may not exceed the configured context_window; increase --filler_turns/--filler_tokens.")

    # Precompute vectors for all candidate memory ids (excluding protected recent leaves).
    candidates = tree.iter_memory_ids(protect_last_leaves=int(args.history_turns), max_items=None)
    vectors: Dict[str, List[float]] = {}
    skipped = 0
    for nid in candidates:
        text = tree.render_node_for_prompt(nid) or ""
        vec = _get_embedding(tokenizer, model, device, text, embed_max_tokens=int(args.embed_max_tokens))
        if not vec or (embed_dim and len(vec) != embed_dim):
            skipped += 1
            continue
        vectors[nid] = vec
    print(f"[info] indexed candidates={len(vectors)}  skipped={skipped}  protect_last_leaves={int(args.history_turns)}")

    # Evaluate retrieval recall.
    total = 0
    hit_any = 0
    hit_top1 = 0
    mrr_sum = 0.0

    gen_total = 0
    gen_hit_baseline = 0
    gen_hit_with_mem = 0

    for item in items:
        question = item.question
        query_text = _build_retrieval_query(
            tokenizer,
            history_pairs,
            question,
            retrieval_query_turns=int(args.retrieval_query_turns),
            retrieval_query_tokens=int(args.retrieval_query_tokens),
        )
        query_vec = _get_embedding(tokenizer, model, device, query_text, embed_max_tokens=int(args.embed_max_tokens))
        if not query_vec:
            continue

        scored: List[Tuple[str, float]] = []
        for nid in candidates:
            v = vectors.get(nid)
            if not v:
                continue
            score = _dot(query_vec, v)
            if float(score) >= float(args.min_score):
                scored.append((nid, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        topk = max(int(args.topk), 0)
        oversample = min(max(topk * 8, topk), 256)
        candidates_ranked = scored[:oversample]

        rerank_mode = str(args.retrieval_rerank or "off")
        rerank_weight = float(args.retrieval_rerank_weight)
        rerank_doc_tokens = int(args.retrieval_rerank_doc_tokens)
        if rerank_mode != "off" and rerank_weight > 0 and candidates_ranked:
            if rerank_mode == "keyword":
                kws = _extract_keywords(query_text)
                if kws:
                    tmp: List[Tuple[str, float, float]] = []
                    for nid, base in candidates_ranked:
                        doc = tree.render_node_for_prompt(nid) or ""
                        hits = sum((1 for k in kws if k in doc))
                        bonus = rerank_weight * (float(hits) / float(len(kws)))
                        tmp.append((nid, base, base + bonus))
                    tmp.sort(key=lambda x: x[2], reverse=True)
                    candidates_ranked = [(nid, base) for nid, base, _ in tmp]
            elif rerank_mode == "token_overlap":
                qset = _token_id_set(tokenizer, query_text, max_tokens=max(int(args.retrieval_query_tokens), 8))
                if qset:
                    doc_max = max(rerank_doc_tokens, 8)
                    tmp2: List[Tuple[str, float, float]] = []
                    for nid, base in candidates_ranked:
                        doc = tree.render_node_for_prompt(nid) or ""
                        dset = _token_id_set(tokenizer, doc, max_tokens=doc_max)
                        overlap = 0.0
                        if dset:
                            overlap = len(qset & dset) / float(max(len(qset), 1))
                        tmp2.append((nid, base, base + rerank_weight * float(overlap)))
                    tmp2.sort(key=lambda x: x[2], reverse=True)
                    candidates_ranked = [(nid, base) for nid, base, _ in tmp2]

        ranked = candidates_ranked[:topk]
        retrieved_ids = [nid for nid, _ in ranked]

        if int(args.debug_queries) > 0 and total < int(args.debug_queries):
            print(f"\n[debug] q{item.idx} name={item.name} expected={item.expected}")
            print(f"[debug] query_text:\n{query_text}\n")
            print("[debug] top results:")
            for nid, score in ranked[:10]:
                node = tree.nodes.get(nid)
                title = getattr(node, "title", "") if node else ""
                role = getattr(node, "role", "") if node else ""
                preview = tree.render_node_for_prompt(nid) or ""
                preview = _truncate_to_tokens(tokenizer, preview, 80)
                mark = " <==" if nid == item.leaf_id else ""
                print(f"[debug] - score={score:.3f} id={nid} role={role} title={title}{mark}\n  {preview}")

        total += 1
        if retrieved_ids and retrieved_ids[0] == item.leaf_id:
            hit_top1 += 1
        if item.leaf_id in retrieved_ids:
            hit_any += 1
            rank = retrieved_ids.index(item.leaf_id) + 1
            mrr_sum += 1.0 / float(rank)

        if args.eval_generation and gen_total < int(args.eval_generation_n):
            system_base = (
                "你在回答时必须遵守：\n"
                "- 如果输入里没有足够信息，就回答“未知”。\n"
                "- 按用户要求，只输出标签内容本身，不要输出多余解释。\n"
            )

            # Baseline: no memory injection.
            enc0, input_len0, in_tok0 = _build_trimmed_messages(
                tokenizer,
                system_text=system_base,
                history_pairs=history_pairs,
                user_text=question,
                max_history_turns=int(args.history_turns),
                prompt_budget=prompt_budget,
                chat_template=str(args.chat_template),
            )
            reply0 = _generate_from_enc(
                model,
                tokenizer,
                device,
                enc0,
                input_len0,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            )
            reply0 = _strip_think(reply0)
            if item.expected in (reply0 or ""):
                gen_hit_baseline += 1

            # With memory injection (top-k texts).
            mem_lines: List[str] = []
            for nid, score in ranked:
                t = tree.render_node_for_prompt(nid) or ""
                if not t:
                    continue
                t = _truncate_to_tokens(tokenizer, t, int(args.mem_preview_tokens))
                mem_lines.append(f"(score={score:.3f}) {t}")
            mem_block = ""
            if mem_lines:
                mem_block = "以下是检索到的相关记忆（可能有噪声，仅供参考）：\n" + "\n".join(f"- {t}" for t in mem_lines)
            system_with_mem = (mem_block + "\n\n" + system_base).strip()

            enc1, input_len1, in_tok1 = _build_trimmed_messages(
                tokenizer,
                system_text=system_with_mem,
                history_pairs=history_pairs,
                user_text=question,
                max_history_turns=int(args.history_turns),
                prompt_budget=prompt_budget,
                chat_template=str(args.chat_template),
            )
            reply1 = _generate_from_enc(
                model,
                tokenizer,
                device,
                enc1,
                input_len1,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
            )
            reply1 = _strip_think(reply1)
            if item.expected in (reply1 or ""):
                gen_hit_with_mem += 1

            gen_total += 1
            if gen_total <= 3:
                print(
                    f"\n[sample] q{item.idx} expected={item.expected}\n"
                    f"[sample] baseline(input_tokens={in_tok0}) => {reply0!r}\n"
                    f"[sample] with_mem(input_tokens={in_tok1}) => {reply1!r}"
                )

    if total <= 0:
        print("[error] no queries evaluated (check model/embedding).")
        raise SystemExit(1)

    recall_at_k = hit_any / float(total)
    top1 = hit_top1 / float(total)
    mrr = mrr_sum / float(total)
    print(
        "\n[retrieval]\n"
        f"- queries={total}\n"
        f"- topk={int(args.topk)}  min_score={float(args.min_score)}\n"
        f"- recall@{int(args.topk)}={recall_at_k:.3f}  top1={top1:.3f}  mrr={mrr:.3f}"
    )

    if args.eval_generation:
        print(
            "\n[generation]\n"
            f"- evaluated={gen_total}\n"
            f"- baseline_acc={gen_hit_baseline / float(max(gen_total, 1)):.3f} ({gen_hit_baseline}/{gen_total})\n"
            f"- with_memory_acc={gen_hit_with_mem / float(max(gen_total, 1)):.3f} ({gen_hit_with_mem}/{gen_total})"
        )


if __name__ == "__main__":
    main()
