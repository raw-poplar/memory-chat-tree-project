from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set.*",
    category=FutureWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model"

HELP_TEXT = """命令：
- /help：显示本帮助
- /exit, /quit：退出
- /reset：清空当前会话历史（不影响模型与参数）
- /system <text>：设置 system prompt（下一轮生效；对支持 chat template 的模型更有用）
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
    for attr in ("n_positions", "max_position_embeddings", "n_ctx"):
        v = getattr(getattr(model, "config", None), attr, None)
        if isinstance(v, int) and 8 <= v <= 100000:
            candidates.append(v)
    tmax = getattr(tokenizer, "model_max_length", None)
    if isinstance(tmax, int) and 8 <= tmax <= 100000:
        candidates.append(tmax)
    if candidates:
        return min(candidates)
    return int(fallback)


def _collapse_cjk_spaces(text: str) -> str:
    if not text:
        return text
    import re

    return re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)


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

    model_kwargs: Dict[str, Any] = {}
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


def _build_messages(
    history_pairs: List[Tuple[str, str]],
    user_text: str,
    *,
    system: str,
    max_history_turns: int,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    pairs = history_pairs[-max_history_turns:] if max_history_turns > 0 else []
    for u, a in pairs:
        u = (u or "").strip()
        a = (a or "").strip()
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": (user_text or "").strip()})
    return messages


def _render_plain_prompt(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            lines.append(content)
            continue
        if role == "assistant":
            lines.append(f"Assistant: {content}")
            continue
        lines.append(f"User: {content}")
    lines.append("Assistant:")
    return "\n".join(lines).strip()


def _encode_prompt(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    *,
    chat_template: str,
    enable_thinking: bool,
) -> Tuple[Dict[str, torch.Tensor], int]:
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

    prompt = _render_plain_prompt(messages)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    return enc, int(enc["input_ids"].shape[1])


def _generate(
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

    gen_kwargs: Dict[str, Any] = {
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

    new_tokens = out[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if text:
        return text
    return tokenizer.decode(new_tokens, skip_special_tokens=False).strip()


def main() -> None:
    _try_reconfigure_stdio_utf8()
    parser = argparse.ArgumentParser(description="Plain chat (no memory), for quick model sanity checks.")
    parser.add_argument(
        "--model_dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to a HF save_pretrained model dir (default: ./model; if ./model has a single child dir with config.json, use that).",
    )
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda or specific torch device string")
    parser.add_argument("--system", default="", help="Optional system prompt (for chat-template models).")
    parser.add_argument(
        "--chat_template",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use tokenizer chat template if available (auto), force on, or disable (off).",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="When using chat templates that support it (e.g. Qwen3), allow <think> reasoning output (default: off).",
    )
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
    parser.add_argument("--history_turns", type=int, default=8, help="How many (user,assistant) turns to include")
    parser.add_argument(
        "--keep_tokenizer_spaces",
        action="store_true",
        help="Do not collapse spaces between Chinese characters in output",
    )
    args = parser.parse_args()

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

    print(
        "进入纯对话模式（不含记忆）。\n"
        "输入 /help 查看命令说明。\n"
        f"prompt_budget={prompt_budget}  context_window={context_window}  chat_template={args.chat_template}"
    )

    history: List[Tuple[str, str]] = []
    system_prompt = str(args.system or "").strip()

    while True:
        try:
            user_text = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not user_text:
            continue

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
                print("已清空当前会话历史。")
                continue
            if cmd == "/system":
                system_prompt = arg.strip()
                print("已更新 system prompt。")
                continue
            print(f"未知命令：{cmd}（用 /help 查看）")
            continue

        messages = _build_messages(history, user_text, system=system_prompt, max_history_turns=max(int(args.history_turns), 0))
        enc, input_len = _encode_prompt(
            tokenizer, messages, chat_template=str(args.chat_template), enable_thinking=bool(args.enable_thinking)
        )

        # Guard: avoid accidental huge prompts. This is a best-effort heuristic.
        if int(enc["input_ids"].shape[1]) > int(prompt_budget):
            # Keep only the most recent turns if overflow.
            trimmed_turns = max(min(int(args.history_turns), 2), 0)
            messages = _build_messages(history, user_text, system=system_prompt, max_history_turns=trimmed_turns)
            enc, input_len = _encode_prompt(
                tokenizer, messages, chat_template=str(args.chat_template), enable_thinking=bool(args.enable_thinking)
            )

        reply = _generate(
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
        if not args.enable_thinking:
            reply = _strip_think(reply)
        if not args.keep_tokenizer_spaces:
            reply = _collapse_cjk_spaces(reply)
        print(f"Assistant> {reply}")
        history.append((user_text, reply))


if __name__ == "__main__":
    main()
