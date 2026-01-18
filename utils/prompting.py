"""Canonical prompt builder used across training and inference.

This module is the single authoritative entry for assembling prompts that follow
the project's slot-tag protocol (default English tags):
  [SYS] ... [/SYS]
  [MEM:*] ... [/MEM]
  [HIST] ... [/HIST]
  [USER] ... [/USER]

For better readability on Chinese-first workflows, you can switch to Chinese
tags (e.g. [系统]/[记忆]/[历史]/[用户]) via the `PromptProtocol` parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class MemoryBlock:
    slot: str
    text: str


@dataclass(frozen=True)
class PromptProtocol:
    sys_open: str
    sys_close: str
    user_open: str
    user_close: str
    hist_open: str
    hist_close: str
    mem_open_prefix: str
    mem_close: str
    speaker_user_prefix: str = "User:"
    speaker_assistant_prefix: str = "Assistant:"
    mem_slot_map: Mapping[str, str] = field(default_factory=dict)

    def mem_open(self, slot: str) -> str:
        slot_norm = normalize_mem_slot(slot)
        label = self.mem_slot_map.get(slot_norm, slot_norm)
        return f"{self.mem_open_prefix}{label}]"


PROTOCOL_EN = PromptProtocol(
    sys_open="[SYS]",
    sys_close="[/SYS]",
    user_open="[USER]",
    user_close="[/USER]",
    hist_open="[HIST]",
    hist_close="[/HIST]",
    mem_open_prefix="[MEM:",
    mem_close="[/MEM]",
    speaker_user_prefix="User:",
    speaker_assistant_prefix="Assistant:",
    mem_slot_map={"train": "train", "session": "session", "longterm": "longterm", "WRITE": "WRITE"},
)

PROTOCOL_ZH = PromptProtocol(
    sys_open="[系统]",
    sys_close="[/系统]",
    user_open="[用户]",
    user_close="[/用户]",
    hist_open="[历史]",
    hist_close="[/历史]",
    mem_open_prefix="[记忆:",
    mem_close="[/记忆]",
    speaker_user_prefix="用户:",
    speaker_assistant_prefix="助手:",
    mem_slot_map={"train": "训练", "session": "会话", "longterm": "长期", "WRITE": "写入"},
)


def get_protocol(lang: str) -> PromptProtocol:
    lang_norm = (lang or "").strip().lower()
    if lang_norm in {"zh", "cn", "zh-cn", "zh_cn", "chinese"}:
        return PROTOCOL_ZH
    return PROTOCOL_EN


def protocol_special_tokens(protocol: PromptProtocol) -> List[str]:
    """Slot-tag markers that should be registered as special tokens when needed."""
    tokens: List[str] = [
        protocol.sys_open,
        protocol.sys_close,
        protocol.user_open,
        protocol.user_close,
        protocol.hist_open,
        protocol.hist_close,
        protocol.mem_close,
    ]
    for slot in ("train", "session", "longterm", "WRITE"):
        tokens.append(protocol.mem_open(slot))
    # preserve order, de-dup
    return list(dict.fromkeys(tokens))


def normalize_mem_slot(slot: str) -> str:
    slot = (slot or "").strip()
    if not slot:
        return "session"
    slot_lower = slot.lower()
    if slot_lower in {"train", "session", "longterm"}:
        return slot_lower
    if slot_lower in {"write", "mem:write", "mem_write"}:
        return "WRITE"
    return slot


def _join_nonempty(parts: Sequence[str]) -> str:
    cleaned = [p.strip() for p in parts if p and str(p).strip()]
    return " ".join(cleaned).strip()


def format_sys_block(
    persona: Union[str, Sequence[str], None],
    instruction: Optional[str] = None,
    *,
    protocol: PromptProtocol = PROTOCOL_EN,
) -> Optional[str]:
    if persona is None:
        persona_text = ""
    elif isinstance(persona, str):
        persona_text = persona.strip()
    else:
        persona_text = _join_nonempty([str(x) for x in persona])
    instruction_text = (instruction or "").strip()
    content = _join_nonempty([persona_text, instruction_text])
    if not content:
        return None
    return f"{protocol.sys_open} {content} {protocol.sys_close}"


def format_memory_blocks(memories: Sequence[MemoryBlock], *, protocol: PromptProtocol = PROTOCOL_EN) -> List[str]:
    lines: List[str] = []
    for mem in memories or []:
        text = (mem.text or "").strip()
        if not text:
            continue
        mem_open = protocol.mem_open(mem.slot)
        lines.append(f"{mem_open} {text} {protocol.mem_close}")
    return lines


def format_history_block(history_messages: Sequence[Tuple[str, str]], *, protocol: PromptProtocol = PROTOCOL_EN) -> List[str]:
    if not history_messages:
        return []
    lines: List[str] = [protocol.hist_open]
    for speaker, text in history_messages:
        t = (text or "").strip()
        if not t:
            continue
        speaker_norm = (speaker or "").strip().lower()
        prefix = protocol.speaker_user_prefix if speaker_norm == "user" else protocol.speaker_assistant_prefix
        lines.append(f"{prefix} {t}")
    lines.append(protocol.hist_close)
    return lines


def format_user_block(user_text: str, *, protocol: PromptProtocol = PROTOCOL_EN) -> Optional[str]:
    text = (user_text or "").strip()
    if not text:
        return None
    return f"{protocol.user_open} {text} {protocol.user_close}"


def _pairs_to_messages(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    messages: List[Tuple[str, str]] = []
    for user_text, assistant_text in pairs:
        if user_text and str(user_text).strip():
            messages.append(("user", str(user_text)))
        if assistant_text and str(assistant_text).strip():
            messages.append(("assistant", str(assistant_text)))
    return messages


def build_prompt(
    user_text: str,
    *,
    persona: Union[str, Sequence[str], None] = None,
    instruction: Optional[str] = None,
    memories: Optional[Sequence[MemoryBlock]] = None,
    history_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    history_messages: Optional[Sequence[Tuple[str, str]]] = None,
    max_history_turns: int = 0,
    protocol: PromptProtocol = PROTOCOL_EN,
) -> str:
    """Build a slot-tagged prompt string.

    Canonical order:
      [SYS] -> [MEM:*] -> [HIST] -> [USER]
    """
    if history_pairs is not None and history_messages is not None:
        raise ValueError("Provide only one of history_pairs or history_messages.")

    lines: List[str] = []
    sys_line = format_sys_block(persona, instruction, protocol=protocol)
    if sys_line:
        lines.append(sys_line)

    lines.extend(format_memory_blocks(list(memories or []), protocol=protocol))

    if history_pairs is not None:
        pairs = list(history_pairs)
        if max_history_turns and max_history_turns > 0:
            pairs = pairs[-max_history_turns:]
        history_messages = _pairs_to_messages(pairs)

    lines.extend(format_history_block(list(history_messages or []), protocol=protocol))

    user_line = format_user_block(user_text, protocol=protocol)
    if user_line:
        lines.append(user_line)
    return "\n".join(lines).strip()
