from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from loguru import logger

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    import torch
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TextIteratorStreamer = None  # type: ignore
    torch = None  # type: ignore

from .config import AppConfig


@dataclass(slots=True)
class LLMBackend:
    name: str
    tokenizer: object | None = None
    model: object | None = None
    ready: bool = False


def load_llm_backend(cfg: AppConfig) -> LLMBackend:
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        logger.warning("Transformers not available; using heuristic LLM fallback")
        return LLMBackend(name="heuristic-fallback", ready=False)

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_name, token=cfg.hf_token, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_model_name,
            token=cfg.hf_token,
            torch_dtype=getattr(torch, "float16", None) if torch is not None else None,
            device_map="auto",
        )
        logger.info("Loaded LLM model {}", cfg.llm_model_name)
        return LLMBackend(name=cfg.llm_model_name, tokenizer=tokenizer, model=model, ready=True)
    except Exception as exc:
        logger.warning("Could not load LLM model {}: {}", cfg.llm_model_name, exc)
        return LLMBackend(name="heuristic-fallback", ready=False)


def build_prompt(context_pack: dict, cfg: AppConfig) -> str:
    legacy = context_pack.get("legacy") or {}
    modern = context_pack.get("modern") or {}
    legacy_text = legacy.get("text") or ""
    modern_text = modern.get("text") or ""
    stats = context_pack.get("stats") or {}
    neighbors = context_pack.get("legacy_neighbors", []) + context_pack.get("modern_neighbors", [])

    neighbor_text = "\n".join(
        f"- {n.get('page')}: {n.get('text', '')[:500]}" for n in neighbors[:6]
    )

    return f"""You are a policy compliance assistant.
Explain the semantic difference between the legacy and modern policy snippets.

Legacy snippet:
{legacy_text}

Modern snippet:
{modern_text}

Similarity stats:
- cosine: {stats.get('cosine', 0.0):.3f}
- lexical: {stats.get('lexical', 0.0):.3f}
- heading bonus: {stats.get('heading_bonus', 0.0):.3f}
- page bonus: {stats.get('page_bonus', 0.0):.3f}

Neighboring context:
{neighbor_text if neighbor_text else "(none)"}

Return:
1) the change type,
2) concise explanation,
3) possible compliance/regulatory impact,
4) a confidence estimate in plain language.
"""


def heuristic_summary(context_pack: dict) -> str:
    legacy = context_pack.get("legacy") or {}
    modern = context_pack.get("modern") or {}
    change_type = context_pack.get("change_type", "modified")
    legacy_text = (legacy.get("text") or "").strip()
    modern_text = (modern.get("text") or "").strip()

    if change_type == "added":
        return f"Added content in the modern policy. Key text: {modern_text[:280]}"
    if change_type == "removed":
        return f"Removed legacy content. Key text: {legacy_text[:280]}"

    # lightweight heuristic compare
    import difflib

    sm = difflib.SequenceMatcher(None, legacy_text.lower(), modern_text.lower())
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        left = legacy_text[i1:i2].strip()
        right = modern_text[j1:j2].strip()
        if tag == "replace":
            ops.append(f"Rephrased or replaced '{left[:80]}' with '{right[:80]}'.")
        elif tag == "delete":
            ops.append(f"Removed '{left[:80]}'.")
        elif tag == "insert":
            ops.append(f"Added '{right[:80]}'.")
    if not ops:
        ops.append("The wording is substantially similar, with only minor edits.")
    impact = "This may affect obligations, exceptions, or compliance scope." if change_type != "unchanged" else "Low impact change."
    return " ".join(ops) + f" {impact}"


def stream_summary(context_pack: dict, backend: LLMBackend, cfg: AppConfig) -> Iterator[str]:
    if not backend.ready or backend.model is None or backend.tokenizer is None or TextIteratorStreamer is None or torch is None:
        text = heuristic_summary(context_pack)
        # stream in small chunks to mimic real-time generation
        buf = ""
        for chunk in _chunk_text(text, 60):
            buf += chunk
            yield buf
        return

    prompt = build_prompt(context_pack, cfg)
    tokenizer = backend.tokenizer
    model = backend.model

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=cfg.temperature > 0,
        )
        import threading

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        acc = ""
        for chunk in streamer:
            acc += chunk
            yield acc.strip()
    except Exception as exc:
        logger.warning("LLM streaming failed; using heuristic summary: {}", exc)
        yield heuristic_summary(context_pack)


def summarize_context_pack(context_pack: dict, backend: LLMBackend, cfg: AppConfig) -> str:
    text = ""
    for chunk in stream_summary(context_pack, backend, cfg):
        text = chunk
    return text


def _chunk_text(text: str, size: int) -> list[str]:
    text = text.strip()
    if not text:
        return [""]
    return [text[i : i + size] for i in range(0, len(text), size)]
