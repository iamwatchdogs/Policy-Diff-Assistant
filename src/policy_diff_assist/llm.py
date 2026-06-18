from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

from loguru import logger

try:  # vLLM preferred
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    import torch
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TextIteratorStreamer = None  # type: ignore
    torch = None  # type: ignore

from policy_diff_assist.config import AppConfig


@dataclass(slots=True)
class LLMBackend:
    provider: str  # "vllm" | "transformers" | "heuristic"
    name: str
    engine: object | None = None
    tokenizer: object | None = None
    model: object | None = None
    ready: bool = False


def _apply_hf_token(cfg: AppConfig) -> None:
    if cfg.hf_token:
        os.environ.setdefault("HF_TOKEN", cfg.hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", cfg.hf_token)


def load_llm_backend(cfg: AppConfig) -> LLMBackend:
    _apply_hf_token(cfg)

    if cfg.use_vllm and LLM is not None and SamplingParams is not None:
        try:
            kwargs = dict(
                model=cfg.llm_model_name,
                trust_remote_code=True,
                download_dir=str(cfg.data_dir / "hf_cache"),
                dtype="auto",
            )
            # Pass token explicitly when available.
            if cfg.hf_token:
                kwargs["hf_token"] = cfg.hf_token

            engine = LLM(**kwargs)
            logger.info("Loaded vLLM model {}", cfg.llm_model_name)
            return LLMBackend(
                provider="vllm",
                name=cfg.llm_model_name,
                engine=engine,
                ready=True,
            )
        except TypeError as exc:
            # Some vLLM builds may not accept hf_token on the constructor.
            if cfg.hf_token:
                logger.warning(
                    "vLLM constructor rejected hf_token; retrying with env vars only. Error: {}",
                    exc,
                )
            try:
                engine = LLM(
                    model=cfg.llm_model_name,
                    trust_remote_code=True,
                    download_dir=str(cfg.data_dir / "hf_cache"),
                    dtype="auto",
                )
                logger.info("Loaded vLLM model {}", cfg.llm_model_name)
                return LLMBackend(
                    provider="vllm",
                    name=cfg.llm_model_name,
                    engine=engine,
                    ready=True,
                )
            except Exception as inner_exc:
                logger.warning(
                    "Could not load vLLM model {}. Error: {}",
                    cfg.llm_model_name,
                    inner_exc,
                )
        except Exception as exc:
            logger.warning(
                "Could not load vLLM model {}. Error: {}",
                cfg.llm_model_name,
                exc,
            )

    if AutoTokenizer is not None and AutoModelForCausalLM is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.llm_model_name,
                token=cfg.hf_token,
                use_fast=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                cfg.llm_model_name,
                token=cfg.hf_token,
                torch_dtype=getattr(torch, "float16", None)
                if torch is not None
                else None,
                device_map="auto",
            )
            logger.info("Loaded Transformers model {}", cfg.llm_model_name)
            return LLMBackend(
                provider="transformers",
                name=cfg.llm_model_name,
                tokenizer=tokenizer,
                model=model,
                ready=True,
            )
        except Exception as exc:
            logger.warning(
                "Could not load Transformers model {}. Falling back to heuristic mode. Error: {}",
                cfg.llm_model_name,
                exc,
            )

    logger.warning("LLM backend unavailable; using heuristic summaries.")
    return LLMBackend(provider="heuristic", name="heuristic-fallback", ready=False)


def build_prompt(context_pack: dict, cfg: AppConfig) -> str:
    legacy = context_pack.get("legacy") or {}
    modern = context_pack.get("modern") or {}
    legacy_text = legacy.get("text") or ""
    modern_text = modern.get("text") or ""
    stats = context_pack.get("stats") or {}
    neighbors = context_pack.get("legacy_neighbors", []) + context_pack.get(
        "modern_neighbors", []
    )

    neighbor_text = "\n".join(
        f"- {n.get('page')}: {n.get('text', '')[:500]}" for n in neighbors[:6]
    )

    return f"""You are a policy compliance assistant.
Write a concise but useful explanation of the semantic difference between the two policy snippets.

Legacy snippet:
{legacy_text}

Modern snippet:
{modern_text}

Similarity stats:
- cosine: {stats.get("cosine", 0.0):.3f}
- lexical: {stats.get("lexical", 0.0):.3f}
- heading bonus: {stats.get("heading_bonus", 0.0):.3f}
- page bonus: {stats.get("page_bonus", 0.0):.3f}

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

    import difflib

    sm = difflib.SequenceMatcher(None, legacy_text.lower(), modern_text.lower())
    ops: list[str] = []
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
    impact = (
        "This may affect obligations, exceptions, or compliance scope."
        if change_type != "unchanged"
        else "Low impact change."
    )
    return " ".join(ops) + f" {impact}"


def _yield_text_chunks(text: str, chunk_size: int = 70) -> Iterator[str]:
    text = (text or "").strip()
    if not text:
        yield ""
        return
    for i in range(0, len(text), chunk_size):
        yield text[: i + chunk_size]


def _stream_vllm_text(
    backend: LLMBackend, prompt: str, cfg: AppConfig
) -> Iterator[str]:
    params = SamplingParams(
        temperature=cfg.temperature,
        top_p=0.9,
        max_tokens=cfg.max_new_tokens,
    )
    outputs = backend.engine.generate([prompt], params)  # type: ignore[union-attr]
    text = ""
    if outputs:
        first = outputs[0]
        if getattr(first, "outputs", None):
            text = first.outputs[0].text or ""
    for chunk in _yield_text_chunks(text, 70):
        yield chunk


def _stream_transformers_text(
    backend: LLMBackend, prompt: str, cfg: AppConfig
) -> Iterator[str]:
    tokenizer = backend.tokenizer
    model = backend.model
    if (
        tokenizer is None
        or model is None
        or TextIteratorStreamer is None
        or torch is None
    ):
        yield heuristic_summary({})
        return

    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        do_sample=cfg.temperature > 0,
    )

    import threading

    thread = threading.Thread(
        target=model.generate, kwargs=generation_kwargs, daemon=True
    )
    thread.start()

    acc = ""
    for chunk in streamer:
        acc += chunk
        yield acc.strip()


def stream_summary(
    context_pack: dict, backend: LLMBackend, cfg: AppConfig
) -> Iterator[str]:
    logger.info("Streaming summary through {}", backend.provider)

    if backend.provider == "vllm" and backend.ready and backend.engine is not None:
        prompt = build_prompt(context_pack, cfg)
        yield from _stream_vllm_text(backend, prompt, cfg)
        return

    if backend.provider == "transformers" and backend.ready:
        prompt = build_prompt(context_pack, cfg)
        yield from _stream_transformers_text(backend, prompt, cfg)
        return

    text = heuristic_summary(context_pack)
    acc = ""
    for chunk in _yield_text_chunks(text, 60):
        acc += chunk
        yield acc.strip()


def summarize_context_pack(
    context_pack: dict, backend: LLMBackend, cfg: AppConfig
) -> str:
    text = ""
    for chunk in stream_summary(context_pack, backend, cfg):
        text = chunk
    return text
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

from loguru import logger

try:  # vLLM preferred
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    import torch
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TextIteratorStreamer = None  # type: ignore
    torch = None  # type: ignore

from policy_diff_assist.config import AppConfig


@dataclass(slots=True)
class LLMBackend:
    provider: str  # "vllm" | "transformers" | "heuristic"
    name: str
    engine: object | None = None
    tokenizer: object | None = None
    model: object | None = None
    ready: bool = False


def _apply_hf_token(cfg: AppConfig) -> None:
    if cfg.hf_token:
        os.environ.setdefault("HF_TOKEN", cfg.hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", cfg.hf_token)


def load_llm_backend(cfg: AppConfig) -> LLMBackend:
    _apply_hf_token(cfg)

    if cfg.use_vllm and LLM is not None and SamplingParams is not None:
        try:
            kwargs = dict(
                model=cfg.llm_model_name,
                trust_remote_code=True,
                download_dir=str(cfg.data_dir / "hf_cache"),
                dtype="auto",
            )
            # Pass token explicitly when available.
            if cfg.hf_token:
                kwargs["hf_token"] = cfg.hf_token

            engine = LLM(**kwargs)
            logger.info("Loaded vLLM model {}", cfg.llm_model_name)
            return LLMBackend(
                provider="vllm",
                name=cfg.llm_model_name,
                engine=engine,
                ready=True,
            )
        except TypeError as exc:
            # Some vLLM builds may not accept hf_token on the constructor.
            if cfg.hf_token:
                logger.warning(
                    "vLLM constructor rejected hf_token; retrying with env vars only. Error: {}",
                    exc,
                )
            try:
                engine = LLM(
                    model=cfg.llm_model_name,
                    trust_remote_code=True,
                    download_dir=str(cfg.data_dir / "hf_cache"),
                    dtype="auto",
                )
                logger.info("Loaded vLLM model {}", cfg.llm_model_name)
                return LLMBackend(
                    provider="vllm",
                    name=cfg.llm_model_name,
                    engine=engine,
                    ready=True,
                )
            except Exception as inner_exc:
                logger.warning(
                    "Could not load vLLM model {}. Error: {}",
                    cfg.llm_model_name,
                    inner_exc,
                )
        except Exception as exc:
            logger.warning(
                "Could not load vLLM model {}. Error: {}",
                cfg.llm_model_name,
                exc,
            )

    if AutoTokenizer is not None and AutoModelForCausalLM is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.llm_model_name,
                token=cfg.hf_token,
                use_fast=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                cfg.llm_model_name,
                token=cfg.hf_token,
                torch_dtype=getattr(torch, "float16", None)
                if torch is not None
                else None,
                device_map="auto",
            )
            logger.info("Loaded Transformers model {}", cfg.llm_model_name)
            return LLMBackend(
                provider="transformers",
                name=cfg.llm_model_name,
                tokenizer=tokenizer,
                model=model,
                ready=True,
            )
        except Exception as exc:
            logger.warning(
                "Could not load Transformers model {}. Falling back to heuristic mode. Error: {}",
                cfg.llm_model_name,
                exc,
            )

    logger.warning("LLM backend unavailable; using heuristic summaries.")
    return LLMBackend(provider="heuristic", name="heuristic-fallback", ready=False)


def build_prompt(context_pack: dict, cfg: AppConfig) -> str:
    legacy = context_pack.get("legacy") or {}
    modern = context_pack.get("modern") or {}
    legacy_text = legacy.get("text") or ""
    modern_text = modern.get("text") or ""
    stats = context_pack.get("stats") or {}
    neighbors = context_pack.get("legacy_neighbors", []) + context_pack.get(
        "modern_neighbors", []
    )

    neighbor_text = "\n".join(
        f"- {n.get('page')}: {n.get('text', '')[:500]}" for n in neighbors[:6]
    )

    return f"""You are a policy compliance assistant.
Write a concise but useful explanation of the semantic difference between the two policy snippets.

Legacy snippet:
{legacy_text}

Modern snippet:
{modern_text}

Similarity stats:
- cosine: {stats.get("cosine", 0.0):.3f}
- lexical: {stats.get("lexical", 0.0):.3f}
- heading bonus: {stats.get("heading_bonus", 0.0):.3f}
- page bonus: {stats.get("page_bonus", 0.0):.3f}

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

    import difflib

    sm = difflib.SequenceMatcher(None, legacy_text.lower(), modern_text.lower())
    ops: list[str] = []
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
    impact = (
        "This may affect obligations, exceptions, or compliance scope."
        if change_type != "unchanged"
        else "Low impact change."
    )
    return " ".join(ops) + f" {impact}"


def _yield_text_chunks(text: str, chunk_size: int = 70) -> Iterator[str]:
    text = (text or "").strip()
    if not text:
        yield ""
        return
    for i in range(0, len(text), chunk_size):
        yield text[: i + chunk_size]


def _stream_vllm_text(
    backend: LLMBackend, prompt: str, cfg: AppConfig
) -> Iterator[str]:
    params = SamplingParams(
        temperature=cfg.temperature,
        top_p=0.9,
        max_tokens=cfg.max_new_tokens,
    )
    outputs = backend.engine.generate([prompt], params)  # type: ignore[union-attr]
    text = ""
    if outputs:
        first = outputs[0]
        if getattr(first, "outputs", None):
            text = first.outputs[0].text or ""
    for chunk in _yield_text_chunks(text, 70):
        yield chunk


def _stream_transformers_text(
    backend: LLMBackend, prompt: str, cfg: AppConfig
) -> Iterator[str]:
    tokenizer = backend.tokenizer
    model = backend.model
    if (
        tokenizer is None
        or model is None
        or TextIteratorStreamer is None
        or torch is None
    ):
        yield heuristic_summary({})
        return

    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        do_sample=cfg.temperature > 0,
    )

    import threading

    thread = threading.Thread(
        target=model.generate, kwargs=generation_kwargs, daemon=True
    )
    thread.start()

    acc = ""
    for chunk in streamer:
        acc += chunk
        yield acc.strip()


def stream_summary(
    context_pack: dict, backend: LLMBackend, cfg: AppConfig
) -> Iterator[str]:
    logger.info("Streaming summary through {}", backend.provider)

    if backend.provider == "vllm" and backend.ready and backend.engine is not None:
        prompt = build_prompt(context_pack, cfg)
        yield from _stream_vllm_text(backend, prompt, cfg)
        return

    if backend.provider == "transformers" and backend.ready:
        prompt = build_prompt(context_pack, cfg)
        yield from _stream_transformers_text(backend, prompt, cfg)
        return

    text = heuristic_summary(context_pack)
    acc = ""
    for chunk in _yield_text_chunks(text, 60):
        acc += chunk
        yield acc.strip()


def summarize_context_pack(
    context_pack: dict, backend: LLMBackend, cfg: AppConfig
) -> str:
    text = ""
    for chunk in stream_summary(context_pack, backend, cfg):
        text = chunk
    return text
