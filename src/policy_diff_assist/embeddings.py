from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as sk_normalize

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

from transformers import AutoModel, AutoTokenizer



@dataclass(slots=True)
class EmbeddingBackend:
    name: str
    dim: int
    tokenizer: object | None = None
    model: object | None = None
    device: str = "cpu"
    dtype: torch.dtype | None = None
    max_length: int = 512
    fallback: bool = False
    backend_type: Literal["transformers", "sentence_transformers", "hash"] = "transformers"  


def _configure_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    cpu_count = os.cpu_count() or 1
    try:
        torch.set_num_threads(min(32, cpu_count))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _safe_max_length(tokenizer: object, default: int = 512) -> int:
    raw = getattr(tokenizer, "model_max_length", default)
    try:
        raw_int = int(raw)
    except Exception:
        return default

    # Hugging Face tokenizers sometimes expose absurd sentinel values.
    if raw_int <= 0 or raw_int > 8192:
        return default
    return min(raw_int, default)


def _infer_dim(model: object, tokenizer: object | None = None, device: str = "cpu") -> int:
    for attr in ("sentence_embedding_dimension", "hidden_size", "dim", "embedding_size", "word_embed_proj_dim"):
        try:
            value = getattr(getattr(model, "config", None), attr, None)
            if value is not None:
                return int(value)
        except Exception:
            pass

    if tokenizer is not None and hasattr(model, "__call__"):
        try:
            sample = tokenizer("dimension probe", return_tensors="pt", padding=True, truncation=True)
            sample = {k: v.to(device) for k, v in sample.items()}
            with torch.inference_mode():
                outputs = model(**sample)  # type: ignore[misc]
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return int(outputs.pooler_output.shape[-1])
            if hasattr(outputs, "last_hidden_state"):
                return int(outputs.last_hidden_state.shape[-1])
        except Exception:
            pass

    return 768


@lru_cache(maxsize=4)
def load_embedding_backend(model_name: str, fallback_name: str, hf_token: str | None = None) -> EmbeddingBackend:
    _configure_runtime()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Fast path: use Transformers directly on GPU/ROCm.
    try:
        tok_kwargs = {"trust_remote_code": True}
        mdl_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
        }
        if hf_token:
            tok_kwargs["token"] = hf_token
            mdl_kwargs["token"] = hf_token

        tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        tokenizer.padding_side = "right"

        model = AutoModel.from_pretrained(model_name, **mdl_kwargs)
        model.to(device)
        model.eval()

        dim = _infer_dim(model, tokenizer, device=device)
        max_length = _safe_max_length(tokenizer, default=512)

        logger.info("Loaded embedding model {} on {} (dim={}, max_length={})", model_name, device, dim, max_length)
        return EmbeddingBackend(
            name=model_name,
            dim=dim,
            tokenizer=tokenizer,
            model=model,
            device=device,
            dtype=dtype,
            max_length=max_length,
            fallback=False,
            backend_type="transformers",
        )
    except Exception as exc:
        logger.warning("Could not load transformers embedding model {}: {}", model_name, exc)

    # Secondary fallback: SentenceTransformer
    if SentenceTransformer is not None:
        try:
            kwargs = {}
            if hf_token:
                kwargs["token"] = hf_token
            st_model = SentenceTransformer(model_name, **kwargs)  # type: ignore[arg-type]
            dim = int(st_model.get_embedding_dimension())
            logger.info("Loaded SentenceTransformer embedding model {}", model_name)
            return EmbeddingBackend(
                name=model_name,
                dim=dim,
                model=st_model,
                device=device,
                dtype=dtype,
                max_length=512,
                fallback=False,
                backend_type="sentence_transformers",
            )
        except Exception as exc:
            logger.warning("Could not load SentenceTransformer model {}: {}", model_name, exc)

        try:
            kwargs = {}
            if hf_token:
                kwargs["token"] = hf_token
            st_model = SentenceTransformer(fallback_name, **kwargs)  # type: ignore[arg-type]
            dim = int(st_model.get_embedding_dimension())
            logger.info("Loaded fallback SentenceTransformer model {}", fallback_name)
            return EmbeddingBackend(
                name=fallback_name,
                dim=dim,
                model=st_model,
                device=device,
                dtype=dtype,
                max_length=512,
                fallback=False,
                backend_type="sentence_transformers",
            )
        except Exception as exc:
            logger.warning("Could not load fallback SentenceTransformer model {}: {}", fallback_name, exc)

    logger.warning("Using hash-based embeddings fallback")
    return EmbeddingBackend(
        name="hashing-fallback",
        dim=768,
        model=None,
        device="cpu",
        dtype=torch.float32,
        max_length=512,
        fallback=True,
        backend_type="hash",
    )


def _hash_embeddings(texts: list[str], dim: int = 768) -> np.ndarray:
    vectorizer = HashingVectorizer(n_features=dim, alternate_sign=False, norm=None, analyzer="word")
    mat = vectorizer.transform(texts)
    arr = mat.toarray().astype(np.float32, copy=False)
    arr = sk_normalize(arr, norm="l2", axis=1, copy=False)
    return arr.astype(np.float32, copy=False)


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-9)
    return summed / denom


def embed_texts(
    backend: EmbeddingBackend,
    texts: list[str],
    batch_size: int = 1024,
    max_length: int | None = None,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, backend.dim), dtype=np.float32)

    if backend.fallback or backend.model is None:
        return _hash_embeddings(texts, dim=backend.dim)

    if backend.backend_type == "sentence_transformers":
        model = backend.model
        try:
            emb = model.encode(  # type: ignore[attr-defined]
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(emb, dtype=np.float32)
        except Exception as exc:
            logger.warning("SentenceTransformer encode failed ({}); falling back to hash vectors", exc)
            return _hash_embeddings(texts, dim=backend.dim)

    tokenizer = backend.tokenizer
    model = backend.model
    device = backend.device
    max_length = max_length or backend.max_length

    result = np.empty((len(texts), backend.dim), dtype=np.float32)
    write_pos = 0

    amp_enabled = device == "cuda" and backend.dtype is not None
    amp_ctx = torch.autocast(device_type="cuda", dtype=backend.dtype) if amp_enabled else nullcontext()

    try:
        with torch.inference_mode():
            for batch_texts in _batched(texts, batch_size):
                enc = tokenizer(  # type: ignore[operator]
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

                with amp_ctx:
                    outputs = model(**enc)  # type: ignore[misc]

                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output
                else:
                    emb = _mean_pool(outputs.last_hidden_state, enc["attention_mask"])  # type: ignore[attr-defined]

                emb = F.normalize(emb, p=2, dim=1)
                batch_np = emb.detach().to("cpu").float().numpy()

                end_pos = write_pos + batch_np.shape[0]
                result[write_pos:end_pos] = batch_np
                write_pos = end_pos

        return result
    except Exception as exc:
        logger.warning("Transformers embedding failed ({}); falling back to hash vectors", exc)
        return _hash_embeddings(texts, dim=backend.dim)


def embed_two_corpora(
    backend: EmbeddingBackend,
    legacy_texts: list[str],
    modern_texts: list[str],
    batch_size: int = 1024,
    max_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    all_texts = legacy_texts + modern_texts
    if not all_texts:
        empty = np.zeros((0, backend.dim), dtype=np.float32)
        return empty, empty

    all_emb = embed_texts(backend, all_texts, batch_size=batch_size, max_length=max_length)
    split_at = len(legacy_texts)
    return all_emb[:split_at], all_emb[split_at:]


def embed_iterable(backend: EmbeddingBackend, texts: Iterable[str], batch_size: int = 1024) -> np.ndarray:
    return embed_texts(backend, list(texts), batch_size=batch_size)