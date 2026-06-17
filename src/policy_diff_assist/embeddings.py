from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import torch
import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as sk_normalize

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
@dataclass(slots=True)
class EmbeddingBackend:
    name: str
    dim: int
    model: object | None = None
    fallback: bool = False


@lru_cache(maxsize=4)
def load_embedding_backend(model_name: str, fallback_name: str, hf_token: str | None = None) -> EmbeddingBackend:
    if SentenceTransformer is not None:
        try:
            kwargs = {}
            if hf_token:
                kwargs["token"] = hf_token
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            logger.info("Configuring ST with {} device", device)
            model = SentenceTransformer(    # type: ignore[arg-type]
                model_name,
                device=device,
                **kwargs,
            ) 
            dim = int(model.get_embedding_dimension())
            logger.info("Loaded embedding model {}", model_name)
            return EmbeddingBackend(name=model_name, dim=dim, model=model, fallback=False)
        except Exception as exc:
            logger.warning("Could not load embedding model {}: {}", model_name, exc)

    logger.warning("Using hash-based embeddings fallback")
    return EmbeddingBackend(name="hashing-fallback", dim=768, model=None, fallback=True)


def _hash_embeddings(texts: list[str], dim: int = 768) -> np.ndarray:
    vectorizer = HashingVectorizer(n_features=dim, alternate_sign=False, norm=None, analyzer="word")
    mat = vectorizer.transform(texts)
    arr = mat.toarray().astype(np.float32, copy=False)
    arr = sk_normalize(arr, norm="l2", axis=1, copy=False)
    return arr.astype(np.float32, copy=False)

def embed_two_corpora(
    backend: EmbeddingBackend,
    legacy_texts: list[str],
    modern_texts: list[str],
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Batching the legacy & modern text together instead of individual ST calls for embedding.")

    all_texts = legacy_texts + modern_texts
    if not all_texts:
        empty = np.zeros((0, backend.dim), dtype=np.float32)
        return empty, empty

    all_emb = embed_texts(backend, all_texts, batch_size=batch_size)

    logger.info("Seperated the legacy & modern embeddings from one another.")
    
    split_at = len(legacy_texts)
    return all_emb[:split_at], all_emb[split_at:]

def embed_texts(backend: EmbeddingBackend, texts: list[str], batch_size: int = 2048) -> np.ndarray:
    if not texts:
        return np.zeros((0, backend.dim), dtype=np.float32)

    if backend.fallback or backend.model is None:
        return _hash_embeddings(texts, dim=backend.dim)

    model = backend.model
    try:
        logger.info("Started embedding")
        emb = model.encode(  # type: ignore[attr-defined]
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        logger.info("Returning embedding as numpy arrays")
        return np.asarray(emb, dtype=np.float32)
    except Exception as exc:
        logger.warning("Embedding encode failed ({}); falling back to hash vectors", exc)
        return _hash_embeddings(texts, dim=backend.dim)


def embed_iterable(backend: EmbeddingBackend, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
    return embed_texts(backend, list(texts), batch_size=batch_size)
