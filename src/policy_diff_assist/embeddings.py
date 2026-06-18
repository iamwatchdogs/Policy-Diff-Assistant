from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal, Sequence

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

from .models import DocumentTree, SourceNode


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
    backend_type: Literal["transformers", "sentence_transformers", "hash"] = (
        "transformers"
    )


@dataclass(slots=True)
class EmbeddedNode:
    node_id: str
    level: str  # "section" | "paragraph" | "clause" | ...
    text: str
    page: int
    kind: str
    path: list[str]
    parent_id: str | None
    token_count: int
    stable_hash: str
    section_id: str | None = None
    context_text: str | None = None


@dataclass(slots=True)
class HierarchicalEmbeddingBundle:
    section_node_ids: list[str]
    section_texts: list[str]
    section_embeddings: np.ndarray
    paragraph_node_ids: list[str]
    paragraph_texts: list[str]
    paragraph_embeddings: np.ndarray
    section_index_by_node_id: dict[str, int]
    paragraph_index_by_node_id: dict[str, int]
    section_meta: list[EmbeddedNode]
    paragraph_meta: list[EmbeddedNode]


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


def _infer_dim(
    model: object, tokenizer: object | None = None, device: str = "cpu"
) -> int:
    for attr in (
        "sentence_embedding_dimension",
        "hidden_size",
        "dim",
        "embedding_size",
        "word_embed_proj_dim",
    ):
        try:
            value = getattr(getattr(model, "config", None), attr, None)
            if value is not None:
                return int(value)
        except Exception:
            pass

    if tokenizer is not None and hasattr(model, "__call__"):
        try:
            sample = tokenizer(
                "dimension probe", return_tensors="pt", padding=True, truncation=True
            )
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
def load_embedding_backend(
    model_name: str, fallback_name: str, hf_token: str | None = None
) -> EmbeddingBackend:
    """
    Load the embedding backend with GPU-aware transformers first, then
    SentenceTransformer, then a hashing fallback.

    This keeps the existing fast path while making the backend suitable for
    tree-aware embedding workflows.
    """
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

        logger.info(
            "Loaded embedding model {} on {} (dim={}, max_length={})",
            model_name,
            device,
            dim,
            max_length,
        )
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
        logger.warning(
            "Could not load transformers embedding model {}: {}", model_name, exc
        )

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
            logger.warning(
                "Could not load SentenceTransformer model {}: {}", model_name, exc
            )

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
            logger.warning(
                "Could not load fallback SentenceTransformer model {}: {}",
                fallback_name,
                exc,
            )

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
    vectorizer = HashingVectorizer(
        n_features=dim, alternate_sign=False, norm=None, analyzer="word"
    )
    mat = vectorizer.transform(texts)
    arr = mat.toarray().astype(np.float32, copy=False)
    arr = sk_normalize(arr, norm="l2", axis=1, copy=False)
    return arr.astype(np.float32, copy=False)


def _batched(items: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def _mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-9)
    return summed / denom


def _ordered_nodes(
    tree: DocumentTree,
    kinds: set[str] | None = None,
) -> list[SourceNode]:
    """
    Preserve the document's structural order. leaf_ids is already ordered by the
    ingestion stage; we keep that and filter by kind.
    """
    nodes: list[SourceNode] = []
    for node_id in tree.leaf_ids:
        node = tree.nodes.get(node_id)
        if node is None:
            continue
        if kinds is not None and node.kind not in kinds:
            continue
        nodes.append(node)
    return nodes


def _path_text(path: list[str]) -> str:
    return " > ".join(p for p in path if p)


def _ancestor_section(tree: DocumentTree, node: SourceNode) -> SourceNode | None:
    parent_id = node.parent_id
    while parent_id:
        parent = tree.nodes.get(parent_id)
        if parent is None:
            return None
        if parent.kind == "section":
            return parent
        parent_id = parent.parent_id
    return None


def _sibling_texts(tree: DocumentTree, node: SourceNode, window: int = 1) -> list[str]:
    idx = tree.leaf_positions.get(node.node_id)
    if idx is None:
        return []
    texts: list[str] = []
    start = max(0, idx - window)
    end = min(len(tree.leaf_ids), idx + window + 1)
    for pos in range(start, end):
        leaf_id = tree.leaf_ids[pos]
        if leaf_id == node.node_id:
            continue
        sibling = tree.nodes.get(leaf_id)
        if sibling is not None and sibling.text:
            texts.append(sibling.text)
    return texts


def _truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _compose_section_text(
    tree: DocumentTree,
    node: SourceNode,
    child_limit: int = 4,
    child_chars: int = 1200,
    context_chars: int = 2200,
) -> str:
    """
    Build a section representation that includes:
    - heading path
    - section title
    - a short preview of the first few child paragraphs/bullets
    This makes section-level embeddings substantially more semantic.
    """
    parts: list[str] = []

    if node.path:
        parts.append(f"Heading path: {_path_text(node.path)}")

    if node.text:
        parts.append(f"Section title: {node.text.strip()}")

    child_texts: list[str] = []
    total_child_chars = 0
    for child_id in node.children:
        child = tree.nodes.get(child_id)
        if child is None:
            continue
        if child.kind not in {"paragraph", "bullet", "clause"}:
            continue
        child_text = child.text.strip()
        if not child_text:
            continue
        child_texts.append(child_text)
        total_child_chars += len(child_text)
        if len(child_texts) >= child_limit or total_child_chars >= child_chars:
            break

    if child_texts:
        parts.append("Preview: " + " ".join(child_texts))

    return _truncate_text("\n\n".join(parts), context_chars)


def _compose_paragraph_text(
    tree: DocumentTree,
    node: SourceNode,
    sibling_window: int = 1,
    context_chars: int = 1800,
) -> str:
    """
    Build a paragraph/clause representation with section context.
    This helps the encoder see the tree structure without changing the schema.
    """
    parts: list[str] = []

    section = _ancestor_section(tree, node)
    if section is not None:
        if section.path:
            parts.append(f"Section path: {_path_text(section.path)}")
        if section.text:
            parts.append(f"Section heading: {section.text.strip()}")

    if node.path:
        parts.append(f"Node path: {_path_text(node.path)}")

    parts.append(f"{node.kind.capitalize()}: {node.text.strip()}")

    siblings = _sibling_texts(tree, node, window=sibling_window)
    if siblings:
        parts.append(
            "Neighbor context: "
            + " ".join(_truncate_text(t, 320) for t in siblings[:2])
        )

    return _truncate_text("\n\n".join(parts), context_chars)


def _compose_generic_text(node: SourceNode, context_chars: int = 1200) -> str:
    parts: list[str] = []
    if node.path:
        parts.append(f"Path: {_path_text(node.path)}")
    if node.text:
        parts.append(node.text.strip())
    return _truncate_text("\n\n".join(parts), context_chars)


def _build_embedded_node(
    tree: DocumentTree,
    node: SourceNode,
    level: str,
    context_window: int = 1,
    section_child_limit: int = 4,
) -> EmbeddedNode:
    section = _ancestor_section(tree, node)
    section_id = section.node_id if section is not None else None
    context_text = None

    if level == "section":
        context_text = _compose_section_text(
            tree,
            node,
            child_limit=section_child_limit,
        )
    elif level in {"paragraph", "clause", "bullet"}:
        context_text = _compose_paragraph_text(
            tree,
            node,
            sibling_window=context_window,
        )
    else:
        context_text = _compose_generic_text(node)

    return EmbeddedNode(
        node_id=node.node_id,
        level=level,
        text=node.text,
        page=node.page,
        kind=node.kind,
        path=node.path[:],
        parent_id=node.parent_id,
        token_count=node.token_count,
        stable_hash=node.stable_hash,
        section_id=section_id,
        context_text=context_text,
    )


def build_hierarchical_embedding_plan(
    tree: DocumentTree,
    include_sections: bool = True,
    include_paragraphs: bool = True,
    include_clauses: bool = True,
    include_bullets: bool = True,
    context_window: int = 1,
    section_child_limit: int = 4,
) -> tuple[list[EmbeddedNode], list[EmbeddedNode]]:
    """
    Build two ordered levels of embedding items from the tree:
    1) section-level nodes
    2) paragraph/clause/bullet-level nodes

    This aligns with ingestion's hierarchical tree and keeps embeddings
    section-aware before paragraph-wise similarity.
    """
    section_meta: list[EmbeddedNode] = []
    paragraph_meta: list[EmbeddedNode] = []

    for node in _ordered_nodes(tree, kinds={"section"}):
        if include_sections:
            section_meta.append(
                _build_embedded_node(
                    tree,
                    node,
                    level="section",
                    context_window=context_window,
                    section_child_limit=section_child_limit,
                )
            )

    target_kinds = set()
    if include_paragraphs:
        target_kinds.add("paragraph")
    if include_clauses:
        target_kinds.add("clause")
    if include_bullets:
        target_kinds.add("bullet")

    for node in _ordered_nodes(tree, kinds=target_kinds):
        paragraph_meta.append(
            _build_embedded_node(
                tree,
                node,
                level=node.kind,
                context_window=context_window,
                section_child_limit=section_child_limit,
            )
        )

    return section_meta, paragraph_meta


def _embed_context_texts(
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
            logger.warning(
                "SentenceTransformer encode failed ({}); falling back to hash vectors",
                exc,
            )
            return _hash_embeddings(texts, dim=backend.dim)

    tokenizer = backend.tokenizer
    model = backend.model
    device = backend.device
    max_length = max_length or backend.max_length

    result = np.empty((len(texts), backend.dim), dtype=np.float32)
    write_pos = 0

    amp_enabled = device == "cuda" and backend.dtype is not None
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=backend.dtype)
        if amp_enabled
        else nullcontext()
    )

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

                if (
                    hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    emb = outputs.pooler_output
                else:
                    emb = _mean_pool(
                        outputs.last_hidden_state,
                        enc["attention_mask"],  # type: ignore[attr-defined]
                    )

                emb = F.normalize(emb, p=2, dim=1)
                batch_np = emb.detach().to("cpu").float().numpy()

                end_pos = write_pos + batch_np.shape[0]
                result[write_pos:end_pos] = batch_np
                write_pos = end_pos

        return result
    except Exception as exc:
        logger.warning(
            "Transformers embedding failed ({}); falling back to hash vectors", exc
        )
        return _hash_embeddings(texts, dim=backend.dim)


def embed_texts(
    backend: EmbeddingBackend,
    texts: list[str],
    batch_size: int = 1024,
    max_length: int | None = None,
) -> np.ndarray:
    """
    Backwards-compatible text embedding API.
    """
    return _embed_context_texts(
        backend=backend,
        texts=texts,
        batch_size=batch_size,
        max_length=max_length,
    )


def embed_nodes(
    backend: EmbeddingBackend,
    nodes: list[EmbeddedNode],
    batch_size: int = 1024,
    max_length: int | None = None,
) -> np.ndarray:
    texts = [node.context_text or node.text for node in nodes]
    return _embed_context_texts(
        backend=backend,
        texts=texts,
        batch_size=batch_size,
        max_length=max_length,
    )


def embed_hierarchical_tree(
    backend: EmbeddingBackend,
    tree: DocumentTree,
    batch_size: int = 1024,
    max_length: int | None = None,
    include_sections: bool = True,
    include_paragraphs: bool = True,
    include_clauses: bool = True,
    include_bullets: bool = True,
    context_window: int = 1,
    section_child_limit: int = 4,
) -> HierarchicalEmbeddingBundle:
    """
    Embed the tree in two semantic passes:
    1) sections
    2) paragraphs / clauses / bullets

    This is the tree-aware API that should align with the ingestion logic.
    """
    section_meta, paragraph_meta = build_hierarchical_embedding_plan(
        tree,
        include_sections=include_sections,
        include_paragraphs=include_paragraphs,
        include_clauses=include_clauses,
        include_bullets=include_bullets,
        context_window=context_window,
        section_child_limit=section_child_limit,
    )

    section_texts = [node.context_text or node.text for node in section_meta]
    paragraph_texts = [node.context_text or node.text for node in paragraph_meta]

    section_embeddings = _embed_context_texts(
        backend=backend,
        texts=section_texts,
        batch_size=batch_size,
        max_length=max_length,
    )
    paragraph_embeddings = _embed_context_texts(
        backend=backend,
        texts=paragraph_texts,
        batch_size=batch_size,
        max_length=max_length,
    )

    section_index_by_node_id = {
        node.node_id: idx for idx, node in enumerate(section_meta)
    }
    paragraph_index_by_node_id = {
        node.node_id: idx for idx, node in enumerate(paragraph_meta)
    }

    return HierarchicalEmbeddingBundle(
        section_node_ids=[node.node_id for node in section_meta],
        section_texts=section_texts,
        section_embeddings=section_embeddings,
        paragraph_node_ids=[node.node_id for node in paragraph_meta],
        paragraph_texts=paragraph_texts,
        paragraph_embeddings=paragraph_embeddings,
        section_index_by_node_id=section_index_by_node_id,
        paragraph_index_by_node_id=paragraph_index_by_node_id,
        section_meta=section_meta,
        paragraph_meta=paragraph_meta,
    )


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

    all_emb = _embed_context_texts(
        backend, all_texts, batch_size=batch_size, max_length=max_length
    )
    split_at = len(legacy_texts)
    return all_emb[:split_at], all_emb[split_at:]


def embed_iterable(
    backend: EmbeddingBackend, texts: Iterable[str], batch_size: int = 1024
) -> np.ndarray:
    return embed_texts(backend, list(texts), batch_size=batch_size)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Ensure row-wise L2 normalization and a float32 output array.
    """
    if embeddings.size == 0:
        return embeddings.astype(np.float32, copy=False)
    arr = np.asarray(embeddings, dtype=np.float32)
    return sk_normalize(arr, norm="l2", axis=1, copy=False).astype(
        np.float32, copy=False
    )
