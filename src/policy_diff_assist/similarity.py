from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
import multiprocessing as mp
import os
from typing import Sequence

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional dependency
    linear_sum_assignment = None  # type: ignore[assignment]


def _as_float32_numpy(x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={arr.shape}")
    return arr


def _normalize_rows_numpy(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return x / norms


def cosine_matrix(
    left: np.ndarray | "torch.Tensor",
    right: np.ndarray | "torch.Tensor",
    *,
    normalize: bool = True,
    use_gpu: str = "auto",
    return_numpy: bool = True,
) -> np.ndarray | "torch.Tensor":
    """
    Fast cosine matrix with optional ROCm/CUDA acceleration via PyTorch.

    - If torch is available and use_gpu != "cpu", the matmul is executed on GPU.
    - If inputs are already torch tensors, they stay on the same device.
    - If inputs are numpy arrays, they are moved only for the cosine step.
    """
    if (
        getattr(left, "numel", None) is not None
        and getattr(right, "numel", None) is not None
        and _TORCH_AVAILABLE
    ):
        left_tensor = left if isinstance(left, torch.Tensor) else torch.as_tensor(left)
        right_tensor = (
            right if isinstance(right, torch.Tensor) else torch.as_tensor(right)
        )

        if left_tensor.ndim != 2 or right_tensor.ndim != 2:
            raise ValueError(
                f"Expected 2D tensors, got {tuple(left_tensor.shape)} and {tuple(right_tensor.shape)}"
            )

        if use_gpu == "auto":
            device = (
                left_tensor.device
                if left_tensor.is_cuda or right_tensor.is_cuda
                else ("cuda" if torch.cuda.is_available() else left_tensor.device)
            )
        elif use_gpu == "gpu":
            device = "cuda"
        else:
            device = "cpu"

        left_tensor = left_tensor.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        right_tensor = right_tensor.to(
            device=device, dtype=torch.float32, non_blocking=True
        )

        if left_tensor.numel() == 0 or right_tensor.numel() == 0:
            out = torch.zeros(
                (left_tensor.shape[0], right_tensor.shape[0]),
                dtype=torch.float32,
                device=device,
            )
            return out.cpu().numpy() if return_numpy else out

        if normalize:
            left_tensor = F.normalize(left_tensor, p=2, dim=1)
            right_tensor = F.normalize(right_tensor, p=2, dim=1)

        out = left_tensor @ right_tensor.T
        return out.cpu().numpy() if return_numpy else out

    l_np = _as_float32_numpy(left)
    r_np = _as_float32_numpy(right)
    if l_np.size == 0 or r_np.size == 0:
        return np.zeros((l_np.shape[0], r_np.shape[0]), dtype=np.float32)

    if normalize:
        l_np = _normalize_rows_numpy(l_np)
        r_np = _normalize_rows_numpy(r_np)

    return l_np @ r_np.T


# ---------------------------------------------------------------------------
# Lexical similarity
# ---------------------------------------------------------------------------

_LEXICAL_RIGHTS: list[str] = []


def _init_lexical_pool(right_texts: list[str]) -> None:
    global _LEXICAL_RIGHTS
    _LEXICAL_RIGHTS = right_texts


def _lexical_row(row_text: str) -> np.ndarray:
    return np.fromiter(
        (
            SequenceMatcher(None, row_text, right_text).ratio()
            for right_text in _LEXICAL_RIGHTS
        ),
        dtype=np.float32,
        count=len(_LEXICAL_RIGHTS),
    )


def lexical_score(a: str, b: str) -> float:
    # Keep the exact same logic, but normalize casing once.
    return float(SequenceMatcher(None, a.lower(), b.lower()).ratio())


def lexical_matrix(
    left_texts: Sequence[str],
    right_texts: Sequence[str],
    *,
    max_workers: int | None = None,
) -> np.ndarray:
    left = [t.lower() for t in left_texts]
    right = [t.lower() for t in right_texts]

    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.float32)

    # Small matrices are faster serially.
    if len(left) * len(right) < 4096:
        return np.vstack([_lexical_row_serial(row, right) for row in left])

    if max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = max(1, cpu - 2)

    try:
        ctx = mp.get_context("fork")
        chunksize = max(1, len(left) // (max_workers * 4))
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_lexical_pool,
            initargs=(right,),
        ) as ex:
            rows = list(ex.map(_lexical_row, left, chunksize=chunksize))
        return np.vstack(rows).astype(np.float32, copy=False)
    except Exception as exc:
        logger.warning("Falling back to serial lexical scoring: {}", exc)
        return np.vstack([_lexical_row_serial(row, right) for row in left])


def _lexical_row_serial(row_text: str, right_texts: Sequence[str]) -> np.ndarray:
    return np.fromiter(
        (
            SequenceMatcher(None, row_text, right_text).ratio()
            for right_text in right_texts
        ),
        dtype=np.float32,
        count=len(right_texts),
    )


# ---------------------------------------------------------------------------
# Structural similarity
# ---------------------------------------------------------------------------


def _path_to_tuple(path: list[str] | None) -> tuple[str, ...]:
    return tuple(path) if path else ()


def heading_similarity(path_a: list[str] | None, path_b: list[str] | None) -> float:
    if not path_a or not path_b:
        return 0.0
    if path_a == path_b:
        return 1.0
    common = 0
    for xa, xb in zip(path_a, path_b):
        if xa == xb:
            common += 1
        else:
            break
    return common / max(len(path_a), len(path_b), 1)


_HEADING_RIGHTS: list[tuple[str, ...]] = []


def _init_heading_pool(right_paths: list[tuple[str, ...]]) -> None:
    global _HEADING_RIGHTS
    _HEADING_RIGHTS = right_paths


def _heading_row(left_path: tuple[str, ...]) -> np.ndarray:
    right_paths = _HEADING_RIGHTS
    if not left_path or not right_paths:
        return np.zeros((len(right_paths),), dtype=np.float32)

    out = np.empty((len(right_paths),), dtype=np.float32)
    left_len = len(left_path)
    for j, right_path in enumerate(right_paths):
        if not right_path:
            out[j] = 0.0
            continue
        if left_path == right_path:
            out[j] = 1.0
            continue
        common = 0
        for xa, xb in zip(left_path, right_path):
            if xa == xb:
                common += 1
            else:
                break
        out[j] = common / max(left_len, len(right_path), 1)
    return out


def heading_similarity_matrix(
    left_paths: Sequence[list[str] | None],
    right_paths: Sequence[list[str] | None],
    *,
    max_workers: int | None = None,
) -> np.ndarray:
    left = [_path_to_tuple(p) for p in left_paths]
    right = [_path_to_tuple(p) for p in right_paths]

    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.float32)

    if len(left) * len(right) < 4096:
        return np.vstack([_heading_row_serial(p, right) for p in left])

    if max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = max(1, cpu - 2)

    try:
        ctx = mp.get_context("fork")
        chunksize = max(1, len(left) // (max_workers * 4))
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_heading_pool,
            initargs=(right,),
        ) as ex:
            rows = list(ex.map(_heading_row, left, chunksize=chunksize))
        return np.vstack(rows).astype(np.float32, copy=False)
    except Exception as exc:
        logger.warning("Falling back to serial heading scoring: {}", exc)
        return np.vstack([_heading_row_serial(p, right) for p in left])


def _heading_row_serial(
    left_path: tuple[str, ...], right_paths: Sequence[tuple[str, ...]]
) -> np.ndarray:
    if not left_path or not right_paths:
        return np.zeros((len(right_paths),), dtype=np.float32)

    out = np.empty((len(right_paths),), dtype=np.float32)
    left_len = len(left_path)
    for j, right_path in enumerate(right_paths):
        if not right_path:
            out[j] = 0.0
            continue
        if left_path == right_path:
            out[j] = 1.0
            continue
        common = 0
        for xa, xb in zip(left_path, right_path):
            if xa == xb:
                common += 1
            else:
                break
        out[j] = common / max(left_len, len(right_path), 1)
    return out


def kind_compatibility(kind_a: str | None, kind_b: str | None) -> float:
    if kind_a is None or kind_b is None:
        return 0.0
    if kind_a == kind_b:
        return 1.0 if kind_a == "section" else 0.6
    if {kind_a, kind_b} <= {"paragraph", "bullet"}:
        return 0.35
    return -0.5


def token_balance_bonus(token_count_a: int | None, token_count_b: int | None) -> float:
    if token_count_a is None or token_count_b is None:
        return 0.0
    a = max(int(token_count_a), 1)
    b = max(int(token_count_b), 1)
    return 0.05 * (min(a, b) / max(a, b))


def page_proximity_bonus(page_a: int | None, page_b: int | None) -> float:
    if page_a is None or page_b is None:
        return 0.0
    dist = abs(int(page_a) - int(page_b))
    if dist == 0:
        return 0.03
    if dist == 1:
        return 0.02
    if dist <= 3:
        return 0.01
    return 0.0


# ---------------------------------------------------------------------------
# Vectorized pairwise feature matrices
# ---------------------------------------------------------------------------

_KIND_TO_ID = {
    None: 0,
    "section": 1,
    "paragraph": 2,
    "bullet": 3,
}


def _kind_ids(kinds: Sequence[str | None]) -> np.ndarray:
    return np.asarray([_KIND_TO_ID.get(k, 4) for k in kinds], dtype=np.int16)


def kind_compatibility_matrix(
    left_kinds: Sequence[str | None], right_kinds: Sequence[str | None]
) -> np.ndarray:
    lk = _kind_ids(left_kinds)
    rk = _kind_ids(right_kinds)

    n, m = lk.shape[0], rk.shape[0]
    out = np.zeros((n, m), dtype=np.float32)

    left_valid = lk != 0
    right_valid = rk != 0

    same = (lk[:, None] == rk[None, :]) & left_valid[:, None] & right_valid[None, :]
    out[same & (lk[:, None] == 1)] = 1.0
    out[same & (lk[:, None] != 1)] = 0.6

    pb_left = np.isin(lk, [2, 3])[:, None]
    pb_right = np.isin(rk, [2, 3])[None, :]
    pb_pair = pb_left & pb_right & (~same)
    out[pb_pair] = 0.35

    other_valid = left_valid[:, None] & right_valid[None, :]
    other = other_valid & (~same) & (~pb_pair)
    out[other] = -0.5

    return out


def token_balance_bonus_matrix(
    left_token_counts: Sequence[int | None], right_token_counts: Sequence[int | None]
) -> np.ndarray:
    a = np.asarray(
        [int(x) if x is not None else 0 for x in left_token_counts], dtype=np.float32
    )
    b = np.asarray(
        [int(x) if x is not None else 0 for x in right_token_counts], dtype=np.float32
    )

    valid = (a[:, None] > 0) & (b[None, :] > 0)
    minv = np.minimum(a[:, None], b[None, :])
    maxv = np.maximum(a[:, None], b[None, :])
    ratio = np.divide(minv, maxv, out=np.zeros_like(minv), where=maxv > 0)
    return np.where(valid, 0.05 * ratio, 0.0).astype(np.float32, copy=False)


def page_proximity_bonus_matrix(
    left_pages: Sequence[int | None], right_pages: Sequence[int | None]
) -> np.ndarray:
    a = np.asarray(
        [int(x) if x is not None else -1 for x in left_pages], dtype=np.int32
    )
    b = np.asarray(
        [int(x) if x is not None else -1 for x in right_pages], dtype=np.int32
    )

    valid = (a[:, None] >= 0) & (b[None, :] >= 0)
    dist = np.abs(a[:, None] - b[None, :])

    out = np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    out[valid & (dist == 0)] = 0.03
    out[valid & (dist == 1)] = 0.02
    out[valid & (dist <= 3) & (dist > 1)] = 0.01
    return out


# ---------------------------------------------------------------------------
# Core hybrid score matrix
# ---------------------------------------------------------------------------


def build_hybrid_score_matrix(
    left_texts: Sequence[str],
    right_texts: Sequence[str],
    left_embeddings: np.ndarray | "torch.Tensor",
    right_embeddings: np.ndarray | "torch.Tensor",
    left_paths: Sequence[list[str] | None],
    right_paths: Sequence[list[str] | None],
    left_pages: Sequence[int | None],
    right_pages: Sequence[int | None],
    left_kinds: Sequence[str | None],
    right_kinds: Sequence[str | None],
    left_token_counts: Sequence[int | None],
    right_token_counts: Sequence[int | None],
    *,
    use_gpu: str = "auto",
    max_workers: int | None = None,
) -> np.ndarray:
    """
    Build the same hybrid score logic, but in bulk.

    The expensive pieces are computed once as matrices:
    - cosine on GPU (if available)
    - lexical similarity with multiprocessing
    - heading similarity with multiprocessing
    - page / token / kind bonuses via NumPy broadcasting
    """
    n = len(left_texts)
    m = len(right_texts)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    cos = cosine_matrix(
        left_embeddings,
        right_embeddings,
        normalize=True,
        use_gpu=use_gpu,
        return_numpy=True,
    )
    lex = lexical_matrix(left_texts, right_texts, max_workers=max_workers)
    head = heading_similarity_matrix(left_paths, right_paths, max_workers=max_workers)

    page_bonus = page_proximity_bonus_matrix(left_pages, right_pages)
    size_bonus = token_balance_bonus_matrix(left_token_counts, right_token_counts)
    kind_bonus = kind_compatibility_matrix(left_kinds, right_kinds)

    left_kind_ids = _kind_ids(left_kinds)
    right_kind_ids = _kind_ids(right_kinds)

    section_mask = (left_kind_ids[:, None] == 1) & (right_kind_ids[None, :] == 1)
    pb_mask = (
        np.isin(left_kind_ids, [2, 3])[:, None]
        & np.isin(right_kind_ids, [2, 3])[None, :]
    )
    pb_mask = pb_mask & (~section_mask)

    default_mask = ~(section_mask | pb_mask)

    cos_coef = np.full((n, m), 0.80, dtype=np.float32)
    cos_coef[section_mask] = 0.74
    cos_coef[pb_mask] = 0.82

    head_coef = np.ones((n, m), dtype=np.float32)
    head_coef[pb_mask] = 0.5

    score = np.zeros((n, m), dtype=np.float32)
    score += cos_coef * cos
    score += 0.10 * lex
    score += head_coef * (0.08 * head)
    score += page_bonus
    score += size_bonus

    # Same-kind / compatibility bonus is only used outside section-section.
    score += np.where(default_mask | pb_mask, kind_bonus, 0.0).astype(
        np.float32, copy=False
    )
    score += np.where(section_mask, 0.05, 0.0).astype(np.float32, copy=False)

    return score


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------


def hungarian_match(
    score_matrix: np.ndarray, *, maximize: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if linear_sum_assignment is None:
        raise ImportError(
            "scipy is required for Hungarian matching (scipy.optimize.linear_sum_assignment)."
        )
    if maximize:
        row_ind, col_ind = linear_sum_assignment(-score_matrix)
    else:
        row_ind, col_ind = linear_sum_assignment(score_matrix)
    return row_ind, col_ind, score_matrix[row_ind, col_ind]


# ---------------------------------------------------------------------------
# Compatibility helpers preserved from the original module
# ---------------------------------------------------------------------------


def hybrid_score(
    cosine: float,
    lexical: float,
    path_a: list[str] | None,
    path_b: list[str] | None,
    page_a: int | None,
    page_b: int | None,
    *,
    kind_a: str | None = None,
    kind_b: str | None = None,
    token_count_a: int | None = None,
    token_count_b: int | None = None,
) -> tuple[float, float, float]:
    same_heading = heading_similarity(path_a, path_b)
    heading_bonus = 0.08 * same_heading

    kind_bonus = kind_compatibility(kind_a, kind_b)
    size_bonus = token_balance_bonus(token_count_a, token_count_b)
    page_bonus = page_proximity_bonus(page_a, page_b)

    if kind_a == kind_b == "section":
        score = (
            0.74 * cosine
            + 0.10 * lexical
            + heading_bonus
            + page_bonus
            + 0.05
            + size_bonus
        )
    elif {kind_a, kind_b} <= {"paragraph", "bullet"}:
        score = (
            0.82 * cosine
            + 0.10 * lexical
            + 0.5 * heading_bonus
            + page_bonus
            + kind_bonus
            + size_bonus
        )
    else:
        score = (
            0.80 * cosine
            + 0.10 * lexical
            + heading_bonus
            + page_bonus
            + kind_bonus
            + size_bonus
        )

    return float(score), float(heading_bonus), float(page_bonus)


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
