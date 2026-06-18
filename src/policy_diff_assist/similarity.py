from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np
from loguru import logger


def cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    logger.info("Computing cosine similarities")
    if left.size == 0 or right.size == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    return left @ right.T


def lexical_score(a: str, b: str) -> float:
    logger.info("Computing lexical scoring")
    return float(SequenceMatcher(None, a.lower(), b.lower()).ratio())


def heading_similarity(path_a: list[str] | None, path_b: list[str] | None) -> float:
    logger.info("Computing heading similarities")
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


def kind_compatibility(kind_a: str | None, kind_b: str | None) -> float:
    """
    Encourage section-section matches first, then paragraph/bullet matches.
    """
    if kind_a is None or kind_b is None:
        return 0.0
    if kind_a == kind_b:
        if kind_a == "section":
            return 1.0
        return 0.6
    pair = {kind_a, kind_b}
    if pair <= {"paragraph", "bullet"}:
        return 0.35
    return -0.5


def token_balance_bonus(token_count_a: int | None, token_count_b: int | None) -> float:
    if token_count_a is None or token_count_b is None:
        return 0.0
    a = max(int(token_count_a), 1)
    b = max(int(token_count_b), 1)
    ratio = min(a, b) / max(a, b)
    return 0.05 * ratio


def page_proximity_bonus(page_a: int | None, page_b: int | None) -> float:
    logger.info("Computing page proximity bonus")
    if page_a is None or page_b is None:
        return 0.0
    dist = abs(page_a - page_b)
    if dist == 0:
        return 0.03
    if dist == 1:
        return 0.02
    if dist <= 3:
        return 0.01
    return 0.0


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
    """
    Tree-aware hybrid score.

    The section-first tree should rely on:
    - embedding similarity
    - heading/path agreement
    - node kind agreement
    - rough size balance
    - page proximity
    """
    same_heading = heading_similarity(path_a, path_b)
    heading_bonus = 0.08 * same_heading

    kind_bonus = kind_compatibility(kind_a, kind_b)
    size_bonus = token_balance_bonus(token_count_a, token_count_b)
    page_bonus = page_proximity_bonus(page_a, page_b)

    # Stronger structural prior for section matches, lighter for body nodes.
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

    logger.info("Computed hybrid score: {}", score)
    return float(score), float(heading_bonus), float(page_bonus)


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
