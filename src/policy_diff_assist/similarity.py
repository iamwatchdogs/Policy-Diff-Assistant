from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np
from loguru import logger

def cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    logger.info("Computing cosine simiarities")
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


def page_proximity_bonus(page_a: int | None, page_b: int | None) -> float:
    logger.info("Computing page proximit bonus.")
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
) -> tuple[float, float, float]:
    same_heading = heading_similarity(path_a, path_b)
    heading_bonus = 0.06 * same_heading
    page_bonus = page_proximity_bonus(page_a, page_b)
    score = 0.84 * cosine + 0.10 * lexical + heading_bonus + page_bonus
    logger.info("Compuated score: {}", score)
    return float(score), float(heading_bonus), float(page_bonus)


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
