from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

from policy_diff_assist.config import AppConfig
from policy_diff_assist.models import DocumentTree, MatchRecord
from policy_diff_assist.similarity import (
    clip01,
    cosine_matrix,
    hybrid_score,
    lexical_score,
)


@dataclass(slots=True)
class AlignmentOutput:
    matches: list[MatchRecord]
    sim_matrix: np.ndarray
    candidate_matrix: np.ndarray


def _leaf_nodes(tree: DocumentTree) -> list:
    nodes = tree.nodes
    return [
        nodes[nid]
        for nid in tree.leaf_ids
        if nid in nodes and nodes[nid].kind != "document"
    ]


def _candidate_mask(left_nodes, right_nodes, i: int, j: int) -> bool:
    a = left_nodes[i]
    b = right_nodes[j]
    if a.kind != b.kind:
        return True
    if a.path and b.path:
        # Allow cross-path matching but prefer same-prefix sections via bonuses.
        return True
    return True


def align_trees(
    legacy_tree: DocumentTree,
    modern_tree: DocumentTree,
    legacy_emb: np.ndarray,
    modern_emb: np.ndarray,
    cfg: AppConfig,
) -> AlignmentOutput:
    legacy_nodes = _leaf_nodes(legacy_tree)
    modern_nodes = _leaf_nodes(modern_tree)

    if not legacy_nodes and not modern_nodes:
        logger.info("No Matches found between legacy and modern.")
        return AlignmentOutput(
            matches=[],
            sim_matrix=np.zeros((0, 0), dtype=np.float32),
            candidate_matrix=np.zeros((0, 0), dtype=np.float32),
        )

    sim = cosine_matrix(legacy_emb, modern_emb)

    # Boost/reduce candidates based on lexical and structural signals.
    candidate = np.zeros_like(sim, dtype=np.float32)
    for i, ln in enumerate(legacy_nodes):
        for j, rn in enumerate(modern_nodes):
            if not _candidate_mask(legacy_nodes, modern_nodes, i, j):
                continue
            lex = lexical_score(ln.text, rn.text)
            score, heading_bonus, page_bonus = hybrid_score(
                float(sim[i, j]), lex, ln.path, rn.path, ln.page, rn.page
            )
            candidate[i, j] = clip01(score)

    matches = _hungarian_with_unmatched(legacy_nodes, modern_nodes, candidate, cfg)

    return AlignmentOutput(matches=matches, sim_matrix=sim, candidate_matrix=candidate)


def _hungarian_with_unmatched(
    left_nodes, right_nodes, score: np.ndarray, cfg: AppConfig
) -> list[MatchRecord]:
    logger.info("Processing unmatched embedding within hungarian matching.")
    n, m = score.shape
    if n == 0 and m == 0:
        return []

    size = n + m
    dummy_penalty = 1.0 - cfg.min_similarity
    cost = np.full((size, size), dummy_penalty, dtype=np.float32)
    if n and m:
        cost[:n, :m] = 1.0 - score

    row_ind, col_ind = linear_sum_assignment(cost)
    assigned_left: set[int] = set()
    assigned_right: set[int] = set()
    matches: list[MatchRecord] = []

    for i, j in zip(row_ind, col_ind):
        if i < n and j < m:
            s = float(score[i, j])
            if s < cfg.min_similarity:
                continue
            assigned_left.add(i)
            assigned_right.add(j)
            ln = left_nodes[i]
            rn = right_nodes[j]
            lexical = lexical_score(ln.text, rn.text)
            change_type = _change_type(s, cfg)
            matches.append(
                MatchRecord(
                    legacy_id=ln.node_id,
                    modern_id=rn.node_id,
                    legacy_path=ln.path[:],
                    modern_path=rn.path[:],
                    legacy_page=ln.page,
                    modern_page=rn.page,
                    similarity=s,
                    lexical_score=lexical,
                    change_type=change_type,
                    evidence_ids=[ln.node_id, rn.node_id],
                    legacy_text=ln.text,
                    modern_text=rn.text,
                    legacy_span=(ln.start_char, ln.end_char),
                    modern_span=(rn.start_char, rn.end_char),
                )
            )

    # Unmatched left => removed, unmatched right => added.
    for idx, node in enumerate(left_nodes):
        if idx not in assigned_left:
            matches.append(
                MatchRecord(
                    legacy_id=node.node_id,
                    legacy_path=node.path[:],
                    legacy_page=node.page,
                    similarity=0.0,
                    lexical_score=0.0,
                    change_type="removed",
                    evidence_ids=[node.node_id],
                    legacy_text=node.text,
                    legacy_span=(node.start_char, node.end_char),
                )
            )
    for idx, node in enumerate(right_nodes):
        if idx not in assigned_right:
            matches.append(
                MatchRecord(
                    modern_id=node.node_id,
                    modern_path=node.path[:],
                    modern_page=node.page,
                    similarity=0.0,
                    lexical_score=0.0,
                    change_type="added",
                    evidence_ids=[node.node_id],
                    modern_text=node.text,
                    modern_span=(node.start_char, node.end_char),
                )
            )

    # Stable order: legacy page then modern page then type
    def sort_key(m: MatchRecord):
        return (
            m.legacy_page if m.legacy_page is not None else 10**9,
            m.modern_page if m.modern_page is not None else 10**9,
            {
                "unchanged": 0,
                "modified": 1,
                "split": 2,
                "merged": 3,
                "removed": 4,
                "added": 5,
            }.get(m.change_type, 9),
            -(m.similarity or 0.0),
        )

    matches.sort(key=sort_key)
    return matches


def _change_type(score: float, cfg: AppConfig) -> str:
    if score >= cfg.unchanged_threshold:
        return "unchanged"
    if score >= cfg.modified_threshold:
        return "modified"
    return "modified"
