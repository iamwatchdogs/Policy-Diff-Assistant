from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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


def _node_index_map(tree: DocumentTree) -> dict[str, int]:
    return {
        nid: i
        for i, nid in enumerate(tree.leaf_ids)
        if nid in tree.nodes and tree.nodes[nid].kind != "document"
    }


def _section_nodes(tree: DocumentTree) -> list:
    nodes = tree.nodes
    return [
        nodes[nid]
        for nid in tree.leaf_ids
        if nid in nodes and nodes[nid].kind == "section"
    ]


def _body_nodes(tree: DocumentTree) -> list:
    nodes = tree.nodes
    return [
        nodes[nid]
        for nid in tree.leaf_ids
        if nid in nodes and nodes[nid].kind not in {"document", "section"}
    ]


def _section_id_for_node(tree: DocumentTree, node_id: str | None) -> str | None:
    if not node_id or node_id not in tree.nodes:
        return None
    node = tree.nodes[node_id]
    if node.kind == "section":
        return node.node_id
    if node.parent_id and node.parent_id in tree.nodes:
        parent = tree.nodes[node.parent_id]
        if parent.kind == "section":
            return parent.node_id
    return None


def _children_for_section(tree: DocumentTree, section_id: str | None) -> list:
    if not section_id or section_id not in tree.nodes:
        return []
    sec = tree.nodes[section_id]
    return [
        tree.nodes[cid]
        for cid in sec.children
        if cid in tree.nodes and tree.nodes[cid].kind not in {"document", "section"}
    ]


def _candidate_mask(a, b) -> bool:
    if a.kind == "section" or b.kind == "section":
        return a.kind == b.kind == "section"
    if a.kind == b.kind:
        return True
    if {a.kind, b.kind} <= {"paragraph", "bullet"}:
        return True
    return False


def _score_matrix(
    left_nodes: list,
    right_nodes: list,
    left_emb: np.ndarray,
    right_emb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sim = cosine_matrix(left_emb, right_emb)
    candidate = np.zeros_like(sim, dtype=np.float32)

    for i, ln in enumerate(left_nodes):
        for j, rn in enumerate(right_nodes):
            if not _candidate_mask(ln, rn):
                continue

            lex = lexical_score(ln.text, rn.text)
            score, _, _ = hybrid_score(
                float(sim[i, j]),
                lex,
                ln.path,
                rn.path,
                ln.page,
                rn.page,
                kind_a=ln.kind,
                kind_b=rn.kind,
                token_count_a=ln.token_count,
                token_count_b=rn.token_count,
            )
            candidate[i, j] = clip01(score)

    return sim, candidate


def _hungarian_with_unmatched(
    left_nodes, right_nodes, score: np.ndarray, cfg: AppConfig
) -> tuple[list[MatchRecord], set[int], set[int]]:
    logger.info("Running Hungarian alignment on {}x{} candidate matrix", *score.shape)
    n, m = score.shape
    if n == 0 and m == 0:
        return [], set(), set()

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
            score_val, heading_bonus, page_bonus = hybrid_score(
                s,
                lexical,
                ln.path,
                rn.path,
                ln.page,
                rn.page,
                kind_a=ln.kind,
                kind_b=rn.kind,
                token_count_a=ln.token_count,
                token_count_b=rn.token_count,
            )
            change_type = _change_type(score_val, cfg)
            matches.append(
                MatchRecord(
                    legacy_id=ln.node_id,
                    modern_id=rn.node_id,
                    legacy_path=ln.path[:],
                    modern_path=rn.path[:],
                    legacy_page=ln.page,
                    modern_page=rn.page,
                    similarity=float(score_val),
                    lexical_score=float(lexical),
                    change_type=change_type,
                    evidence_ids=[ln.node_id, rn.node_id],
                    legacy_text=ln.text,
                    modern_text=rn.text,
                    legacy_span=(ln.start_char, ln.end_char),
                    modern_span=(rn.start_char, rn.end_char),
                    heading_bonus=float(heading_bonus),
                    page_bonus=float(page_bonus),
                )
            )

    def sort_key(m: MatchRecord):
        kind_rank = {
            "unchanged": 0,
            "modified": 1,
            "split": 2,
            "merged": 3,
            "removed": 4,
            "added": 5,
        }
        return (
            m.legacy_page if m.legacy_page is not None else 10**9,
            m.modern_page if m.modern_page is not None else 10**9,
            kind_rank.get(m.change_type, 9),
            -(m.similarity or 0.0),
        )

    matches.sort(key=sort_key)
    return matches, assigned_left, assigned_right


def _emit_unmatched(
    nodes: list,
    change_type: str,
    section_id_lookup: Callable[[str], str | None] | None = None,
    evidence_prefix: list[str] | None = None,
) -> list[MatchRecord]:
    out: list[MatchRecord] = []
    evidence_prefix = evidence_prefix or []
    for node in nodes:
        evidence_ids = evidence_prefix[:] + [node.node_id]
        if section_id_lookup is not None:
            sec_id = section_id_lookup(node.node_id)
            if sec_id and sec_id not in evidence_ids:
                evidence_ids.insert(0, sec_id)

        if change_type == "removed":
            out.append(
                MatchRecord(
                    legacy_id=node.node_id,
                    legacy_path=node.path[:],
                    legacy_page=node.page,
                    similarity=0.0,
                    lexical_score=0.0,
                    change_type="removed",
                    evidence_ids=evidence_ids,
                    legacy_text=node.text,
                    legacy_span=(node.start_char, node.end_char),
                    heading_bonus=0.0,
                    page_bonus=0.0,
                )
            )
        else:
            out.append(
                MatchRecord(
                    modern_id=node.node_id,
                    modern_path=node.path[:],
                    modern_page=node.page,
                    similarity=0.0,
                    lexical_score=0.0,
                    change_type="added",
                    evidence_ids=evidence_ids,
                    modern_text=node.text,
                    modern_span=(node.start_char, node.end_char),
                    heading_bonus=0.0,
                    page_bonus=0.0,
                )
            )
    return out


def _global_rows(index_map: dict[str, int], nodes: list) -> list[int]:
    return [index_map[n.node_id] for n in nodes if n.node_id in index_map]


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

    legacy_index = _node_index_map(legacy_tree)
    modern_index = _node_index_map(modern_tree)

    # Full matrices preserve a complete diagnostic view for downstream consumers.
    full_sim, full_candidate = _score_matrix(
        legacy_nodes,
        modern_nodes,
        legacy_emb[[legacy_index[n.node_id] for n in legacy_nodes]]
        if legacy_nodes
        else np.zeros((0, legacy_emb.shape[1]), dtype=np.float32),
        modern_emb[[modern_index[n.node_id] for n in modern_nodes]]
        if modern_nodes
        else np.zeros((0, modern_emb.shape[1]), dtype=np.float32),
    )

    legacy_sections = _section_nodes(legacy_tree)
    modern_sections = _section_nodes(modern_tree)
    legacy_bodies = _body_nodes(legacy_tree)
    modern_bodies = _body_nodes(modern_tree)

    all_matches: list[MatchRecord] = []

    matched_legacy_section_ids: set[str] = set()
    matched_modern_section_ids: set[str] = set()
    matched_legacy_body_ids: set[str] = set()
    matched_modern_body_ids: set[str] = set()

    section_pair_map: list[tuple[str, str]] = []

    # 1) Section-first alignment.
    if legacy_sections and modern_sections:
        left_rows = _global_rows(legacy_index, legacy_sections)
        right_rows = _global_rows(modern_index, modern_sections)
        section_score = full_candidate[np.ix_(left_rows, right_rows)]

        section_matches, left_assigned, right_assigned = _hungarian_with_unmatched(
            legacy_sections, modern_sections, section_score, cfg
        )
        all_matches.extend(section_matches)

        for i in left_assigned:
            matched_legacy_section_ids.add(legacy_sections[i].node_id)
        for j in right_assigned:
            matched_modern_section_ids.add(modern_sections[j].node_id)

        for m in section_matches:
            if m.legacy_id and m.modern_id:
                section_pair_map.append((m.legacy_id, m.modern_id))

    # 2) Match paragraph/bullet nodes inside matched sections.
    for legacy_section_id, modern_section_id in section_pair_map:
        legacy_children = _children_for_section(legacy_tree, legacy_section_id)
        modern_children = _children_for_section(modern_tree, modern_section_id)

        legacy_children = [n for n in legacy_children if n.kind != "section"]
        modern_children = [n for n in modern_children if n.kind != "section"]

        if not legacy_children and not modern_children:
            continue

        left_rows = _global_rows(legacy_index, legacy_children)
        right_rows = _global_rows(modern_index, modern_children)
        child_score = full_candidate[np.ix_(left_rows, right_rows)]

        body_matches, left_assigned, right_assigned = _hungarian_with_unmatched(
            legacy_children, modern_children, child_score, cfg
        )
        if body_matches:
            all_matches.extend(body_matches)

        for i in left_assigned:
            matched_legacy_body_ids.add(legacy_children[i].node_id)
        for j in right_assigned:
            matched_modern_body_ids.add(modern_children[j].node_id)

        unmatched_left = [
            n for idx, n in enumerate(legacy_children) if idx not in left_assigned
        ]
        unmatched_right = [
            n for idx, n in enumerate(modern_children) if idx not in right_assigned
        ]

        if unmatched_left:
            all_matches.extend(
                _emit_unmatched(
                    unmatched_left,
                    "removed",
                    section_id_lookup=lambda _: legacy_section_id,
                    evidence_prefix=[legacy_section_id],
                )
            )
        if unmatched_right:
            all_matches.extend(
                _emit_unmatched(
                    unmatched_right,
                    "added",
                    section_id_lookup=lambda _: modern_section_id,
                    evidence_prefix=[modern_section_id],
                )
            )

    # 3) Fallback global body alignment for bodies in unmatched sections.
    remaining_legacy_bodies = [
        n
        for n in legacy_bodies
        if n.node_id not in matched_legacy_body_ids
        and _section_id_for_node(legacy_tree, n.node_id)
        not in matched_legacy_section_ids
    ]
    remaining_modern_bodies = [
        n
        for n in modern_bodies
        if n.node_id not in matched_modern_body_ids
        and _section_id_for_node(modern_tree, n.node_id)
        not in matched_modern_section_ids
    ]

    if remaining_legacy_bodies and remaining_modern_bodies:
        left_rows = _global_rows(legacy_index, remaining_legacy_bodies)
        right_rows = _global_rows(modern_index, remaining_modern_bodies)
        fallback_score = full_candidate[np.ix_(left_rows, right_rows)]

        fallback_matches, left_assigned, right_assigned = _hungarian_with_unmatched(
            remaining_legacy_bodies, remaining_modern_bodies, fallback_score, cfg
        )
        if fallback_matches:
            all_matches.extend(fallback_matches)

        for i in left_assigned:
            matched_legacy_body_ids.add(remaining_legacy_bodies[i].node_id)
        for j in right_assigned:
            matched_modern_body_ids.add(remaining_modern_bodies[j].node_id)

    # 4) Emit unmatched sections and any remaining unmatched body nodes.
    unmatched_legacy_sections = [
        n for n in legacy_sections if n.node_id not in matched_legacy_section_ids
    ]
    unmatched_modern_sections = [
        n for n in modern_sections if n.node_id not in matched_modern_section_ids
    ]

    if unmatched_legacy_sections:
        all_matches.extend(
            _emit_unmatched(
                unmatched_legacy_sections,
                "removed",
                section_id_lookup=lambda nid: nid,
            )
        )
    if unmatched_modern_sections:
        all_matches.extend(
            _emit_unmatched(
                unmatched_modern_sections,
                "added",
                section_id_lookup=lambda nid: nid,
            )
        )

    for node in legacy_bodies:
        if node.node_id in matched_legacy_body_ids:
            continue
        sec_id = _section_id_for_node(legacy_tree, node.node_id)
        if sec_id in matched_legacy_section_ids:
            continue
        all_matches.extend(
            _emit_unmatched(
                [node],
                "removed",
                section_id_lookup=lambda _: sec_id,
                evidence_prefix=[sec_id] if sec_id else [],
            )
        )

    for node in modern_bodies:
        if node.node_id in matched_modern_body_ids:
            continue
        sec_id = _section_id_for_node(modern_tree, node.node_id)
        if sec_id in matched_modern_section_ids:
            continue
        all_matches.extend(
            _emit_unmatched(
                [node],
                "added",
                section_id_lookup=lambda _: sec_id,
                evidence_prefix=[sec_id] if sec_id else [],
            )
        )

    def sort_key(m: MatchRecord):
        kind_rank = {
            "unchanged": 0,
            "modified": 1,
            "split": 2,
            "merged": 3,
            "removed": 4,
            "added": 5,
        }
        return (
            m.legacy_page if m.legacy_page is not None else 10**9,
            m.modern_page if m.modern_page is not None else 10**9,
            kind_rank.get(m.change_type, 9),
            -(m.similarity or 0.0),
        )

    all_matches.sort(key=sort_key)

    return AlignmentOutput(
        matches=all_matches,
        sim_matrix=full_sim,
        candidate_matrix=full_candidate,
    )


def _change_type(score: float, cfg: AppConfig) -> str:
    if score >= cfg.unchanged_threshold:
        return "unchanged"
    if score >= cfg.modified_threshold:
        return "modified"
    return "modified"
