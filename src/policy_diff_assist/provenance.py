from __future__ import annotations

from typing import Any
from loguru import logger

from policy_diff_assist.models import DocumentTree, MatchRecord, SourceNode


def build_parent_map(tree: DocumentTree) -> dict[str, str | None]:
    return {node_id: node.parent_id for node_id, node in tree.nodes.items()}


def build_children_map(tree: DocumentTree) -> dict[str, list[str]]:
    return {node_id: node.children[:] for node_id, node in tree.nodes.items()}


def trace_node(tree: DocumentTree, node_id: str) -> dict[str, Any] | None:
    node = tree.nodes.get(node_id)
    if node is None:
        return None

    parents: list[dict[str, Any]] = []
    cur = node
    seen = set()
    while cur.parent_id and cur.parent_id in tree.nodes and cur.parent_id not in seen:
        seen.add(cur.parent_id)
        parent = tree.nodes[cur.parent_id]
        parents.append(_node_to_dict(parent))
        cur = parent

    return {
        "node": _node_to_dict(node),
        "parents": parents,
        "children": [
            _node_to_dict(tree.nodes[c]) for c in node.children if c in tree.nodes
        ],
    }


def _node_to_dict(node: SourceNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "doc_side": node.doc_side,
        "page": node.page,
        "kind": node.kind,
        "path": node.path[:],
        "text": node.text,
        "parent_id": node.parent_id,
        "children": node.children[:],
        "start_char": node.start_char,
        "end_char": node.end_char,
        "bbox": node.bbox,
        "token_count": node.token_count,
        "stable_hash": node.stable_hash,
    }


def _neighbor_ids(tree: DocumentTree, node_id: str, window: int = 1) -> list[str]:
    idx = tree.leaf_positions.get(node_id)
    if idx is None:
        return []
    ids: list[str] = []
    start = max(0, idx - window)
    end = min(len(tree.leaf_ids), idx + window + 1)
    for j in range(start, end):
        if j == idx:
            continue
        ids.append(tree.leaf_ids[j])
    return ids


def build_context_pack(
    match: MatchRecord,
    legacy_tree: DocumentTree,
    modern_tree: DocumentTree,
    window: int = 1,
) -> dict[str, Any]:
    legacy_node = legacy_tree.nodes.get(match.legacy_id) if match.legacy_id else None
    modern_node = modern_tree.nodes.get(match.modern_id) if match.modern_id else None

    legacy_neighbors = (
        [
            _node_to_dict(legacy_tree.nodes[nid])
            for nid in _neighbor_ids(legacy_tree, match.legacy_id or "", window)
            if nid in legacy_tree.nodes
        ]
        if match.legacy_id
        else []
    )
    modern_neighbors = (
        [
            _node_to_dict(modern_tree.nodes[nid])
            for nid in _neighbor_ids(modern_tree, match.modern_id or "", window)
            if nid in modern_tree.nodes
        ]
        if match.modern_id
        else []
    )

    logger.info("Built trace back context for LLM processing.")

    return {
        "match_id": f"{match.legacy_id or 'none'}::{match.modern_id or 'none'}",
        "change_type": match.change_type,
        "stats": {
            "cosine": match.similarity,
            "lexical": match.lexical_score,
            "heading_bonus": match.heading_bonus,
            "page_bonus": match.page_bonus,
        },
        "legacy": _node_to_dict(legacy_node) if legacy_node else None,
        "modern": _node_to_dict(modern_node) if modern_node else None,
        "legacy_neighbors": legacy_neighbors,
        "modern_neighbors": modern_neighbors,
        "evidence_ids": match.evidence_ids[:],
    }


def expand_context_for_section(
    tree: DocumentTree, node_id: str, window: int = 2
) -> list[dict[str, Any]]:
    if node_id not in tree.nodes:
        return []
    idx = tree.leaf_positions.get(node_id)
    if idx is None:
        return []
    out: list[dict[str, Any]] = []
    start = max(0, idx - window)
    end = min(len(tree.leaf_ids), idx + window + 1)
    for j in range(start, end):
        nid = tree.leaf_ids[j]
        node = tree.nodes.get(nid)
        if node is not None:
            out.append(_node_to_dict(node))
    return out
