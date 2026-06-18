from __future__ import annotations

from typing import Any

from loguru import logger

from policy_diff_assist.models import DocumentTree, MatchRecord, SourceNode


def build_parent_map(tree: DocumentTree) -> dict[str, str | None]:
    return {node_id: node.parent_id for node_id, node in tree.nodes.items()}


def build_children_map(tree: DocumentTree) -> dict[str, list[str]]:
    return {node_id: node.children[:] for node_id, node in tree.nodes.items()}


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


def _ancestor_chain(tree: DocumentTree, node: SourceNode) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    cur = node
    while cur.parent_id and cur.parent_id in tree.nodes and cur.parent_id not in seen:
        seen.add(cur.parent_id)
        parent = tree.nodes[cur.parent_id]
        chain.append(_node_to_dict(parent))
        cur = parent
    return chain


def trace_node(tree: DocumentTree, node_id: str) -> dict[str, Any] | None:
    node = tree.nodes.get(node_id)
    if node is None:
        return None

    return {
        "node": _node_to_dict(node),
        "parents": _ancestor_chain(tree, node),
        "children": [
            _node_to_dict(tree.nodes[c]) for c in node.children if c in tree.nodes
        ],
    }


def _section_ancestor_id(tree: DocumentTree, node_id: str | None) -> str | None:
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


def _section_children(
    tree: DocumentTree, section_id: str | None, limit: int = 8
) -> list[dict[str, Any]]:
    if not section_id or section_id not in tree.nodes:
        return []
    section = tree.nodes[section_id]
    out: list[dict[str, Any]] = []
    for child_id in section.children[:limit]:
        if child_id in tree.nodes:
            out.append(_node_to_dict(tree.nodes[child_id]))
    return out


def _section_context(
    tree: DocumentTree, section_id: str | None, window: int = 1
) -> dict[str, Any] | None:
    if not section_id or section_id not in tree.nodes:
        return None
    section = tree.nodes[section_id]
    return {
        "section": _node_to_dict(section),
        "ancestors": _ancestor_chain(tree, section),
        "children": _section_children(tree, section_id, limit=12),
        "neighbors": [
            _node_to_dict(tree.nodes[nid])
            for nid in _neighbor_ids(tree, section_id, window)
            if nid in tree.nodes
        ],
    }


def build_context_pack(
    match: MatchRecord,
    legacy_tree: DocumentTree,
    modern_tree: DocumentTree,
    window: int = 1,
) -> dict[str, Any]:
    legacy_node = legacy_tree.nodes.get(match.legacy_id) if match.legacy_id else None
    modern_node = modern_tree.nodes.get(match.modern_id) if match.modern_id else None

    legacy_section_id = _section_ancestor_id(legacy_tree, match.legacy_id)
    modern_section_id = _section_ancestor_id(modern_tree, match.modern_id)

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

    legacy_section_ctx = _section_context(legacy_tree, legacy_section_id, window=window)
    modern_section_ctx = _section_context(modern_tree, modern_section_id, window=window)

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
        "legacy_section_id": legacy_section_id,
        "modern_section_id": modern_section_id,
        "legacy_section": legacy_section_ctx,
        "modern_section": modern_section_ctx,
        "legacy_neighbors": legacy_neighbors,
        "modern_neighbors": modern_neighbors,
        "evidence_ids": match.evidence_ids[:],
    }


def expand_context_for_section(
    tree: DocumentTree, node_id: str, window: int = 2
) -> list[dict[str, Any]]:
    if node_id not in tree.nodes:
        return []

    node = tree.nodes[node_id]
    out: list[dict[str, Any]] = []

    # Always include the selected node.
    out.append(_node_to_dict(node))

    # If it's a section, include its immediate children and local leaf neighbors.
    if node.kind == "section":
        out.extend(_section_children(tree, node_id, limit=20))
        for nid in _neighbor_ids(tree, node_id, window):
            if nid in tree.nodes:
                out.append(_node_to_dict(tree.nodes[nid]))
        return out

    # For body nodes, include the parent section and nearby leaf nodes.
    section_id = _section_ancestor_id(tree, node_id)
    if section_id and section_id in tree.nodes:
        out.insert(0, _node_to_dict(tree.nodes[section_id]))
        out[1:1] = _section_children(tree, section_id, limit=10)

    for nid in _neighbor_ids(tree, node_id, window):
        if nid in tree.nodes:
            out.append(_node_to_dict(tree.nodes[nid]))

    return out
