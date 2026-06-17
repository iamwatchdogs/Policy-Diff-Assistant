from __future__ import annotations

from typing import Any

import msgspec


class SourceNode(msgspec.Struct, omit_defaults=True):
    node_id: str
    doc_side: str  # "legacy" or "modern"
    page: int
    kind: str  # document/page/section/paragraph/clause/bullet
    path: list[str]
    text: str
    parent_id: str | None = None
    children: list[str] = msgspec.field(default_factory=list)
    start_char: int = 0
    end_char: int = 0
    bbox: tuple[float, float, float, float] | None = None
    token_count: int = 0
    stable_hash: str = ""


class DocumentTree(msgspec.Struct, omit_defaults=True):
    doc_id: str
    doc_side: str  # "legacy" or "modern"
    source_path: str
    page_count: int
    created_at: str
    nodes: dict[str, SourceNode] = msgspec.field(default_factory=dict)
    leaf_ids: list[str] = msgspec.field(default_factory=list)
    leaf_positions: dict[str, int] = msgspec.field(default_factory=dict)


class MatchRecord(msgspec.Struct, omit_defaults=True):
    legacy_id: str | None = None
    modern_id: str | None = None
    legacy_path: list[str] | None = None
    modern_path: list[str] | None = None
    legacy_page: int | None = None
    modern_page: int | None = None
    similarity: float = 0.0
    lexical_score: float = 0.0
    change_type: str = (
        "modified"  # unchanged / modified / added / removed / split / merged
    )
    evidence_ids: list[str] = msgspec.field(default_factory=list)
    legacy_text: str | None = None
    modern_text: str | None = None
    legacy_span: tuple[int, int] | None = None
    modern_span: tuple[int, int] | None = None
    heading_bonus: float = 0.0
    page_bonus: float = 0.0


class ReportArtifact(msgspec.Struct, omit_defaults=True):
    session_id: str
    report_md_path: str
    report_pdf_path: str
    report_json_path: str
    summary: str
    match_count: int
    removed_count: int
    added_count: int
    modified_count: int
    unchanged_count: int


class ProgressState(msgspec.Struct, omit_defaults=True):
    stage: str
    percent: int
    message: str
    completed: int = 0
    total: int = 0
    session_id: str = ""
    detail: str = ""


class ComparisonResult(msgspec.Struct, omit_defaults=True):
    session_id: str
    legacy_tree: DocumentTree
    modern_tree: DocumentTree
    matches: list[MatchRecord] = msgspec.field(default_factory=list)
    summary: str = ""
    report_md_path: str = ""
    report_pdf_path: str = ""
    report_json_path: str = ""
    progress: ProgressState | None = None
    extra: dict[str, Any] = msgspec.field(default_factory=dict)
