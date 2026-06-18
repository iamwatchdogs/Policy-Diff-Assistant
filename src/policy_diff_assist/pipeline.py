from __future__ import annotations

import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal

import numpy as np
from loguru import logger

from policy_diff_assist.alignment import align_trees
from policy_diff_assist.config import AppConfig, build_default_session_dir
from policy_diff_assist.embeddings import embed_texts, load_embedding_backend
from policy_diff_assist.ingestion import (
    build_tree,
    iter_leaf_nodes,
    normalize_text,
    write_tree,
)
from policy_diff_assist.llm import load_llm_backend, stream_summary
from policy_diff_assist.models import ComparisonResult, ProgressState
from policy_diff_assist.provenance import build_context_pack
from policy_diff_assist.reporting import build_report_artifact


ProgressCallback = Callable[[ProgressState], None]


@dataclass(slots=True)
class StreamEvent:
    status: str
    summary: str
    report_path: str | None = None
    progress: ProgressState | None = None
    result: ComparisonResult | None = None


def _status_text(state: ProgressState) -> str:
    return f"**{state.stage}** — {state.percent}% — {state.message}"


def _emit(cb: ProgressCallback | None, **kwargs) -> ProgressState:
    state = ProgressState(**kwargs)
    if cb is not None:
        cb(state)
    return state


def _render_summary(summary_parts: list[str], live_item: str | None = None) -> str:
    parts = list(summary_parts)
    if live_item:
        parts.append(live_item)
    return "\n".join(parts).strip()


def compare_documents_stream(
    legacy_pdf: str | Path,
    modern_pdf: str | Path,
    output_root: str | Path | None = None,
    cfg: AppConfig | None = None,
    progress_cb: ProgressCallback | None = None,
) -> Iterator[StreamEvent]:
    cfg = cfg or AppConfig.load()
    session_id = uuid.uuid4().hex
    session_dir = (
        build_default_session_dir(cfg, session_id)
        if output_root is None
        else Path(output_root) / session_id
    )
    session_dir.mkdir(parents=True, exist_ok=True)

    legacy_pdf = Path(legacy_pdf)
    modern_pdf = Path(modern_pdf)

    logger.info("Session {} initiated", session_id)

    state = _emit(
        progress_cb,
        stage="starting",
        percent=1,
        message="Starting comparison",
        session_id=session_id,
    )
    yield StreamEvent(status=_status_text(state), summary="", progress=state)

    legacy_work = session_dir / "legacy"
    modern_work = session_dir / "modern"
    legacy_work.mkdir(parents=True, exist_ok=True)
    modern_work.mkdir(parents=True, exist_ok=True)

    shutil.copy2(legacy_pdf, legacy_work / legacy_pdf.name)
    shutil.copy2(modern_pdf, modern_work / modern_pdf.name)

    state = _emit(
        progress_cb,
        stage="parsing",
        percent=8,
        message="Parsing PDF files",
        session_id=session_id,
    )
    yield StreamEvent(status=_status_text(state), summary="", progress=state)

    def docs_to_tree(
        doc_path: str | Path,
        doc_type: Literal["legacy", "modern"],
        config: AppConfig,
    ):
        tree_content = build_tree(doc_path, doc_type, config)
        write_tree(tree_content, session_dir / f"{doc_type}.msgspec.json")
        return tree_content

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_legacy = executor.submit(
            docs_to_tree, legacy_work / legacy_pdf.name, "legacy", cfg
        )
        future_modern = executor.submit(
            docs_to_tree, modern_work / modern_pdf.name, "modern", cfg
        )
        legacy_tree = future_legacy.result()
        modern_tree = future_modern.result()

    state = _emit(
        progress_cb,
        stage="embedding",
        percent=25,
        message="Embedding leaf segments",
        session_id=session_id,
    )
    yield StreamEvent(status=_status_text(state), summary="", progress=state)

    legacy_texts = [normalize_text(n.text) for n in iter_leaf_nodes(legacy_tree)]
    modern_texts = [normalize_text(n.text) for n in iter_leaf_nodes(modern_tree)]

    emb_backend = load_embedding_backend(
        cfg.embedding_model_name,
        cfg.fallback_embedding_model_name,
        cfg.hf_token,
    )

    embedding_batch_size = max(int(getattr(cfg, "batch_size", 64)), 1024)
    legacy_emb = embed_texts(emb_backend, legacy_texts, batch_size=embedding_batch_size)
    modern_emb = embed_texts(emb_backend, modern_texts, batch_size=embedding_batch_size)

    np.save(session_dir / "legacy_embeddings.npy", legacy_emb)
    np.save(session_dir / "modern_embeddings.npy", modern_emb)

    state = _emit(
        progress_cb,
        stage="matching",
        percent=55,
        message="Running cosine + Hungarian alignment",
        session_id=session_id,
    )
    yield StreamEvent(status=_status_text(state), summary="", progress=state)

    aligned = align_trees(legacy_tree, modern_tree, legacy_emb, modern_emb, cfg)

    state = _emit(
        progress_cb,
        stage="summarizing",
        percent=72,
        message="Building provenance-rich context packs",
        session_id=session_id,
    )
    yield StreamEvent(status=_status_text(state), summary="", progress=state)

    llm_backend = load_llm_backend(cfg)

    summary_parts: list[str] = []
    max_items = max(len(aligned.matches), 1)

    for idx, match in enumerate(aligned.matches, start=1):
        if match.change_type == "unchanged":
            continue

        ctx = build_context_pack(
            match, legacy_tree, modern_tree, window=cfg.neighbors_window
        )

        state = _emit(
            progress_cb,
            stage="llm",
            percent=min(92, 72 + int(18 * idx / max_items)),
            message=f"Summarizing change {idx}/{max_items}",
            completed=idx,
            total=max_items,
            session_id=session_id,
            detail=match.change_type,
        )

        live_label = (
            f"- **{match.change_type.title()}** `{match.legacy_id or match.modern_id}`:"
        )
        live_text = ""
        for chunk in stream_summary(ctx, llm_backend, cfg):
            live_text = chunk
            yield StreamEvent(
                status=_status_text(state),
                summary=_render_summary(
                    summary_parts, f"{live_label} {live_text}".strip()
                ),
                progress=state,
            )

        if live_text:
            summary_parts.append(f"{live_label} {live_text}".strip())
            match.evidence_ids = ctx.get("evidence_ids", match.evidence_ids)

    summary_text = (
        "\n".join(summary_parts).strip()
        if summary_parts
        else "No material changes detected."
    )

    result = ComparisonResult(
        session_id=session_id,
        legacy_tree=legacy_tree,
        modern_tree=modern_tree,
        matches=aligned.matches,
        summary=summary_text,
    )

    state = _emit(
        progress_cb,
        stage="reporting",
        percent=95,
        message="Rendering report artifacts",
        session_id=session_id,
    )
    yield StreamEvent(status=_status_text(state), summary=summary_text, progress=state)

    artifact = build_report_artifact(result, session_dir / "report")
    result.report_md_path = artifact.report_md_path
    result.report_pdf_path = artifact.report_pdf_path
    result.report_json_path = artifact.report_json_path

    state = _emit(
        progress_cb,
        stage="done",
        percent=100,
        message="Comparison complete",
        session_id=session_id,
    )
    yield StreamEvent(
        status=_status_text(state),
        summary=summary_text,
        report_path=result.report_pdf_path,
        progress=state,
        result=result,
    )


def compare_documents(
    legacy_pdf: str | Path,
    modern_pdf: str | Path,
    output_root: str | Path | None = None,
    cfg: AppConfig | None = None,
    progress_cb: ProgressCallback | None = None,
) -> ComparisonResult:
    final_result: ComparisonResult | None = None
    for event in compare_documents_stream(
        legacy_pdf,
        modern_pdf,
        output_root=output_root,
        cfg=cfg,
        progress_cb=progress_cb,
    ):
        if event.result is not None:
            final_result = event.result
    if final_result is None:
        raise RuntimeError("Comparison did not produce a final result.")
    return final_result
