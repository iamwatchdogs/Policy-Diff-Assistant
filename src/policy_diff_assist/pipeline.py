from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Callable, Iterator, Literal
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from loguru import logger

from policy_diff_assist.alignment import align_trees
from policy_diff_assist.config import AppConfig, build_default_session_dir
from policy_diff_assist.embeddings import embed_two_corpora, load_embedding_backend
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


def _emit(cb: ProgressCallback | None, **kwargs) -> ProgressState:
    logger.info("Updated Progress in UI")
    state = ProgressState(**kwargs)
    if cb is not None:
        cb(state)
    return state


def compare_documents(
    legacy_pdf: str | Path,
    modern_pdf: str | Path,
    output_root: str | Path | None = None,
    cfg: AppConfig | None = None,
    progress_cb: ProgressCallback | None = None,
) -> ComparisonResult:
    cfg = cfg or AppConfig.load()
    session_id = uuid.uuid4().hex
    session_dir = (
        build_default_session_dir(cfg, session_id)
        if output_root is None
        else Path(output_root) / session_id
    )
    session_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Session {} Iniitated", session_id)

    legacy_pdf = Path(legacy_pdf)
    modern_pdf = Path(modern_pdf)

    _emit(
        progress_cb,
        stage="starting",
        percent=1,
        message="Starting comparison",
        session_id=session_id,
    )

    legacy_work = session_dir / "legacy"
    modern_work = session_dir / "modern"
    legacy_work.mkdir(parents=True, exist_ok=True)
    modern_work.mkdir(parents=True, exist_ok=True)

    shutil.copy2(legacy_pdf, legacy_work / legacy_pdf.name)
    shutil.copy2(modern_pdf, modern_work / modern_pdf.name)

    logger.info("Saved PDF files to temp session location.")

    _emit(
        progress_cb,
        stage="parsing",
        percent=8,
        message="Parsing PDF files",
        session_id=session_id,
    )

    def docs_to_json(
        doc_path: str | Path, doc_type: Literal["legacy", "modern"], config: AppConfig
    ) -> None:
        tree_content = build_tree(doc_path, doc_type, config)
        write_tree(tree_content, session_dir / f"{doc_type}.msgspec.json")
        return tree_content

    with ThreadPoolExecutor(max_workers=2) as executor:
        thread_legacy = executor.submit(
            docs_to_json, legacy_work / legacy_pdf.name, "legacy", cfg
        )
        thread_modern = executor.submit(
            docs_to_json, modern_work / modern_pdf.name, "modern", cfg
        )
        legacy_tree = thread_legacy.result()
        modern_tree = thread_modern.result()

    logger.info("Saved both Index trees in temp session location.")

    _emit(
        progress_cb,
        stage="embedding",
        percent=25,
        message="Embedding leaf segments",
        session_id=session_id,
    )

    legacy_texts = [normalize_text(n.text) for n in iter_leaf_nodes(legacy_tree)]
    modern_texts = [normalize_text(n.text) for n in iter_leaf_nodes(modern_tree)]

    logger.info("Normalized Text within the Index Trees.")

    emb_backend = load_embedding_backend(
        cfg.embedding_model_name, cfg.fallback_embedding_model_name, cfg.hf_token
    )

    embedding_batch_size = max(int(getattr(cfg, "batch_size", 64)), 128)

    legacy_emb, modern_emb = embed_two_corpora(
        emb_backend,
        legacy_texts,
        modern_texts,
        batch_size=embedding_batch_size,
    )

    np.save(session_dir / "legacy_embeddings.npy", legacy_emb)
    np.save(session_dir / "modern_embeddings.npy", modern_emb)

    logger.info("Saved embedding in temp session path.")

    _emit(
        progress_cb,
        stage="matching",
        percent=55,
        message="Running cosine + Hungarian alignment",
        session_id=session_id,
    )

    aligned = align_trees(legacy_tree, modern_tree, legacy_emb, modern_emb, cfg)

    _emit(
        progress_cb,
        stage="summarizing",
        percent=72,
        message="Building provenance-rich context packs",
        session_id=session_id,
    )

    llm_backend = load_llm_backend(cfg)

    summaries: list[str] = []
    max_items = len(aligned.matches)
    for idx, match in enumerate(aligned.matches, start=1):
        if match.change_type == "unchanged":
            continue
        ctx = build_context_pack(
            match, legacy_tree, modern_tree, window=cfg.neighbors_window
        )
        _emit(
            progress_cb,
            stage="llm",
            percent=min(92, 72 + int(18 * idx / max(max_items, 1))),
            message=f"Summarizing change {idx}/{max_items}",
            completed=idx,
            total=max_items,
            session_id=session_id,
            detail=match.change_type,
        )
        summary = ""
        for chunk in stream_summary(ctx, llm_backend, cfg):
            summary = chunk
        if summary:
            summaries.append(
                f"- **{match.change_type.title()}** `{match.legacy_id or match.modern_id}`: {summary}"
            )
            match.evidence_ids = ctx.get("evidence_ids", match.evidence_ids)

    summary_text = (
        "\n".join(summaries) if summaries else "No material changes detected."
    )

    result = ComparisonResult(
        session_id=session_id,
        legacy_tree=legacy_tree,
        modern_tree=modern_tree,
        matches=aligned.matches,
        summary=summary_text,
    )

    _emit(
        progress_cb,
        stage="reporting",
        percent=95,
        message="Rendering report artifacts",
        session_id=session_id,
    )

    artifact = build_report_artifact(result, session_dir / "report")
    result.report_md_path = artifact.report_md_path
    result.report_pdf_path = artifact.report_pdf_path
    result.report_json_path = artifact.report_json_path

    _emit(
        progress_cb,
        stage="done",
        percent=100,
        message="Comparison complete",
        session_id=session_id,
    )

    logger.info("Document Comparison Completed!")
    return result


def compare_documents_stream(
    legacy_pdf: str | Path,
    modern_pdf: str | Path,
    output_root: str | Path | None = None,
    cfg: AppConfig | None = None,
) -> Iterator[tuple[ProgressState, ComparisonResult | None]]:
    latest_state: ProgressState | None = None

    def cb(state: ProgressState) -> None:
        nonlocal latest_state
        latest_state = state

    result = compare_documents(
        legacy_pdf, modern_pdf, output_root=output_root, cfg=cfg, progress_cb=cb
    )
    if latest_state is not None:
        yield latest_state, None
    yield (
        ProgressState(
            stage="done",
            percent=100,
            message="Comparison complete",
            session_id=result.session_id,
        ),
        result,
    )
