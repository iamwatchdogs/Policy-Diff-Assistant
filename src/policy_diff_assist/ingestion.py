from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz
import msgspec

from policy_diff_assist.config import AppConfig
from policy_diff_assist.models import DocumentTree, SourceNode
from policy_diff_assist.logging import get_logger

log = get_logger(__name__)

_HEADING_RE = re.compile(r"^((\d+)(?:\.(\d+))*[.)]?)\s+(.+)$")
_BULLET_RE = re.compile(r"^([•\-*]|\d+[.)])\s+")


@dataclass(slots=True)
class ExtractedBlock:
    text: str
    bbox: tuple[float, float, float, float] | None
    font_size: float
    page: int
    kind: str = "paragraph"
    is_bold: bool = False


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _page_blocks(page: fitz.Page) -> list[ExtractedBlock]:
    """
    Extract layout-aware text blocks from one page.

    This keeps block order, font sizes, and bboxes, which we later use to
    merge wrapped lines into paragraph-like chunks.
    """
    data = page.get_text("dict", sort=True)
    blocks: list[ExtractedBlock] = []

    for block in data.get("blocks", []):
        if block.get("type", 0) != 0:
            continue

        text_lines: list[str] = []
        font_sizes: list[float] = []
        bold_flags: list[bool] = []

        for line in block.get("lines", []):
            line_text_parts: list[str] = []
            for span in line.get("spans", []):
                txt = span.get("text", "")
                if txt:
                    line_text_parts.append(txt)
                    font_sizes.append(float(span.get("size", 0.0)))
                    flags = int(span.get("flags", 0))
                    # PyMuPDF flags vary by font; keep it heuristic.
                    bold_flags.append(bool(flags & 16))
            if line_text_parts:
                text_lines.append("".join(line_text_parts))

        text = normalize_text("\n".join(text_lines))
        if not text:
            continue

        bbox = tuple(float(v) for v in block.get("bbox", (0, 0, 0, 0)))
        blocks.append(
            ExtractedBlock(
                text=text,
                bbox=bbox,  # type: ignore[arg-type]
                font_size=max(font_sizes) if font_sizes else 0.0,
                page=page.number + 1,
                kind="paragraph",
                is_bold=any(bold_flags),
            )
        )

    return blocks


def _is_header_footer_candidate(text: str) -> bool:
    t = normalize_text(text)
    if len(t) > 120:
        return False
    return bool(re.fullmatch(r"[\w\s\-.,:/()|]+", t)) and any(ch.isalpha() for ch in t)


def _dedupe_repeated_blocks(
    blocks_by_page: list[list[ExtractedBlock]],
) -> list[list[ExtractedBlock]]:
    """
    Remove repeated running headers/footers by detecting text that appears
    on a large fraction of pages.
    """
    counts = Counter()
    for page_blocks in blocks_by_page:
        seen = set()
        for blk in page_blocks:
            norm = normalize_text(blk.text).lower()
            if _is_header_footer_candidate(norm):
                seen.add(norm)
        for norm in seen:
            counts[norm] += 1

    page_count = max(len(blocks_by_page), 1)
    repeated = {k for k, v in counts.items() if v >= max(2, int(page_count * 0.35))}

    cleaned: list[list[ExtractedBlock]] = []
    for page_blocks in blocks_by_page:
        out = []
        for blk in page_blocks:
            norm = normalize_text(blk.text).lower()
            if norm in repeated and _is_header_footer_candidate(norm):
                continue
            out.append(blk)
        cleaned.append(out)
    return cleaned


def _bbox_sort_key(block: ExtractedBlock) -> tuple[float, float]:
    if block.bbox is None:
        return (0.0, 0.0)
    return (float(block.bbox[1]), float(block.bbox[0]))


def _union_bbox(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _token_count(text: str) -> int:
    return max(1, len(re.findall(r"\w+", text)))


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _node_id(
    doc_side: str, page: int, kind: str, path: list[str], local_index: int
) -> str:
    path_part = ".".join(path) if path else "root"
    return f"{doc_side}:p{page:02d}:{path_part}:{kind}{local_index}"


def _looks_like_heading_text(text: str) -> bool:
    """
    Semantic heading check on text alone.
    Strict on purpose: false positives destroy the tree.
    """
    t = normalize_text(text)
    if not t:
        return False

    words = re.findall(r"\w+", t)
    if len(words) > 14:
        return False

    if _HEADING_RE.match(t):
        return True

    if len(t) <= 80 and t.isupper() and len(words) <= 10:
        return True

    # Title-like short lines without sentence punctuation.
    if len(t) <= 70 and not re.search(r"[.!?]", t):
        title_like = sum(1 for w in words if w[:1].isupper()) >= max(1, len(words) - 1)
        if title_like:
            return True

    return False


def _is_heading(block: ExtractedBlock, median_font: float) -> bool:
    text = normalize_text(block.text)
    if not text:
        return False

    words = re.findall(r"\w+", text)
    if len(words) > 18 or len(text) > 120:
        return False

    # Strong heading signals.
    if _HEADING_RE.match(text):
        return True

    if text.isupper() and len(words) <= 10:
        return True

    if block.is_bold and block.font_size >= median_font * 1.15 and len(words) <= 12:
        return True

    if block.font_size >= median_font * 1.35 and len(words) <= 12:
        return True

    # Too permissive if we allow colons or question marks alone.
    # Keep those as paragraphs unless other signals are strong.
    if _looks_like_heading_text(text) and block.font_size >= median_font * 1.20:
        return True

    return False


def _is_bullet(block: ExtractedBlock) -> bool:
    return bool(_BULLET_RE.match(normalize_text(block.text)))


def _heading_path_from_text(text: str, current_path: list[str]) -> list[str]:
    stripped = normalize_text(text)
    match = _HEADING_RE.match(stripped)
    if match:
        prefix = match.group(1).rstrip(".)")
        levels = [part for part in prefix.split(".") if part]
        if not levels:
            levels = current_path[:]
        return levels

    # Generic short title-like heading: use a stable slug and preserve hierarchy.
    slug = re.sub(r"[^a-z0-9\.\- ]+", "", stripped.lower())
    slug = re.sub(r"\s+", "-", slug).strip("-")
    return (
        current_path[:] + [slug or "section"] if not current_path else current_path[:]
    )


def _short_title(text: str) -> str:
    t = normalize_text(text)
    t = re.sub(r"\s+", " ", t)
    if len(t) > 90:
        t = t[:90].rstrip()
    return t


def _merge_adjacent_blocks(
    blocks: list[ExtractedBlock],
    median_font: float,
    x_tol: float = 28.0,
    y_gap_tol: float = 18.0,
) -> list[ExtractedBlock]:
    """
    Merge wrapped PDF lines/blocks into a single paragraph-like block.

    This is still layout-aware, but it does NOT split on sentences.
    """
    if not blocks:
        return []

    ordered = sorted(blocks, key=_bbox_sort_key)
    merged: list[ExtractedBlock] = []
    buffer: list[ExtractedBlock] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return

        parts: list[str] = []
        bbox = buffer[0].bbox
        font_size = max(b.font_size for b in buffer)
        is_bold = any(b.is_bold for b in buffer)
        page = buffer[0].page

        for blk in buffer:
            txt = normalize_text(blk.text)
            if not txt:
                continue
            if parts and parts[-1].endswith("-"):
                parts[-1] = parts[-1][:-1] + txt.lstrip()
            else:
                parts.append(txt)
            bbox = _union_bbox(bbox, blk.bbox)

        combined_text = normalize_text(" ".join(parts))
        if combined_text:
            merged.append(
                ExtractedBlock(
                    text=combined_text,
                    bbox=bbox,
                    font_size=font_size,
                    page=page,
                    kind="paragraph",
                    is_bold=is_bold,
                )
            )
        buffer = []

    for blk in ordered:
        text = normalize_text(blk.text)
        if not text:
            continue

        if _is_heading(blk, median_font) or _is_bullet(blk):
            flush()
            merged.append(blk)
            continue

        if not buffer:
            buffer = [blk]
            continue

        prev = buffer[-1]
        if prev.bbox is None or blk.bbox is None:
            # If we cannot reason about layout, keep a conservative buffer.
            buffer.append(blk)
            continue

        prev_x0, _, _, prev_y1 = prev.bbox
        curr_x0, curr_y0, _, _ = blk.bbox
        same_indent = abs(prev_x0 - curr_x0) <= x_tol
        vertical_gap = curr_y0 - prev_y1

        # New paragraph if the gap is large or indentation changed a lot.
        if not same_indent or vertical_gap > y_gap_tol:
            flush()
            buffer = [blk]
            continue

        # Same paragraph continuation.
        buffer.append(blk)

    flush()
    return merged


def _join_chunks(chunks: list[str]) -> str:
    chunks = [normalize_text(c) for c in chunks if normalize_text(c)]
    if not chunks:
        return ""
    return "\n\n".join(chunks)


def _split_on_explicit_paragraphs(text: str) -> list[str]:
    """
    Split only on explicit blank lines.
    No sentence splitting here.
    """
    text = normalize_text(text)
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()] or [text]


def _should_start_new_semantic_chunk(
    current_text: str,
    next_text: str,
    current_tokens: int,
    next_tokens: int,
    current_page: int,
    next_page: int,
    max_tokens_per_node: int,
    max_pages_per_node: int,
) -> bool:
    if current_page != next_page:
        return True

    if current_tokens + next_tokens > max_tokens_per_node:
        return True

    if max_pages_per_node <= 1 and current_page != next_page:
        return True

    # If the next block is clearly a new bullet or heading-like line, start a new chunk.
    if _BULLET_RE.match(next_text):
        return True
    if _looks_like_heading_text(next_text):
        return True

    # If the current chunk already looks complete and the next block starts a new idea,
    # prefer a new node instead of forcing a sentence-level merge.
    if re.search(r"[.!?;:]$", current_text) and next_text[:1].isupper():
        return True

    return False


@dataclass(slots=True)
class _ChunkBuffer:
    parts: list[str]
    bboxes: list[tuple[float, float, float, float] | None]
    token_count: int
    start_page: int
    end_page: int
    kind: str
    is_bold: bool

    @property
    def text(self) -> str:
        return _join_chunks(self.parts)

    def add(
        self,
        text: str,
        bbox: tuple[float, float, float, float] | None,
        page: int,
        kind: str,
        is_bold: bool,
    ) -> None:
        text = normalize_text(text)
        if not text:
            return
        self.parts.append(text)
        self.bboxes.append(bbox)
        self.token_count += _token_count(text)
        self.end_page = page
        if kind != "paragraph":
            self.kind = kind
        self.is_bold = self.is_bold or is_bold

    def merged_bbox(self) -> tuple[float, float, float, float] | None:
        bbox = None
        for b in self.bboxes:
            bbox = _union_bbox(bbox, b)
        return bbox


def _make_chunk_buffer(block: ExtractedBlock) -> _ChunkBuffer:
    return _ChunkBuffer(
        parts=[normalize_text(block.text)],
        bboxes=[block.bbox],
        token_count=_token_count(block.text),
        start_page=block.page,
        end_page=block.page,
        kind=block.kind,
        is_bold=block.is_bold,
    )


def _split_and_normalize_block(block: ExtractedBlock) -> list[str]:
    """
    Preserve explicit paragraph breaks only.
    If a block is extremely long and contains explicit blank-line separated
    sections, split them. Otherwise keep the block intact.
    """
    text = normalize_text(block.text)
    if not text:
        return []

    parts = _split_on_explicit_paragraphs(text)

    # Important: do NOT split into sentences. That's what caused bad tree nodes.
    return parts


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def build_tree(
    pdf_path: str | Path, doc_side: str, cfg: AppConfig | None = None
) -> DocumentTree:
    """
    Build a PageIndex-like hierarchical tree without changing the existing schema.

    What this does:
    - page-by-page extraction
    - removes repetitive headers/footers
    - merges wrapped lines into paragraph-like blocks
    - detects headings conservatively
    - groups body text into semantic chunks capped by tokens/pages
    - keeps section nodes as structural parents
    """
    cfg = cfg or AppConfig.load()
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    # Tunable limits; read from config if present, otherwise use safe defaults.
    max_tokens_per_node = int(getattr(cfg, "max_tokens_per_node", 220))
    max_pages_per_node = int(getattr(cfg, "max_pages_per_node", 1))
    max_tokens_per_node = max(64, max_tokens_per_node)
    max_pages_per_node = max(1, max_pages_per_node)

    blocks_by_page = [_page_blocks(page) for page in doc]
    blocks_by_page = _dedupe_repeated_blocks(blocks_by_page)

    all_sizes = [
        blk.font_size
        for page_blocks in blocks_by_page
        for blk in page_blocks
        if blk.font_size > 0
    ]
    median_font = 0.0
    if all_sizes:
        all_sizes_sorted = sorted(all_sizes)
        median_font = all_sizes_sorted[len(all_sizes_sorted) // 2]

    # Merge wrapped layout fragments before tree creation.
    blocks_by_page = [
        _merge_adjacent_blocks(page_blocks, median_font)
        for page_blocks in blocks_by_page
    ]

    page_count = len(blocks_by_page)
    node_map: dict[str, SourceNode] = {}
    leaf_ids: list[str] = []
    leaf_positions: dict[str, int] = {}

    doc_id = f"{doc_side}-{pdf_path.stem}-{hashlib.sha1(str(pdf_path).encode()).hexdigest()[:8]}"
    root_id = f"{doc_side}:document"
    node_map[root_id] = SourceNode(
        node_id=root_id,
        doc_side=doc_side,
        page=0,
        kind="document",
        path=[],
        text=pdf_path.name,
        parent_id=None,
    )

    current_section_id = root_id
    current_path: list[str] = []
    section_counter_by_page: dict[int, int] = defaultdict(int)
    paragraph_counter_by_page: dict[int, int] = defaultdict(int)

    chunk_buffer: _ChunkBuffer | None = None

    def flush_chunk_buffer() -> None:
        nonlocal chunk_buffer
        if chunk_buffer is None:
            return

        chunk_text = normalize_text(chunk_buffer.text)
        if not chunk_text:
            chunk_buffer = None
            return

        paragraph_counter_by_page[chunk_buffer.start_page] += 1
        para_id = _node_id(
            doc_side,
            chunk_buffer.start_page,
            "para",
            current_path,
            paragraph_counter_by_page[chunk_buffer.start_page],
        )

        kind = "bullet" if chunk_buffer.kind == "bullet" else "paragraph"

        node_map[para_id] = SourceNode(
            node_id=para_id,
            doc_side=doc_side,
            page=chunk_buffer.start_page,
            kind=kind,
            path=current_path[:],
            text=chunk_text,
            parent_id=current_section_id,
            bbox=chunk_buffer.merged_bbox(),
            start_char=0,
            end_char=len(chunk_text),
            token_count=_token_count(chunk_text),
            stable_hash=_stable_hash(chunk_text),
        )
        node_map[current_section_id].children.append(para_id)
        leaf_positions[para_id] = len(leaf_ids)
        leaf_ids.append(para_id)
        chunk_buffer = None

    for page_index, page_blocks in enumerate(blocks_by_page, start=1):
        page_id = f"{doc_side}:p{page_index:02d}:page"
        node_map[page_id] = SourceNode(
            node_id=page_id,
            doc_side=doc_side,
            page=page_index,
            kind="page",
            path=[],
            text=f"Page {page_index}",
            parent_id=root_id,
        )
        node_map[root_id].children.append(page_id)

        for blk in page_blocks:
            text = normalize_text(blk.text)
            if not text:
                continue

            # Conservative heading detection. This is where we avoid creating
            # fake section nodes from short sentence fragments.
            if _is_heading(blk, median_font):
                flush_chunk_buffer()

                current_path = _heading_path_from_text(text, current_path)
                section_counter_by_page[page_index] += 1
                sec_id = _node_id(
                    doc_side,
                    page_index,
                    "section",
                    current_path,
                    section_counter_by_page[page_index],
                )

                # Keep the actual heading text as the section node text.
                node_map[sec_id] = SourceNode(
                    node_id=sec_id,
                    doc_side=doc_side,
                    page=page_index,
                    kind="section",
                    path=current_path[:],
                    text=_short_title(text),
                    parent_id=page_id,
                    bbox=blk.bbox,
                    start_char=0,
                    end_char=len(text),
                    token_count=_token_count(text),
                    stable_hash=_stable_hash(text),
                )
                node_map[page_id].children.append(sec_id)

                # Keep section nodes in leaf_ids to preserve your current schema behavior.
                leaf_positions[sec_id] = len(leaf_ids)
                leaf_ids.append(sec_id)

                current_section_id = sec_id
                continue

            # Non-heading text: group into semantic chunks, not sentences.
            parts = _split_and_normalize_block(blk)
            if not parts:
                continue

            for part in parts:
                part_tokens = _token_count(part)

                if chunk_buffer is None:
                    chunk_buffer = _make_chunk_buffer(
                        ExtractedBlock(
                            text=part,
                            bbox=blk.bbox,
                            font_size=blk.font_size,
                            page=page_index,
                            kind="bullet" if _BULLET_RE.match(part) else "paragraph",
                            is_bold=blk.is_bold,
                        )
                    )
                    continue

                current_text = chunk_buffer.text
                current_page = chunk_buffer.end_page

                if _should_start_new_semantic_chunk(
                    current_text=current_text,
                    next_text=part,
                    current_tokens=chunk_buffer.token_count,
                    next_tokens=part_tokens,
                    current_page=current_page,
                    next_page=page_index,
                    max_tokens_per_node=max_tokens_per_node,
                    max_pages_per_node=max_pages_per_node,
                ):
                    flush_chunk_buffer()
                    chunk_buffer = _make_chunk_buffer(
                        ExtractedBlock(
                            text=part,
                            bbox=blk.bbox,
                            font_size=blk.font_size,
                            page=page_index,
                            kind="bullet" if _BULLET_RE.match(part) else "paragraph",
                            is_bold=blk.is_bold,
                        )
                    )
                    continue

                chunk_buffer.add(
                    text=part,
                    bbox=blk.bbox,
                    page=page_index,
                    kind="bullet" if _BULLET_RE.match(part) else "paragraph",
                    is_bold=blk.is_bold,
                )

        # Page boundary: flush the chunk buffer so page-aware leaf nodes stay clean.
        flush_chunk_buffer()

    doc.close()

    tree = DocumentTree(
        doc_id=doc_id,
        doc_side=doc_side,
        source_path=str(pdf_path),
        page_count=page_count,
        created_at=_now_iso(),
        nodes=node_map,
        leaf_ids=leaf_ids,
        leaf_positions=leaf_positions,
    )

    log.info("Built tree for {}", pdf_path)
    return tree


def get_node(tree: DocumentTree, node_id: str) -> SourceNode | None:
    return tree.nodes.get(node_id)


def iter_leaf_nodes(tree: DocumentTree) -> Iterable[SourceNode]:
    for node_id in tree.leaf_ids:
        node = tree.nodes.get(node_id)
        if node is not None:
            yield node


def write_tree(tree: DocumentTree, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(msgspec.json.encode(tree))
    return out_path


def read_tree(path: str | Path) -> DocumentTree:
    return msgspec.json.decode(Path(path).read_bytes(), type=DocumentTree)
