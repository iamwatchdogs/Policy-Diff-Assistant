from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import msgspec

from .config import AppConfig
from .models import DocumentTree, SourceNode
from .logging import get_logger

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


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _page_blocks(page: fitz.Page) -> list[ExtractedBlock]:
    data = page.get_text("dict")
    blocks: list[ExtractedBlock] = []
    for block in data.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        text_lines: list[str] = []
        font_sizes: list[float] = []
        for line in block.get("lines", []):
            line_text_parts: list[str] = []
            for span in line.get("spans", []):
                txt = span.get("text", "")
                if txt:
                    line_text_parts.append(txt)
                    font_sizes.append(float(span.get("size", 0.0)))
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
            )
        )
    return blocks


def _is_header_footer_candidate(text: str) -> bool:
    t = normalize_text(text)
    if len(t) > 120:
        return False
    return bool(re.fullmatch(r"[\w\s\-.,:/()]+", t)) and any(ch.isalpha() for ch in t)


def _dedupe_repeated_blocks(blocks_by_page: list[list[ExtractedBlock]]) -> list[list[ExtractedBlock]]:
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


def _slug_heading(text: str) -> str:
    t = normalize_text(text)
    if not t:
        return "section"
    if len(t) > 36:
        t = t[:36]
    t = t.lower()
    t = re.sub(r"[^a-z0-9\.\- ]+", "", t)
    t = re.sub(r"\s+", "-", t).strip("-")
    return t or "section"


def _heading_path_from_text(text: str, current_path: list[str]) -> list[str]:
    stripped = normalize_text(text)
    match = _HEADING_RE.match(stripped)
    if match:
        prefix = match.group(1).rstrip(".)")
        levels = [part for part in prefix.split(".") if part]
        if not levels:
            levels = current_path[:]
        return levels
    # generic heading: keep current path but add slug if top-level-ish
    return current_path[:] + [_slug_heading(stripped)] if not current_path else current_path[:]


def _is_heading(block: ExtractedBlock, median_font: float) -> bool:
    text = normalize_text(block.text)
    if not text:
        return False
    if len(text) <= 90 and (
        text.isupper()
        or bool(_HEADING_RE.match(text))
        or text.endswith(":")
        or (median_font > 0 and block.font_size >= median_font * 1.18)
    ):
        return True
    return False


def _token_count(text: str) -> int:
    return max(1, len(re.findall(r"\w+", text)))


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _node_id(doc_side: str, page: int, kind: str, path: list[str], local_index: int) -> str:
    path_part = ".".join(path) if path else "root"
    return f"{doc_side}:p{page:02d}:{path_part}:{kind}{local_index}"


def build_tree(pdf_path: str | Path, doc_side: str, cfg: AppConfig | None = None) -> DocumentTree:
    log.info("Started Built tree for {}", pdf_path)

    cfg = cfg or AppConfig.load()
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    blocks_by_page = [_page_blocks(page) for page in doc]
    blocks_by_page = _dedupe_repeated_blocks(blocks_by_page)

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

    median_font = 0.0
    all_sizes = [blk.font_size for page_blocks in blocks_by_page for blk in page_blocks if blk.font_size > 0]
    if all_sizes:
        all_sizes_sorted = sorted(all_sizes)
        median_font = all_sizes_sorted[len(all_sizes_sorted) // 2]

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

            if _is_heading(blk, median_font):
                current_path = _heading_path_from_text(text, current_path)
                section_counter_by_page[page_index] += 1
                sec_id = _node_id(doc_side, page_index, "section", current_path, section_counter_by_page[page_index])
                current_section_id = sec_id
                node_map[sec_id] = SourceNode(
                    node_id=sec_id,
                    doc_side=doc_side,
                    page=page_index,
                    kind="section",
                    path=current_path[:],
                    text=text,
                    parent_id=page_id,
                    bbox=blk.bbox,
                    token_count=_token_count(text),
                    stable_hash=_stable_hash(text),
                )
                node_map[page_id].children.append(sec_id)
                leaf_positions[sec_id] = len(leaf_ids)
                leaf_ids.append(sec_id)
                continue

            # Split long blocks on blank lines to preserve clause-ish granularity.
            parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            if not parts:
                parts = [text]

            for part in parts:
                paragraph_counter_by_page[page_index] += 1
                para_id = _node_id(doc_side, page_index, "para", current_path, paragraph_counter_by_page[page_index])

                kind = "bullet" if _BULLET_RE.match(part) else "paragraph"
                node_map[para_id] = SourceNode(
                    node_id=para_id,
                    doc_side=doc_side,
                    page=page_index,
                    kind=kind,
                    path=current_path[:],
                    text=part,
                    parent_id=current_section_id,
                    bbox=blk.bbox,
                    token_count=_token_count(part),
                    stable_hash=_stable_hash(part),
                )
                node_map[current_section_id].children.append(para_id)
                leaf_positions[para_id] = len(leaf_ids)
                leaf_ids.append(para_id)

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

    log.info("Successfully Built tree for {}", pdf_path)
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


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
