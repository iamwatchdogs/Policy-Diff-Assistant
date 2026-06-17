from __future__ import annotations

from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import msgspec

from .models import ComparisonResult, MatchRecord, ReportArtifact


def build_report_markdown(result: ComparisonResult) -> str:
    counts = _count_changes(result.matches)
    lines = []
    lines.append("# Policy Diff Report")
    lines.append("")
    lines.append(f"- Session ID: `{result.session_id}`")
    lines.append(f"- Legacy document: `{result.legacy_tree.source_path}`")
    lines.append(f"- Modern document: `{result.modern_tree.source_path}`")
    lines.append(f"- Total matches: **{len(result.matches)}**")
    lines.append(
        f"- Unchanged: **{counts['unchanged']}**, Modified: **{counts['modified']}**, Added: **{counts['added']}**, Removed: **{counts['removed']}**"
    )
    lines.append("")
    if result.summary:
        lines.append("## Executive summary")
        lines.append("")
        lines.append(result.summary.strip())
        lines.append("")

    lines.append("## Detailed changes")
    lines.append("")
    for idx, match in enumerate(result.matches, start=1):
        lines.append(f"### {idx}. {match.change_type.title()} | similarity={match.similarity:.3f}")
        if match.legacy_id:
            lines.append(f"- Legacy: `{match.legacy_id}` (page {match.legacy_page})")
        if match.modern_id:
            lines.append(f"- Modern: `{match.modern_id}` (page {match.modern_page})")
        lines.append(f"- Lexical score: {match.lexical_score:.3f}")
        if match.legacy_text:
            lines.append("")
            lines.append("**Legacy text**")
            lines.append("")
            lines.append(_indent_block(match.legacy_text))
        if match.modern_text:
            lines.append("")
            lines.append("**Modern text**")
            lines.append("")
            lines.append(_indent_block(match.modern_text))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _indent_block(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def write_report_pdf(report_md: str, out_path: str | Path, title: str = "Policy Diff Report") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    page = doc.new_page()
    margin = 50
    y = margin
    width = page.rect.width - 2 * margin

    def new_page():
        nonlocal page, y
        page = doc.new_page()
        y = margin

    def write_line(text: str, font_size: int = 11, indent: int = 0) -> None:
        nonlocal page, y
        rect_height = font_size * 1.8
        if y + rect_height > page.rect.height - margin:
            new_page()
        rect = fitz.Rect(margin + indent, y, margin + width, y + rect_height)
        page.insert_textbox(rect, text, fontsize=font_size, fontname="helv", align=0)
        y += rect_height

    write_line(title, font_size=18)
    y += 6

    for line in report_md.splitlines():
        if line.startswith("# "):
            write_line(line[2:].strip(), font_size=18)
            y += 4
        elif line.startswith("## "):
            write_line(line[3:].strip(), font_size=14)
            y += 2
        elif line.startswith("### "):
            write_line(line[4:].strip(), font_size=12)
        elif line.startswith("- "):
            write_line("• " + line[2:], font_size=10, indent=10)
        elif line.startswith("    "):
            write_line(line.strip(), font_size=9, indent=18)
        elif not line.strip():
            y += 4
        else:
            write_line(line, font_size=10)

    doc.save(out_path)
    doc.close()
    return out_path


def write_report_json(result: ComparisonResult, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(msgspec.json.encode(result))
    return out_path


def build_report_artifact(result: ComparisonResult, out_dir: str | Path) -> ReportArtifact:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "report.md"
    pdf_path = out_dir / "report.pdf"
    json_path = out_dir / "report.json"

    md = build_report_markdown(result)
    md_path.write_text(md, encoding="utf-8")
    write_report_pdf(md, pdf_path)
    write_report_json(result, json_path)

    counts = _count_changes(result.matches)
    return ReportArtifact(
        session_id=result.session_id,
        report_md_path=str(md_path),
        report_pdf_path=str(pdf_path),
        report_json_path=str(json_path),
        summary=result.summary,
        match_count=len(result.matches),
        removed_count=counts["removed"],
        added_count=counts["added"],
        modified_count=counts["modified"],
        unchanged_count=counts["unchanged"],
    )


def _count_changes(matches: Iterable[MatchRecord]) -> dict[str, int]:
    counts = {"removed": 0, "added": 0, "modified": 0, "unchanged": 0}
    for m in matches:
        if m.change_type in counts:
            counts[m.change_type] += 1
        elif m.change_type in {"split", "merged"}:
            counts["modified"] += 1
    return counts
