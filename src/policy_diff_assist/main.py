from __future__ import annotations

import argparse
from pathlib import Path

from .config import AppConfig
from .logging import setup_logging, get_logger
from .pipeline import compare_documents
from .ui import launch

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Policy Diff Assistant")
    parser.add_argument("--legacy", type=str, help="Path to the legacy PDF")
    parser.add_argument("--modern", type=str, help="Path to the modern PDF")
    parser.add_argument("--output-root", type=str, default=None, help="Output root directory")
    parser.add_argument("--ui", action="store_true", help="Launch the Gradio UI")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = AppConfig.load()
    setup_logging(cfg)

    if args.ui or not (args.legacy and args.modern):
        launch()
        return

    result = compare_documents(
        Path(args.legacy),
        Path(args.modern),
        output_root=args.output_root,
        cfg=cfg,
    )
    print(f"Session: {result.session_id}")
    print(f"Report markdown: {result.report_md_path}")
    print(f"Report PDF: {result.report_pdf_path}")
    print(f"Report JSON: {result.report_json_path}")
    print(result.summary)


if __name__ == "__main__":  # pragma: no cover
    main()
