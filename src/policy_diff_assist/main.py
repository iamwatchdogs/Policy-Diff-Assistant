from __future__ import annotations

import argparse
from pathlib import Path

from policy_diff_assist.config import AppConfig
from policy_diff_assist.logging import setup_logging, get_logger
from policy_diff_assist.pipeline import compare_documents
from policy_diff_assist.ui import launch

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
    log.info("Session: {}", result.session_id)
    log.info("Report markdown: {}", result.report_md_path)
    log.info("Report PDF: {}", result.report_pdf_path)
    log.info("Report JSON: {}", result.report_json_path)
    log.info("Result Summary:\n{}", result.summary)


if __name__ == "__main__":  # pragma: no cover
    main()

