from __future__ import annotations

import gradio as gr

from policy_diff_assist.config import AppConfig
from policy_diff_assist.logging import setup_logging, get_logger
from policy_diff_assist.pipeline import compare_documents
from policy_diff_assist.models import ProgressState

log = get_logger(__name__)


def create_ui(cfg: AppConfig | None = None) -> gr.Blocks:
    cfg = cfg or AppConfig.load()
    setup_logging(cfg)

    with gr.Blocks(title="Policy Diff Assistant") as demo:
        gr.Markdown("# Policy Diff Assistant")
        gr.Markdown("Upload two PDFs and get a semantic diff report with provenance.")

        with gr.Row():
            legacy_file = gr.File(label="Legacy policy PDF", file_types=[".pdf"], type="filepath")
            modern_file = gr.File(label="Modern policy PDF", file_types=[".pdf"], type="filepath")

        run_btn = gr.Button("Compare")
        status = gr.Markdown("Idle.")
        summary = gr.Markdown("")
        report_file = gr.File(label="Download report", interactive=False)

        def _run(legacy_fp, modern_fp, progress=gr.Progress(track_tqdm=False)):
            if legacy_fp is None or modern_fp is None:
                yield "Please upload both PDFs.", "", None
                return

            log.info("User submitted PDFs")

            def progress_cb(state: ProgressState):
                progress(
                    min(max(state.percent / 100.0, 0.0), 1.0),
                    desc=f"{state.stage}: {state.message}",
                )

            result = compare_documents(
                legacy_fp,
                modern_fp,
                output_root=cfg.sessions_dir,
                cfg=cfg,
                progress_cb=progress_cb,
            )
            status_msg = f"Done. Session `{result.session_id}`"
            yield status_msg, result.summary, result.report_pdf_path

        run_btn.click(_run, inputs=[legacy_file, modern_file], outputs=[status, summary, report_file])
    
    log.info("UI created successfully")    
    return demo


def launch() -> None:
    log.info("Application Initiated")

    cfg = AppConfig.load()
    demo = create_ui(cfg)
    demo.queue(default_concurrency_limit=1).launch(share=True)


if __name__ == "__main__":  # pragma: no cover
    launch()
