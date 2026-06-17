from __future__ import annotations

import sys
from datetime import datetime
from loguru import logger

from policy_diff_assistpolicy_diff_assist.config import AppConfig


def setup_logging(cfg: AppConfig) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        cfg.logs_dir / f"policy_diff_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        backtrace=False,
        diagnose=False,
    )


def get_logger(name: str | None = None):
    return logger.bind(module=name or "policy_diff_assist")
