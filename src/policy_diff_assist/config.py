from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import tomllib as _toml  # py3.11+
except Exception:  # pragma: no cover
    try:
        import tomli as _toml  # type: ignore
    except Exception:  # pragma: no cover
        _toml = None  # type: ignore


load_dotenv()


def _read_toml(path: Path) -> dict[str, Any]:
    if _toml is None or not path.exists():
        return {}
    with path.open("rb") as f:
        return _toml.load(f)


@dataclass(slots=True)
class AppConfig:
    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    sessions_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "sessions")
    reports_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "reports")
    logs_dir: Path = field(default_factory=lambda: Path.cwd() / "logs")
    config_dir: Path = field(default_factory=lambda: Path.cwd() / "configs")

    embedding_model_name: str = "Qwen/Qwen3-Embedding-4B"
    fallback_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: str = "Qwen/Qwen3.5-9B-Instruct"
    use_vllm: bool = False
    temperature: float = 0.2
    max_new_tokens: int = 512
    top_k_candidates: int = 5
    min_similarity: float = 0.75
    unchanged_threshold: float = 0.92
    modified_threshold: float = 0.75
    batch_size: int = 64
    chunk_chars: int = 1200
    neighbors_window: int = 1
    keep_intermediate_json: bool = True
    hf_token: str | None = None

    @classmethod
    def load(cls, base_dir: str | os.PathLike[str] | None = None) -> "AppConfig":
        project_root = Path(base_dir or Path.cwd()).resolve()
        config_dir = project_root / "configs"

        app_toml = _read_toml(config_dir / "app.toml")
        env = os.environ

        cfg = cls(
            project_root=project_root,
            data_dir=project_root / "data",
            sessions_dir=project_root / "data" / "sessions",
            reports_dir=project_root / "data" / "reports",
            logs_dir=project_root / "logs",
            config_dir=config_dir,
            embedding_model_name=env.get("EMBEDDING_MODEL", app_toml.get("embedding_model_name", cls.embedding_model_name)),
            fallback_embedding_model_name=env.get(
                "FALLBACK_EMBEDDING_MODEL", app_toml.get("fallback_embedding_model_name", cls.fallback_embedding_model_name)
            ),
            llm_model_name=env.get("LLM_MODEL", app_toml.get("llm_model_name", cls.llm_model_name)),
            use_vllm=_as_bool(env.get("USE_VLLM", app_toml.get("use_vllm", cls.use_vllm))),
            temperature=float(env.get("TEMPERATURE", app_toml.get("temperature", cls.temperature))),
            max_new_tokens=int(env.get("MAX_NEW_TOKENS", app_toml.get("max_new_tokens", cls.max_new_tokens))),
            top_k_candidates=int(env.get("TOP_K_CANDIDATES", app_toml.get("top_k_candidates", cls.top_k_candidates))),
            min_similarity=float(env.get("MIN_SIMILARITY", app_toml.get("min_similarity", cls.min_similarity))),
            unchanged_threshold=float(env.get("UNCHANGED_THRESHOLD", app_toml.get("unchanged_threshold", cls.unchanged_threshold))),
            modified_threshold=float(env.get("MODIFIED_THRESHOLD", app_toml.get("modified_threshold", cls.modified_threshold))),
            batch_size=int(env.get("BATCH_SIZE", app_toml.get("batch_size", cls.batch_size))),
            chunk_chars=int(env.get("CHUNK_CHARS", app_toml.get("chunk_chars", cls.chunk_chars))),
            neighbors_window=int(env.get("NEIGHBORS_WINDOW", app_toml.get("neighbors_window", cls.neighbors_window))),
            keep_intermediate_json=_as_bool(
                env.get("KEEP_INTERMEDIATE_JSON", app_toml.get("keep_intermediate_json", cls.keep_intermediate_json))
            ),
            hf_token=env.get("HF_TOKEN") or app_toml.get("hf_token"),
        )
        cfg.ensure_dirs()
        return cfg

    def ensure_dirs(self) -> None:
        for path in [self.data_dir, self.sessions_dir, self.reports_dir, self.logs_dir, self.config_dir]:
            path.mkdir(parents=True, exist_ok=True)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def build_default_session_dir(cfg: AppConfig, session_id: str) -> Path:
    path = cfg.sessions_dir / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path
