from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field
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


def _bootstrap_env(project_root: Path) -> None:
    # Load .env from common locations. Never override already-exported values.
    load_dotenv(dotenv_path=project_root / ".env", override=False)
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    load_dotenv(override=False)


def _read_toml(path: Path) -> dict[str, Any]:
    if _toml is None or not path.exists():
        return {}
    with path.open("rb") as f:
        return _toml.load(f)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _default_value(cls: type, name: str) -> Any:
    field_def = cls.__dataclass_fields__[name]
    if field_def.default is not MISSING:
        return field_def.default
    if field_def.default_factory is not MISSING:  # type: ignore[truthy-function]
        return field_def.default_factory()  # type: ignore[misc]
    return None


@dataclass(slots=True)
class AppConfig:
    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    sessions_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "sessions")
    reports_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "reports")
    logs_dir: Path = field(default_factory=lambda: Path.cwd() / "logs")
    config_dir: Path = field(default_factory=lambda: Path.cwd() / "configs")

    # Use a valid public HF repo id by default.
    llm_model_name: str = "Qwen/Qwen3.5-9B"
    embedding_model_name: str = "Qwen/Qwen3-Embedding-4B"
    fallback_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Force vLLM first.
    use_vllm: bool = True
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

    similarity_use_gpu: bool = True
    similarity_gpu_mode: str = "auto"
    similarity_max_workers: int = 0
    similarity_parallel_lexical: bool = True
    similarity_parallel_heading: bool = True
    similarity_gpu_cosine: bool = True

    omp_num_threads: int = 0
    mkl_num_threads: int = 0
    openblas_num_threads: int = 0
    numexpr_max_threads: int = 0

    @classmethod
    def load(cls, base_dir: str | os.PathLike[str] | None = None) -> "AppConfig":
        project_root = Path(base_dir or Path.cwd()).resolve()

        # Prefer the repository root that contains configs/ and .env if available.
        candidate = Path(__file__).resolve().parents[2]
        if (candidate / "configs").exists():
            project_root = candidate

        _bootstrap_env(project_root)

        config_dir = project_root / "configs"
        app_toml = _read_toml(config_dir / "app.toml")
        env = os.environ

        def _env_or_toml(env_key: str, toml_key: str, field_name: str) -> Any:
            value = env.get(env_key)
            if value is not None and str(value).strip() != "":
                return value
            if toml_key in app_toml and app_toml[toml_key] is not None:
                return app_toml[toml_key]
            return _default_value(cls, field_name)

        hf_token = (
            env.get("HF_TOKEN")
            or env.get("HUGGINGFACE_HUB_TOKEN")
            or app_toml.get("hf_token")
            or app_toml.get("huggingface_token")
        )

        cfg = cls(
            project_root=project_root,
            data_dir=project_root / "data",
            sessions_dir=project_root / "data" / "sessions",
            reports_dir=project_root / "data" / "reports",
            logs_dir=project_root / "logs",
            config_dir=config_dir,
            llm_model_name=str(
                _env_or_toml("LLM_MODEL", "llm_model_name", "llm_model_name")
            ),
            embedding_model_name=str(
                _env_or_toml(
                    "EMBEDDING_MODEL", "embedding_model_name", "embedding_model_name"
                )
            ),
            fallback_embedding_model_name=str(
                _env_or_toml(
                    "FALLBACK_EMBEDDING_MODEL",
                    "fallback_embedding_model_name",
                    "fallback_embedding_model_name",
                )
            ),
            use_vllm=_as_bool(_env_or_toml("USE_VLLM", "use_vllm", "use_vllm")),
            temperature=float(
                _env_or_toml("TEMPERATURE", "temperature", "temperature")
            ),
            max_new_tokens=int(
                _env_or_toml("MAX_NEW_TOKENS", "max_new_tokens", "max_new_tokens")
            ),
            top_k_candidates=int(
                _env_or_toml("TOP_K_CANDIDATES", "top_k_candidates", "top_k_candidates")
            ),
            min_similarity=float(
                _env_or_toml("MIN_SIMILARITY", "min_similarity", "min_similarity")
            ),
            unchanged_threshold=float(
                _env_or_toml(
                    "UNCHANGED_THRESHOLD", "unchanged_threshold", "unchanged_threshold"
                )
            ),
            modified_threshold=float(
                _env_or_toml(
                    "MODIFIED_THRESHOLD", "modified_threshold", "modified_threshold"
                )
            ),
            batch_size=int(_env_or_toml("BATCH_SIZE", "batch_size", "batch_size")),
            chunk_chars=int(_env_or_toml("CHUNK_CHARS", "chunk_chars", "chunk_chars")),
            neighbors_window=int(
                _env_or_toml("NEIGHBORS_WINDOW", "neighbors_window", "neighbors_window")
            ),
            keep_intermediate_json=_as_bool(
                _env_or_toml(
                    "KEEP_INTERMEDIATE_JSON",
                    "keep_intermediate_json",
                    "keep_intermediate_json",
                )
            ),
            hf_token=str(hf_token) if hf_token else None,
            similarity_use_gpu=_as_bool(
                _env_or_toml(
                    "SIMILARITY_USE_GPU", "similarity_use_gpu", "similarity_use_gpu"
                )
            ),
            similarity_gpu_mode=str(
                _env_or_toml(
                    "SIMILARITY_GPU_MODE", "similarity_gpu_mode", "similarity_gpu_mode"
                )
            ),
            similarity_max_workers=int(
                _env_or_toml(
                    "SIMILARITY_MAX_WORKERS",
                    "similarity_max_workers",
                    "similarity_max_workers",
                )
            ),
            similarity_parallel_lexical=_as_bool(
                _env_or_toml(
                    "SIMILARITY_PARALLEL_LEXICAL",
                    "similarity_parallel_lexical",
                    "similarity_parallel_lexical",
                )
            ),
            similarity_parallel_heading=_as_bool(
                _env_or_toml(
                    "SIMILARITY_PARALLEL_HEADING",
                    "similarity_parallel_heading",
                    "similarity_parallel_heading",
                )
            ),
            similarity_gpu_cosine=_as_bool(
                _env_or_toml(
                    "SIMILARITY_GPU_COSINE",
                    "similarity_gpu_cosine",
                    "similarity_gpu_cosine",
                )
            ),
            omp_num_threads=int(
                _env_or_toml("OMP_NUM_THREADS", "omp_num_threads", "omp_num_threads")
            ),
            mkl_num_threads=int(
                _env_or_toml("MKL_NUM_THREADS", "mkl_num_threads", "mkl_num_threads")
            ),
            openblas_num_threads=int(
                _env_or_toml(
                    "OPENBLAS_NUM_THREADS",
                    "openblas_num_threads",
                    "openblas_num_threads",
                )
            ),
            numexpr_max_threads=int(
                _env_or_toml(
                    "NUMEXPR_MAX_THREADS", "numexpr_max_threads", "numexpr_max_threads"
                )
            ),
        )
        cfg.ensure_dirs()
        cfg._apply_runtime_env()
        return cfg

    def ensure_dirs(self) -> None:
        for path in [
            self.data_dir,
            self.sessions_dir,
            self.reports_dir,
            self.logs_dir,
            self.config_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _apply_runtime_env(self) -> None:
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", self.hf_token)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_default_session_dir(cfg: AppConfig, session_id: str) -> Path:
    path = cfg.sessions_dir / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path
from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field
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


def _bootstrap_env(project_root: Path) -> None:
    # Load .env from common locations. Never override already-exported values.
    load_dotenv(dotenv_path=project_root / ".env", override=False)
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    load_dotenv(override=False)


def _read_toml(path: Path) -> dict[str, Any]:
    if _toml is None or not path.exists():
        return {}
    with path.open("rb") as f:
        return _toml.load(f)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _default_value(cls: type, name: str) -> Any:
    field_def = cls.__dataclass_fields__[name]
    if field_def.default is not MISSING:
        return field_def.default
    if field_def.default_factory is not MISSING:  # type: ignore[truthy-function]
        return field_def.default_factory()  # type: ignore[misc]
    return None


@dataclass(slots=True)
class AppConfig:
    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    sessions_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "sessions")
    reports_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "reports")
    logs_dir: Path = field(default_factory=lambda: Path.cwd() / "logs")
    config_dir: Path = field(default_factory=lambda: Path.cwd() / "configs")

    # Use a valid public HF repo id by default.
    llm_model_name: str = "Qwen/Qwen3.5-9B"
    embedding_model_name: str = "Qwen/Qwen3-Embedding-4B"
    fallback_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Force vLLM first.
    use_vllm: bool = True
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

    similarity_use_gpu: bool = True
    similarity_gpu_mode: str = "auto"
    similarity_max_workers: int = 0
    similarity_parallel_lexical: bool = True
    similarity_parallel_heading: bool = True
    similarity_gpu_cosine: bool = True

    omp_num_threads: int = 0
    mkl_num_threads: int = 0
    openblas_num_threads: int = 0
    numexpr_max_threads: int = 0

    @classmethod
    def load(cls, base_dir: str | os.PathLike[str] | None = None) -> "AppConfig":
        project_root = Path(base_dir or Path.cwd()).resolve()

        # Prefer the repository root that contains configs/ and .env if available.
        candidate = Path(__file__).resolve().parents[2]
        if (candidate / "configs").exists():
            project_root = candidate

        _bootstrap_env(project_root)

        config_dir = project_root / "configs"
        app_toml = _read_toml(config_dir / "app.toml")
        env = os.environ

        def _env_or_toml(env_key: str, toml_key: str, field_name: str) -> Any:
            value = env.get(env_key)
            if value is not None and str(value).strip() != "":
                return value
            if toml_key in app_toml and app_toml[toml_key] is not None:
                return app_toml[toml_key]
            return _default_value(cls, field_name)

        hf_token = (
            env.get("HF_TOKEN")
            or env.get("HUGGINGFACE_HUB_TOKEN")
            or app_toml.get("hf_token")
            or app_toml.get("huggingface_token")
        )

        cfg = cls(
            project_root=project_root,
            data_dir=project_root / "data",
            sessions_dir=project_root / "data" / "sessions",
            reports_dir=project_root / "data" / "reports",
            logs_dir=project_root / "logs",
            config_dir=config_dir,
            llm_model_name=str(
                _env_or_toml("LLM_MODEL", "llm_model_name", "llm_model_name")
            ),
            embedding_model_name=str(
                _env_or_toml(
                    "EMBEDDING_MODEL", "embedding_model_name", "embedding_model_name"
                )
            ),
            fallback_embedding_model_name=str(
                _env_or_toml(
                    "FALLBACK_EMBEDDING_MODEL",
                    "fallback_embedding_model_name",
                    "fallback_embedding_model_name",
                )
            ),
            use_vllm=_as_bool(_env_or_toml("USE_VLLM", "use_vllm", "use_vllm")),
            temperature=float(
                _env_or_toml("TEMPERATURE", "temperature", "temperature")
            ),
            max_new_tokens=int(
                _env_or_toml("MAX_NEW_TOKENS", "max_new_tokens", "max_new_tokens")
            ),
            top_k_candidates=int(
                _env_or_toml("TOP_K_CANDIDATES", "top_k_candidates", "top_k_candidates")
            ),
            min_similarity=float(
                _env_or_toml("MIN_SIMILARITY", "min_similarity", "min_similarity")
            ),
            unchanged_threshold=float(
                _env_or_toml(
                    "UNCHANGED_THRESHOLD", "unchanged_threshold", "unchanged_threshold"
                )
            ),
            modified_threshold=float(
                _env_or_toml(
                    "MODIFIED_THRESHOLD", "modified_threshold", "modified_threshold"
                )
            ),
            batch_size=int(_env_or_toml("BATCH_SIZE", "batch_size", "batch_size")),
            chunk_chars=int(_env_or_toml("CHUNK_CHARS", "chunk_chars", "chunk_chars")),
            neighbors_window=int(
                _env_or_toml("NEIGHBORS_WINDOW", "neighbors_window", "neighbors_window")
            ),
            keep_intermediate_json=_as_bool(
                _env_or_toml(
                    "KEEP_INTERMEDIATE_JSON",
                    "keep_intermediate_json",
                    "keep_intermediate_json",
                )
            ),
            hf_token=str(hf_token) if hf_token else None,
            similarity_use_gpu=_as_bool(
                _env_or_toml(
                    "SIMILARITY_USE_GPU", "similarity_use_gpu", "similarity_use_gpu"
                )
            ),
            similarity_gpu_mode=str(
                _env_or_toml(
                    "SIMILARITY_GPU_MODE", "similarity_gpu_mode", "similarity_gpu_mode"
                )
            ),
            similarity_max_workers=int(
                _env_or_toml(
                    "SIMILARITY_MAX_WORKERS",
                    "similarity_max_workers",
                    "similarity_max_workers",
                )
            ),
            similarity_parallel_lexical=_as_bool(
                _env_or_toml(
                    "SIMILARITY_PARALLEL_LEXICAL",
                    "similarity_parallel_lexical",
                    "similarity_parallel_lexical",
                )
            ),
            similarity_parallel_heading=_as_bool(
                _env_or_toml(
                    "SIMILARITY_PARALLEL_HEADING",
                    "similarity_parallel_heading",
                    "similarity_parallel_heading",
                )
            ),
            similarity_gpu_cosine=_as_bool(
                _env_or_toml(
                    "SIMILARITY_GPU_COSINE",
                    "similarity_gpu_cosine",
                    "similarity_gpu_cosine",
                )
            ),
            omp_num_threads=int(
                _env_or_toml("OMP_NUM_THREADS", "omp_num_threads", "omp_num_threads")
            ),
            mkl_num_threads=int(
                _env_or_toml("MKL_NUM_THREADS", "mkl_num_threads", "mkl_num_threads")
            ),
            openblas_num_threads=int(
                _env_or_toml(
                    "OPENBLAS_NUM_THREADS",
                    "openblas_num_threads",
                    "openblas_num_threads",
                )
            ),
            numexpr_max_threads=int(
                _env_or_toml(
                    "NUMEXPR_MAX_THREADS", "numexpr_max_threads", "numexpr_max_threads"
                )
            ),
        )
        cfg.ensure_dirs()
        cfg._apply_runtime_env()
        return cfg

    def ensure_dirs(self) -> None:
        for path in [
            self.data_dir,
            self.sessions_dir,
            self.reports_dir,
            self.logs_dir,
            self.config_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _apply_runtime_env(self) -> None:
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", self.hf_token)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_default_session_dir(cfg: AppConfig, session_id: str) -> Path:
    path = cfg.sessions_dir / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path
