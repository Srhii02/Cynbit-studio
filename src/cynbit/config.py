"""Central configuration. Values come from environment variables (optionally a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = PROJECT_ROOT / "samples"


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv("CYNBIT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # "auto" -> try Sentence Transformers, fall back to TF-IDF/LSA. "transformer" -> require it. "tfidf" -> force fallback.
    embedding_backend: str = os.getenv("CYNBIT_EMBEDDING_BACKEND", "auto").lower()
    spacy_model: str = os.getenv("CYNBIT_SPACY_MODEL", "en_core_web_sm")
    db_path: Path = Path(os.getenv("CYNBIT_DB_PATH", str(PROJECT_ROOT / ".data" / "cynbit_studio.db")))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    store_history: bool = _env_bool("CYNBIT_STORE_HISTORY", True)
    max_upload_mb: int = int(os.getenv("CYNBIT_MAX_UPLOAD_MB", "5"))

    # Score weights (must sum to 1.0)
    w_semantic: float = 0.40
    w_skills: float = 0.35
    w_experience: float = 0.15
    w_education: float = 0.10


settings = Settings()
