import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SAMPLES = ROOT / "samples"


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    return SAMPLES


@pytest.fixture(scope="session")
def resume_texts() -> dict[str, str]:
    from cynbit.parser import parse_file

    return {p.stem: parse_file(p).text for p in sorted((SAMPLES / "resumes").glob("*")) if p.suffix in {".pdf", ".docx"}}


@pytest.fixture(scope="session")
def job_texts() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted((SAMPLES / "jobs").glob("*.txt"))}


@pytest.fixture(scope="session")
def engine():
    """Whatever engine the environment supports (MiniLM if available, else TF-IDF+LSA)."""
    from cynbit.embeddings import get_engine

    return get_engine()


@pytest.fixture(scope="session")
def tfidf_engine():
    from cynbit.embeddings import get_engine

    return get_engine("tfidf")


@pytest.fixture(scope="session")
def transformer_engine():
    from cynbit.embeddings import TransformerEngine

    try:
        return TransformerEngine("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:  # offline / model not cached
        pytest.skip(f"all-MiniLM-L6-v2 not available: {exc}")
