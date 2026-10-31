"""Pre-download the NLP models so the app works offline afterwards.

    python scripts/download_models.py
"""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    ok = True
    try:
        import spacy

        spacy.load("en_core_web_sm")
        print("✓ spaCy en_core_web_sm already installed")
    except Exception:
        print("… installing spaCy en_core_web_sm")
        ok &= subprocess.call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"]) == 0
    try:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        v = m.encode(["hello world"], normalize_embeddings=True)
        print(f"✓ all-MiniLM-L6-v2 ready (dim={v.shape[1]})")
    except Exception as exc:
        ok = False
        print(f"✗ Could not download all-MiniLM-L6-v2: {exc}\n  The app will use the TF-IDF + LSA fallback.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
