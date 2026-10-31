"""Semantic similarity engines.

Primary: Sentence Transformers `all-MiniLM-L6-v2` (384-d, cosine similarity on normalised embeddings).
Fallback: TF-IDF + LSA with skill-concept normalisation, used only when the transformer model cannot be loaded
(e.g. offline machine). Both produce *real* cosine similarities; they are calibrated separately because their
raw similarity distributions differ.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache

import numpy as np

from .config import settings
from .skills import get_taxonomy

log = logging.getLogger(__name__)


def _chunk(text: str, max_words: int = 90) -> list[str]:
    """Group lines into ~max_words chunks (MiniLM truncates at 256 word-pieces)."""
    chunks, cur, n = [], [], 0
    for line in [ln.strip() for ln in text.split("\n") if ln.strip()]:
        w = len(line.split())
        if n + w > max_words and cur:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text[:1000]]


class BaseEngine:
    name = "base"
    is_transformer = False
    # Linear calibration of raw cosine -> [0, 1] (low maps to 0, high maps to 1).
    doc_low, doc_high = 0.0, 1.0
    sent_low, sent_high = 0.0, 1.0

    def similarity_matrix(self, a: list[str], b: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def doc_similarity(self, a: str, b: str) -> float:
        raise NotImplementedError

    def calibrate(self, sim: float | np.ndarray, kind: str = "sent"):
        lo, hi = (self.doc_low, self.doc_high) if kind == "doc" else (self.sent_low, self.sent_high)
        return np.clip((np.asarray(sim) - lo) / (hi - lo), 0.0, 1.0)


class TransformerEngine(BaseEngine):
    is_transformer = True
    # Empirical ranges for all-MiniLM-L6-v2: unrelated text ~0.0-0.2, same-domain ~0.4-0.8.
    doc_low, doc_high = 0.20, 0.75
    sent_low, sent_high = 0.20, 0.75

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.name = f"Sentence Transformers · {model_name.split('/')[-1]}"
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        todo = [t for t, k in zip(texts, keys) if k not in self._cache]
        if todo:
            vecs = self.model.encode(todo, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
            for t, v in zip(todo, vecs):
                self._cache[hashlib.sha1(t.encode()).hexdigest()] = v
        if len(self._cache) > 20_000:
            self._cache.clear()
        return np.vstack([self._cache[k] for k in keys]) if keys else np.zeros((0, 384))

    def similarity_matrix(self, a, b):
        if not a or not b:
            return np.zeros((len(a), len(b)))
        return self.encode(a) @ self.encode(b).T

    def doc_embedding(self, text: str) -> np.ndarray:
        v = self.encode(_chunk(text)).mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-12)

    def doc_similarity(self, a, b):
        return float(self.doc_embedding(a) @ self.doc_embedding(b))


class TfidfLsaEngine(BaseEngine):
    """Offline fallback: TF-IDF (word 1-2 grams) blended with LSA topics, plus canonical skill tokens so that
    aliases ("sklearn" vs "scikit-learn", "k8s" vs "Kubernetes") match each other."""

    name = "TF-IDF + LSA (offline fallback)"
    doc_low, doc_high = 0.05, 0.55
    sent_low, sent_high = 0.05, 0.55

    def __init__(self):
        tax = get_taxonomy()
        # Background corpus: one line per taxonomy category, so LSA learns which skills co-occur in a domain.
        by_cat: dict[str, list[str]] = {}
        for s in tax.skills.values():
            by_cat.setdefault(s.category, []).append(" ".join([s.name, *s.aliases, self._token(s.name)]))
        self.background = [" ".join(v) for v in by_cat.values()]
        self.background += [" ".join([s.name, *s.aliases, *s.related]) for s in tax.skills.values()]

    @staticmethod
    def _token(name: str) -> str:
        return "skill_" + "".join(ch for ch in name.lower() if ch.isalnum())

    def _normalise(self, text: str) -> str:
        hits = get_taxonomy().extract(text, with_evidence=False)
        return text + " " + " ".join(self._token(n) for n in hits for _ in range(min(hits[n].count, 3)))

    def similarity_matrix(self, a, b):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        if not a or not b:
            return np.zeros((len(a), len(b)))
        docs = [self._normalise(t) for t in a + b]
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1,
                              token_pattern=r"(?u)\b[\w+#./]{2,}\b")
        X = vec.fit_transform(self.background + docs)
        Xa, Xb = X[len(self.background):len(self.background) + len(a)], X[len(self.background) + len(a):]
        tfidf_sim = (Xa @ Xb.T).toarray()
        k = min(60, X.shape[0] - 1, X.shape[1] - 1)
        if k >= 2:
            Z = normalize(TruncatedSVD(n_components=k, random_state=42).fit_transform(X))
            Za, Zb = Z[len(self.background):len(self.background) + len(a)], Z[len(self.background) + len(a):]
            lsa_sim = np.clip(Za @ Zb.T, 0, 1)
            return 0.7 * tfidf_sim + 0.3 * lsa_sim
        return tfidf_sim

    def doc_similarity(self, a, b):
        return float(self.similarity_matrix([a], [b])[0, 0])


@lru_cache(maxsize=2)
def get_engine(backend: str | None = None) -> BaseEngine:
    backend = (backend or settings.embedding_backend).lower()
    if backend in {"auto", "transformer"}:
        try:
            return TransformerEngine(settings.embedding_model)
        except Exception as exc:
            if backend == "transformer":
                raise RuntimeError(f"Could not load {settings.embedding_model}: {exc}") from exc
            log.warning("Sentence Transformers model unavailable (%s). Falling back to TF-IDF + LSA.", exc)
    return TfidfLsaEngine()
