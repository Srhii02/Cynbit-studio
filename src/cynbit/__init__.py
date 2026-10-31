"""Cynbit Studio – AI Resume & Job Matching Intelligence."""
from __future__ import annotations

from dataclasses import dataclass

__version__ = "1.0.0"


@dataclass
class FullAnalysis:
    profile: object
    pairs: list
    recommendations: list
    quality: float
    suggestions: list
    readiness: object


def analyze(resume_text: str, job_texts: list[str], hours_per_week: int = 8, engine=None,
            plan_top_k: int = 3) -> FullAnalysis:
    """One-call pipeline used by the UI, the CLI and tests.

    The learning plan targets the `plan_top_k` best-matching jobs (the most realistic targets), so an ML student
    is not told to learn CSS just because a front-end JD was also in the comparison."""
    from .extractor import build_profile
    from .ranking import rank_jobs
    from .readiness import compute_readiness
    from .recommender import recommend
    from .suggestions import analyze_resume

    profile = build_profile(resume_text)
    pairs = rank_jobs(profile, job_texts, engine)
    results = [r for _, r in pairs]
    missing_kw = [s.skill for s in results[0].missing if s.importance == "required"] if results else []
    quality, suggestions = analyze_resume(profile, missing_kw)
    recs = recommend(results[:plan_top_k], hours_per_week=hours_per_week)
    ready = compute_readiness(profile, results, quality)
    return FullAnalysis(profile, pairs, recs, quality, suggestions, ready)
