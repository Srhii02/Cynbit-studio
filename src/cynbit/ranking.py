"""Multi-job comparison and ranking."""
from __future__ import annotations

import pandas as pd

from .embeddings import BaseEngine, get_engine
from .extractor import ResumeProfile
from .jd_analyzer import JobProfile, analyze_job
from .matcher import MatchResult, match


def rank_jobs(profile: ResumeProfile, job_texts: list[str], engine: BaseEngine | None = None
              ) -> list[tuple[JobProfile, MatchResult]]:
    engine = engine or get_engine()
    pairs = []
    for text in job_texts:
        if text and text.strip():
            job = analyze_job(text)
            pairs.append((job, match(profile, job, engine)))
    pairs.sort(key=lambda p: -p[1].overall)
    return pairs


def ranking_table(pairs: list[tuple[JobProfile, MatchResult]]) -> pd.DataFrame:
    rows = []
    for rank, (job, r) in enumerate(pairs, start=1):
        req = [s for s in r.skill_matches if s.importance == "required"]
        rows.append({
            "Rank": rank, "Role": r.job_title, "Company": r.company or "—", "Seniority": job.seniority,
            "Overall": r.overall, "Semantic": r.semantic, "Skills": r.skills, "Experience": r.experience,
            "Education": r.education, "Verdict": r.verdict,
            "Required matched": f"{sum(s.status == 'matched' for s in req)}/{len(req)}",
            "Top gaps": ", ".join(s.skill for s in r.missing[:4]) or "—",
        })
    return pd.DataFrame(rows)


def skill_matrix(pairs: list[tuple[JobProfile, MatchResult]]) -> pd.DataFrame:
    """Skills (rows) × jobs (cols): 1 matched, 0.5 related, 0 missing, NaN not requested."""
    data: dict[str, dict[str, float]] = {}
    for job, r in pairs:
        col = r.job_title[:28]
        for sm in r.skill_matches:
            data.setdefault(sm.skill, {})[col] = {"matched": 1.0, "related": 0.5, "missing": 0.0}[sm.status]
    df = pd.DataFrame(data).T
    if df.empty:
        return df
    df["_demand"] = df.notna().sum(axis=1)
    return df.sort_values("_demand", ascending=False).drop(columns="_demand")
