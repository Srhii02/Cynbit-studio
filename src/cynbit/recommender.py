"""Personalised learning recommendations from skill gaps across one or more target jobs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

from .config import DATA_DIR
from .matcher import MatchResult
from .skills import get_taxonomy


@dataclass
class Recommendation:
    skill: str
    category: str
    priority: float
    importance: str
    demanded_by: list[str]
    est_hours: int
    status: str  # missing | related
    via: str
    resources: list[dict] = field(default_factory=list)
    project: str = ""
    week_start: int = 0
    week_end: int = 0

    @property
    def priority_label(self) -> str:
        return "High" if self.priority >= 3 else "Medium" if self.priority >= 1.5 else "Low"


@lru_cache(maxsize=1)
def _resources() -> dict:
    return json.loads((DATA_DIR / "learning_resources.json").read_text(encoding="utf-8"))


def resources_for(skill: str, category: str) -> tuple[list[dict], str]:
    data = _resources()
    entry = data["skills"].get(skill)
    if entry:
        return entry["resources"], entry.get("project", "")
    default = data["category_defaults"].get(category)
    project = f"Build a small, documented project that uses {skill} and add it to your resume with a measurable outcome."
    return ([default] if default else []), project


def recommend(results: list[MatchResult], hours_per_week: int = 8, top_n: int = 12) -> list[Recommendation]:
    """Aggregate gaps across jobs. Priority = Σ(job importance weight × gap weight); better-matching jobs weigh more
    because they are the most realistic targets."""
    tax = get_taxonomy()
    agg: dict[str, Recommendation] = {}
    for r in results:
        job_weight = 0.5 + r.overall / 100  # 0.5 – 1.5
        for sm in r.skill_matches:
            if sm.status == "matched":
                continue
            w = (2.0 if sm.importance == "required" else 1.0) * (0.5 if sm.status == "related" else 1.0) * job_weight
            sk = tax.get(sm.skill)
            hours = sk.hours if sk else 20
            if sm.status == "related":
                hours = max(4, hours // 2)
            rec = agg.get(sm.skill)
            if rec is None:
                res, proj = resources_for(sm.skill, sm.category)
                rec = Recommendation(sm.skill, sm.category, 0.0, sm.importance, [], hours, sm.status, sm.via, res, proj)
                agg[sm.skill] = rec
            rec.priority += w
            if sm.importance == "required":
                rec.importance = "required"
            if sm.status == "missing":
                rec.status, rec.est_hours = "missing", sk.hours if sk else hours
            label = r.job_title if not r.company else f"{r.job_title} ({r.company})"
            if label not in rec.demanded_by:
                rec.demanded_by.append(label)
    recs = sorted(agg.values(), key=lambda x: (-round(x.priority, 3), x.est_hours, x.skill))[:top_n]
    # Schedule sequentially into weeks
    week_hours, cursor = max(1, hours_per_week), 0.0
    for rec in recs:
        rec.priority = round(rec.priority, 2)
        rec.week_start = int(cursor // week_hours) + 1
        cursor += rec.est_hours
        rec.week_end = max(rec.week_start, int((cursor - 1e-9) // week_hours) + 1)
    return recs
