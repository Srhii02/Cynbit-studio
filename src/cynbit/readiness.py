"""Career readiness index combining market fit, skills, resume quality, experience and completeness."""
from __future__ import annotations

from dataclasses import dataclass, field

from .extractor import ResumeProfile
from .matcher import MatchResult
from .skills import get_taxonomy

WEIGHTS = {"Market fit": 0.35, "Skill depth": 0.20, "Resume quality": 0.20, "Practical experience": 0.15,
           "Profile completeness": 0.10}


@dataclass
class Readiness:
    index: float
    level: str
    components: dict[str, float]
    category_coverage: dict[str, float] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=list)


def level_for(index: float) -> str:
    if index >= 75:
        return "Job-ready"
    if index >= 60:
        return "Almost ready"
    if index >= 45:
        return "Developing"
    return "Early stage"


def compute_readiness(profile: ResumeProfile, results: list[MatchResult], quality_score: float) -> Readiness:
    top = sorted((r.overall for r in results), reverse=True)[:3]
    market = sum(top) / len(top) if top else 0.0

    tax = get_taxonomy()
    tech_cats = {s.category for n, s in tax.skills.items() if n in profile.skills and s.category != "Soft Skills"}
    evidenced = sum(1 for h in profile.skills.values() if h.count >= 2)
    depth = min(100.0, 6 * len(profile.skills) * 0.5 + 8 * len(tech_cats) * 0.5 + 4 * evidenced)

    projects = profile.sections.get("projects", "")
    n_projects = max(projects.count("\n- ") + projects.startswith("- "), 0) if projects else 0
    practical = min(100.0, 30 * min(profile.experience_years, 2) + 15 * min(profile.internships, 2)
                    + 10 * min(n_projects, 4) + (10 if profile.quantified_bullets else 0))

    completeness_items = [profile.contact["email"], profile.contact["linkedin"], profile.contact["github"],
                          "skills" in profile.sections, "education" in profile.sections,
                          "projects" in profile.sections or "experience" in profile.sections,
                          "summary" in profile.sections]
    completeness = 100 * sum(completeness_items) / len(completeness_items)

    comps = {"Market fit": round(market, 1), "Skill depth": round(depth, 1), "Resume quality": round(quality_score, 1),
             "Practical experience": round(practical, 1), "Profile completeness": round(completeness, 1)}
    index = round(sum(WEIGHTS[k] * v for k, v in comps.items()), 1)

    # Coverage per category across all requested skills
    cov: dict[str, list[float]] = {}
    for r in results:
        for sm in r.skill_matches:
            cov.setdefault(sm.category, []).append({"matched": 1.0, "related": 0.5, "missing": 0.0}[sm.status])
    category_coverage = {k: round(100 * sum(v) / len(v), 1) for k, v in sorted(cov.items())}

    strengths = [k for k, v in comps.items() if v >= 70]
    focus = [k for k, v in sorted(comps.items(), key=lambda kv: kv[1]) if v < 60]
    return Readiness(index, level_for(index), comps, category_coverage, strengths, focus)
