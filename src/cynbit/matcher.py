"""Resume <-> job matching: semantic similarity + skill coverage + experience + education, with explanations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import settings
from .embeddings import BaseEngine, get_engine
from .extractor import ResumeProfile
from .jd_analyzer import JobProfile
from .nlp import sentences
from .skills import get_taxonomy

REQUIRED_WEIGHT, PREFERRED_WEIGHT = 2.0, 1.0
RELATED_CREDIT = 0.5


@dataclass
class SkillMatch:
    skill: str
    category: str
    importance: str  # "required" | "preferred"
    status: str  # "matched" | "related" | "missing"
    via: str = ""  # resume skill giving related credit
    evidence: str = ""


@dataclass
class Alignment:
    requirement: str
    evidence: str
    similarity: float  # calibrated 0-1


@dataclass
class MatchResult:
    job_title: str
    company: str
    overall: float
    semantic: float
    skills: float
    experience: float
    education: float
    verdict: str
    engine: str
    raw_doc_similarity: float
    skill_matches: list[SkillMatch] = field(default_factory=list)
    alignments: list[Alignment] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)

    @property
    def missing(self) -> list[SkillMatch]:
        return [s for s in self.skill_matches if s.status == "missing"]

    @property
    def matched(self) -> list[SkillMatch]:
        return [s for s in self.skill_matches if s.status == "matched"]

    @property
    def related(self) -> list[SkillMatch]:
        return [s for s in self.skill_matches if s.status == "related"]

    def to_dict(self) -> dict:
        return asdict(self)


def verdict_for(score: float) -> str:
    if score >= 75:
        return "Strong match"
    if score >= 60:
        return "Good match"
    if score >= 45:
        return "Moderate match"
    return "Low match"


def _resume_units(profile: ResumeProfile) -> list[str]:
    units = list(dict.fromkeys(profile.bullets + sentences(profile.text)))
    units = [u for u in units if 4 <= len(u.split()) <= 80]
    return units[:150] or [profile.text[:1000]]


def _job_units(job: JobProfile) -> list[str]:
    units = [u for u in job.requirements + job.responsibilities if len(u.split()) >= 3]
    return list(dict.fromkeys(units))[:30] or sentences(job.text)[:30]


def _skill_coverage(profile: ResumeProfile, job: JobProfile, engine: BaseEngine) -> tuple[float, list[SkillMatch]]:
    tax = get_taxonomy()
    have = set(profile.skills)
    results: list[SkillMatch] = []
    pending: list[SkillMatch] = []
    for importance, bucket in (("required", job.required_skills), ("preferred", job.preferred_skills)):
        for name, hit in bucket.items():
            sk = tax.get(name)
            if name in have:
                ev = profile.skills[name].evidence[0] if profile.skills[name].evidence else ""
                results.append(SkillMatch(name, hit.category, importance, "matched", evidence=ev))
                continue
            related = [r for r in (sk.related if sk else ()) if r in have]
            related += [o.name for o in tax.skills.values() if name in o.related and o.name in have]
            if related:
                results.append(SkillMatch(name, hit.category, importance, "related", via=related[0]))
            else:
                sm = SkillMatch(name, hit.category, importance, "missing")
                results.append(sm)
                pending.append(sm)
    # Transformer only: semantic relatedness between a missing skill and the candidate's skills.
    if engine.is_transformer and pending and have:
        have_l = sorted(have)
        sims = engine.similarity_matrix([p.skill for p in pending], have_l)
        for i, sm in enumerate(pending):
            j = int(np.argmax(sims[i]))
            if sims[i, j] >= 0.62 and tax.get(have_l[j]) and tax.get(have_l[j]).category == sm.category:
                sm.status, sm.via = "related", have_l[j]
    total = sum(REQUIRED_WEIGHT if s.importance == "required" else PREFERRED_WEIGHT for s in results)
    if total == 0:
        return -1.0, results
    credit = sum((REQUIRED_WEIGHT if s.importance == "required" else PREFERRED_WEIGHT)
                 * (1.0 if s.status == "matched" else RELATED_CREDIT if s.status == "related" else 0.0)
                 for s in results)
    order = {"required": 0, "preferred": 1}
    results.sort(key=lambda s: (order[s.importance], {"matched": 0, "related": 1, "missing": 2}[s.status], s.skill))
    return credit / total, results


def _experience_score(profile: ResumeProfile, job: JobProfile) -> float:
    years = profile.experience_years + (0.25 * profile.internships if profile.experience_years == 0 else 0)
    if job.min_years <= 0:
        return 1.0
    return float(min(1.0, years / job.min_years))


def _education_score(profile: ResumeProfile, job: JobProfile) -> float:
    if job.degree_level == 0:
        return 1.0
    if profile.education_level == 0:
        return 0.5
    if profile.education_level >= job.degree_level:
        return 1.0
    return 0.6 if job.degree_level - profile.education_level == 1 else 0.3


def match(profile: ResumeProfile, job: JobProfile, engine: BaseEngine | None = None) -> MatchResult:
    engine = engine or get_engine()

    # 1) Semantic similarity: whole-document + requirement-level alignment
    raw_doc = engine.doc_similarity(profile.text, job.text)
    doc_score = float(engine.calibrate(raw_doc, "doc"))
    r_units, j_units = _resume_units(profile), _job_units(job)
    sim = engine.similarity_matrix(j_units, r_units)
    best_idx = sim.argmax(axis=1)
    best = sim[np.arange(len(j_units)), best_idx]
    best_cal = engine.calibrate(best, "sent")
    alignments = [Alignment(j_units[i], r_units[best_idx[i]], float(best_cal[i])) for i in range(len(j_units))]
    alignments.sort(key=lambda a: -a.similarity)
    req_coverage = float(best_cal.mean()) if len(best_cal) else doc_score
    semantic = 0.5 * doc_score + 0.5 * req_coverage

    # 2) Skills, 3) experience, 4) education
    skill_cov, skill_matches = _skill_coverage(profile, job, engine)
    if skill_cov < 0:  # JD without recognisable skills -> rely on semantics
        skill_cov = semantic
    exp = _experience_score(profile, job)
    edu = _education_score(profile, job)

    overall = 100 * (settings.w_semantic * semantic + settings.w_skills * skill_cov
                     + settings.w_experience * exp + settings.w_education * edu)
    result = MatchResult(
        job_title=job.title, company=job.company, overall=round(overall, 1), semantic=round(100 * semantic, 1),
        skills=round(100 * skill_cov, 1), experience=round(100 * exp, 1), education=round(100 * edu, 1),
        verdict=verdict_for(overall), engine=engine.name, raw_doc_similarity=round(raw_doc, 4),
        skill_matches=skill_matches, alignments=alignments,
    )
    result.explanations = explain(result, profile, job)
    return result


def explain(r: MatchResult, profile: ResumeProfile, job: JobProfile) -> list[str]:
    out = []
    req = [s for s in r.skill_matches if s.importance == "required"]
    req_hit = [s for s in req if s.status == "matched"]
    out.append(
        f"Semantic fit is {r.semantic:.0f}/100 (raw document cosine {r.raw_doc_similarity:.3f} using {r.engine}); "
        f"it blends whole-document similarity with how well each job requirement is covered by a resume line."
    )
    if req:
        out.append(f"You show {len(req_hit)} of {len(req)} required skills"
                   + (f" (e.g. {', '.join(s.skill for s in req_hit[:4])})." if req_hit else "."))
    rel = r.related
    if rel:
        out.append("Partial credit for adjacent skills: "
                   + "; ".join(f"{s.skill} ← you have {s.via}" for s in rel[:4]) + ".")
    miss_req = [s.skill for s in r.missing if s.importance == "required"]
    if miss_req:
        out.append(f"Biggest gaps (required): {', '.join(miss_req[:6])}.")
    if job.min_years > 0:
        out.append(f"The role asks for {job.min_years:g}+ years; about {profile.experience_years:g} years detected "
                   f"from dated roles" + (f" plus {profile.internships} internship mention(s)." if profile.internships else "."))
    if job.degree_level:
        out.append(f"Education: role expects {job.degree_label}; detected {profile.education_label}.")
    strong = [a for a in r.alignments if a.similarity >= 0.6][:2]
    for a in strong:
        out.append(f"Strong evidence: “{a.requirement[:90]}” ↔ “{a.evidence[:90]}” ({a.similarity:.0%}).")
    weak = [a for a in r.alignments if a.similarity < 0.3][-2:]
    for a in weak:
        out.append(f"Weakly covered requirement: “{a.requirement[:110]}”.")
    return out
