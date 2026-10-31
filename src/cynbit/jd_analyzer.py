"""Job description analysis: title, seniority, required vs preferred skills, experience and degree needs."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import nlp as nlpmod
from .extractor import DEGREE_PATTERNS
from .skills import SkillHit, extract_skills

_PREFERRED_HEADERS = re.compile(
    r"^(nice[\s-]to[\s-]have|preferred|bonus|good[\s-]to[\s-]have|plus|desirable|preferred qualifications|"
    r"additional|optional)\b", re.I)
_REQUIRED_HEADERS = re.compile(
    r"^(requirements?|required|must[\s-]have|qualifications|minimum qualifications|basic qualifications|"
    r"what you('| wi)ll need|what we('| a)re looking for|skills|who you are|eligibility)\b", re.I)
_RESP_HEADERS = re.compile(r"^(responsibilities|what you('| wi)ll do|role|key responsibilities|the role|your role|duties)\b", re.I)
_PREFERRED_CUES = re.compile(r"\b(nice to have|preferred|is a plus|a plus|bonus|good to have|desirable|familiarity with|exposure to)\b", re.I)
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:-|–|to)?\s*(\d+)?\s*\+?\s*(?:years?|yrs?)", re.I)

SENIORITY_KEYWORDS = [
    ("Intern", re.compile(r"\b(intern|internship|trainee)\b", re.I)),
    ("Entry level", re.compile(r"\b(entry[\s-]level|fresher|graduate|junior|jr\.?|associate)\b", re.I)),
    ("Senior", re.compile(r"\b(senior|sr\.?|lead|principal|staff|architect)\b", re.I)),
    ("Mid level", re.compile(r"\b(mid[\s-]level|intermediate|ii|iii)\b", re.I)),
]


@dataclass
class JobProfile:
    text: str
    title: str
    company: str
    seniority: str
    required_skills: dict[str, SkillHit]
    preferred_skills: dict[str, SkillHit]
    min_years: float
    degree_level: int
    degree_label: str
    responsibilities: list[str]
    requirements: list[str]
    key_phrases: list[str] = field(default_factory=list)

    @property
    def all_skills(self) -> list[str]:
        return list(self.required_skills) + [s for s in self.preferred_skills if s not in self.required_skills]


def _title_and_company(text: str) -> tuple[str, str]:
    title, company = "", ""
    for line in [ln.strip() for ln in text.split("\n") if ln.strip()][:6]:
        m = re.match(r"^(job\s*title|title|role|position)\s*[:\-]\s*(.+)$", line, re.I)
        if m and not title:
            title = m.group(2).strip()
            continue
        m = re.match(r"^(company|organisation|organization|employer)\s*[:\-]\s*(.+)$", line, re.I)
        if m and not company:
            company = m.group(2).strip()
            continue
        if not title and len(line.split()) <= 10 and not line.endswith("."):
            title = line
    if not company:
        m = re.search(r"\bat\s+([A-Z][\w&.]*(?:\s+[A-Z][\w&.]*){0,3})", text[:400])
        company = m.group(1) if m else ""
    return title[:90] or "Untitled role", company[:60]


def _seniority(title: str, text: str) -> str:
    for label, pat in SENIORITY_KEYWORDS:
        if pat.search(title):
            return label
    for label, pat in SENIORITY_KEYWORDS[:3]:
        if pat.search(text[:600]):
            return label
    return "Not specified"


def _segment(text: str) -> tuple[list[str], list[str], list[str]]:
    """Return (requirement lines, preferred lines, responsibility lines)."""
    req, pref, resp, other = [], [], [], []
    bucket = other
    for raw in text.split("\n"):
        line = raw.strip().lstrip("- ").strip()
        if not line:
            continue
        header = line.rstrip(":").strip()
        if len(header.split()) <= 6:
            if _PREFERRED_HEADERS.match(header):
                bucket = pref
                continue
            if _RESP_HEADERS.match(header):
                bucket = resp
                continue
            if _REQUIRED_HEADERS.match(header):
                bucket = req
                continue
        # Apply "nice to have" cues per sentence so one long paragraph isn't marked preferred wholesale.
        for sent in [x for x in re.split(r"(?<=[.;!?])\s+", line) if x.strip()]:
            if bucket is not pref and _PREFERRED_CUES.search(sent):
                pref.append(sent)
            else:
                bucket.append(sent)
    if not req:  # unstructured JD: treat everything not preferred as requirements
        req = resp + other
    return req, pref, resp


def analyze_job(text: str) -> JobProfile:
    title, company = _title_and_company(text)
    req_lines, pref_lines, resp_lines = _segment(text)
    all_hits = extract_skills(text)
    pref_hits = extract_skills("\n".join(pref_lines))
    req_hits_scope = extract_skills("\n".join(req_lines + resp_lines))
    required = {k: all_hits[k] for k in all_hits if k in req_hits_scope or k not in pref_hits}
    preferred = {k: all_hits[k] for k in pref_hits if k not in required}

    years = 0.0
    for line in req_lines + pref_lines + [text[:500]]:
        if re.search(r"experience|exp\b", line, re.I):
            m = _YEARS_RE.search(line)
            if m:
                years = max(years, float(m.group(1)))
                break
    level, label = 0, "Not specified"
    # Degrees are usually listed as "Bachelor's or Master's": the requirement is the minimum mentioned.
    for lvl, lab, pat in reversed(DEGREE_PATTERNS):
        if pat.search(text):
            level, label = lvl, lab
            break

    return JobProfile(
        text=text, title=title, company=company, seniority=_seniority(title, text),
        required_skills=required, preferred_skills=preferred, min_years=years,
        degree_level=level, degree_label=label,
        responsibilities=resp_lines[:12], requirements=(req_lines + pref_lines)[:20],
        key_phrases=nlpmod.key_phrases(text, 12),
    )
