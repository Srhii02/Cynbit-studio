"""Resume information extraction: sections, contact signals, experience, education, skills."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import nlp as nlpmod
from .experience import ExperienceEntry, extract_entries, find_ranges, merged_months
from .skills import SkillHit, extract_skills

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "professional summary", "profile", "objective", "career objective", "about me", "about"),
    "experience": ("experience", "work experience", "professional experience", "employment", "employment history",
                   "internships", "internship", "internship experience", "work history", "relevant experience",
                   "industry experience", "experience & internships", "work experience & internships",
                   "internships & experience", "internship & training", "industrial training", "training & internships",
                   "career history", "professional background"),
    "education": ("education", "academic background", "academics", "education and training", "qualifications"),
    "skills": ("skills", "technical skills", "core skills", "key skills", "skills summary", "technologies",
               "tech stack", "tools", "core competencies"),
    "projects": ("projects", "academic projects", "personal projects", "key projects", "project experience",
                 "selected projects"),
    "certifications": ("certifications", "certificates", "licenses", "courses", "certifications & courses"),
    "achievements": ("achievements", "awards", "honors", "honours", "accomplishments", "awards & achievements"),
    "publications": ("publications", "research", "papers"),
    "activities": ("activities", "extracurricular", "extra-curricular", "leadership", "volunteering",
                   "positions of responsibility"),
}
_HEADER_LOOKUP = {alias: canon for canon, aliases in SECTION_ALIASES.items() for alias in aliases}

_YEARS_STATED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional\s+|work\s+|industry\s+)?experience",
                              re.IGNORECASE)

DEGREE_PATTERNS: list[tuple[int, str, re.Pattern]] = [
    (4, "PhD", re.compile(r"\b(ph\.?\s?d|doctorate|doctor of philosophy)\b", re.I)),
    (3, "Master's", re.compile(r"(\bm\.?\s?tech\b|\bm\.s\.|\bms\s+(?:in|cs|computer|data)\b|\bm\.?\s?sc\b|\bm\.e\.|\bm\.?\s?eng\b|\bmca\b|\bmba\b|\bmasters?\b|\bmaster'?s\b|\bmaster of\b)", re.I)),
    (2, "Bachelor's", re.compile(r"(\bb\.?\s?tech\b|\bb\.e\.|\bb\.?\s?eng\b|\bb\.?\s?sc\b|\bbca\b|\bb\.?\s?com\b|\bbachelors?\b|\bbachelor'?s\b|\bbachelor of\b|\bundergraduate\b|\bb\.s\.|\bbs\s+(?:in|cs|computer)\b)", re.I)),
    (1, "Diploma", re.compile(r"\b(diploma|associate degree|polytechnic)\b", re.I)),
]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<![\w/])\+?\(?\d[\d\s().-]{8,16}\d(?![\w/])")
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w-]+", re.I)
GITHUB_RE = re.compile(r"github\.com/[\w-]+", re.I)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
NUMBER_RE = re.compile(r"(\d+(\.\d+)?\s?(%|x|k|m|\+)|\$\s?\d|₹\s?\d|\b\d{2,}\b)", re.I)


@dataclass
class ExperienceSpan:
    start: float
    end: float
    text: str

    @property
    def years(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ResumeProfile:
    text: str
    name: str
    sections: dict[str, str]
    skills: dict[str, SkillHit]
    experience_years: float
    experience_spans: list[ExperienceSpan]
    internships: int
    education_level: int
    education_label: str
    cgpa: float | None
    bullets: list[str]
    quantified_bullets: list[str]
    contact: dict[str, bool]
    organizations: list[str] = field(default_factory=list)
    word_count: int = 0
    experiences: list[ExperienceEntry] = field(default_factory=list)
    cgpa_raw: str | None = None

    @property
    def internship_months(self) -> float:
        return merged_months([e for e in self.experiences if e.is_internship])

    @property
    def skill_names(self) -> list[str]:
        return list(self.skills.keys())


_SUBLABEL_RE = re.compile(r"responsibilit|details|duties|achievement|project|highlight|description|key\s", re.I)


def _is_header(line: str) -> str | None:
    clean = re.sub(r"[^a-z&\- ]", "", line.lower()).strip()
    clean = re.sub(r"\s{2,}", " ", clean)
    if not clean or len(clean) > 50:
        return None
    if clean in _HEADER_LOOKUP:
        return _HEADER_LOOKUP[clean]
    # Lenient match for compound experience headings: "WORK EXPERIENCE & INTERNSHIPS", "Experience (Internship)"
    raw = line.strip().rstrip(":").strip()
    words = clean.split()
    looks_like_heading = raw.isupper() or line.strip().endswith(":") or all(w[:1].isupper() for w in raw.split()
                                                                          if w[:1].isalpha())
    if (len(words) <= 6 and looks_like_heading and re.search(r"experience|internship|employment", clean)
            and not _SUBLABEL_RE.search(clean) and "project" not in clean):
        return "experience"
    return None


def experience_segments(text: str) -> list[tuple[str, str]]:
    """(raw heading, body) for every experience-like section, in document order."""
    segs: list[tuple[str, list[str]]] = []
    current = None
    for line in text.split("\n"):
        canon = _is_header(line.strip().rstrip(":"))
        if canon:
            current = canon
            if canon == "experience":
                segs.append((line.strip(), []))
            continue
        if current == "experience" and segs:
            segs[-1][1].append(line)
    return [(h, "\n".join(b).strip()) for h, b in segs if "\n".join(b).strip()]


def _heading_is_internship_only(heading: str) -> bool:
    return bool(re.search(r"intern|training", heading, re.I)) and not re.search(
        r"work|employment|professional|&|\band\b|career", heading, re.I)


def split_sections(text: str) -> dict[str, str]:
    """Split resume text into canonical sections using header detection. Text before any header -> 'header'."""
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for line in text.split("\n"):
        canon = _is_header(line.strip().rstrip(":"))
        if canon:
            current = canon
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def _frac(entry_range, end: bool = False) -> float:
    y, m = entry_range.end if end else entry_range.start
    if entry_range.year_precision:
        return float(y)
    return y + (m if end else m - 1) / 12


def find_date_ranges(text: str) -> list[ExperienceSpan]:
    """Backward-compatible helper: every date range in text as fractional-year spans (month-inclusive)."""
    spans = []
    for line in text.split("\n"):
        for r in find_ranges(line):
            if r.months:
                start = _frac(r)
                spans.append(ExperienceSpan(start, start + r.months / 12, line.strip()[:160]))
    return spans


def merge_spans(spans: list[ExperienceSpan]) -> float:
    """Total years covered by possibly-overlapping spans (overlaps counted once)."""
    if not spans:
        return 0.0
    ordered = sorted((s.start, s.end) for s in spans)
    total, cur_s, cur_e = 0.0, *ordered[0]
    for s, e in ordered[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return round(total, 1)


_NON_EXPERIENCE = {"education", "projects", "certifications", "skills", "achievements", "publications", "summary"}


def extract_experiences(text: str, sections: dict[str, str]) -> list[ExperienceEntry]:
    segs = experience_segments(text)
    entries: list[ExperienceEntry] = []
    for heading, body in segs:
        entries += extract_entries(body, section_is_internship=_heading_is_internship_only(heading))
    if not segs:  # no experience heading: only accept dated lines that carry a job title (avoids education dates)
        scope = "\n".join(v for k, v in sections.items() if k not in _NON_EXPERIENCE)
        entries = extract_entries(scope, require_dates=True)
    return entries


def estimate_experience(text: str, sections: dict[str, str]
                        ) -> tuple[float, list[ExperienceSpan], int, list[ExperienceEntry]]:
    entries = extract_experiences(text, sections)
    months = merged_months(entries)
    years = round(months / 12, 2)
    spans = []
    for e in entries:
        if e.date_range and e.date_range.months:
            start = _frac(e.date_range)
            label = " · ".join(x for x in (e.title, e.company, e.date_text) if x)
            spans.append(ExperienceSpan(start, start + e.date_range.months / 12, label))
    stated = [float(m.group(1)) for m in _YEARS_STATED_RE.finditer(text)]
    if stated and not months:  # candidate explicitly states "3+ years of experience" and lists no dated roles
        years = max(stated)
    internships = sum(1 for e in entries if e.is_internship)
    return years, spans, internships, entries


def detect_education(text: str, sections: dict[str, str]) -> tuple[int, str, float | None]:
    scope = sections.get("education", text)
    level, label = 0, "Not detected"
    for lvl, lab, pat in DEGREE_PATTERNS:
        if pat.search(scope):
            level, label = lvl, lab
            break
    cgpa, _ = extract_cgpa(scope)
    return level, label, cgpa


_GRADE_NUM = r"(\d{1,2}(?:\.\d{1,2})?)"
_CGPA_RES = [
    re.compile(rf"\b(?:c\.?g\.?p\.?a|s\.?g\.?p\.?a|gpa|cpi)\b\s*(?:[:\-–=]|of|is)?\s*{_GRADE_NUM}"
               rf"(?:\s*(?:/|out\s+of)\s*{_GRADE_NUM})?", re.I),
    re.compile(rf"(?<![\d.]){_GRADE_NUM}(?:\s*/\s*{_GRADE_NUM})?\s*(?:c\.?g\.?p\.?a|gpa|cpi)\b", re.I),
]


def extract_cgpa(text: str) -> tuple[float | None, str | None]:
    """Return (CGPA normalised to /10, raw text). Nothing is invented: if no grade is written, both are None.
    If the scale is not written, a value > 4 is read as /10 (Indian CGPA); a value <= 4 is ambiguous, so only the
    raw text is returned."""
    for pat in _CGPA_RES:
        m = pat.search(text)
        if not m:
            continue
        val = float(m.group(1))
        scale = float(m.group(2)) if m.group(2) else None
        raw = m.group(0).strip()
        if scale is not None:
            if 0 < val <= scale and scale in (4.0, 5.0, 10.0):
                return round(val / scale * 10, 2), raw
            return None, None
        if 4.0 < val <= 10.0:
            return round(val, 2), raw
        if 0 < val <= 4.0:
            return None, raw + " (scale not stated)"
        return None, None
    return None, None


def guess_name(text: str) -> str:
    for line in text.split("\n")[:5]:
        line = line.strip()
        if not line or EMAIL_RE.search(line) or any(ch.isdigit() for ch in line):
            continue
        words = line.split()
        if 1 < len(words) <= 4 and all(w[:1].isupper() for w in words if w.isalpha()):
            return line
    return "Candidate"


def extract_bullets(text: str) -> list[str]:
    bullets = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("- ") and len(s.split()) >= 4:
            bullets.append(s[2:].strip())
    if not bullets:  # resumes without bullet glyphs: use sentences in experience-like lines
        bullets = [s for s in nlpmod.sentences(text) if 6 <= len(s.split()) <= 60]
    return bullets


def build_profile(text: str) -> ResumeProfile:
    sections = split_sections(text)
    years, spans, internships, entries = estimate_experience(text, sections)
    level, label, cgpa = detect_education(text, sections)
    _, cgpa_raw = extract_cgpa(sections.get("education", text))
    bullets = extract_bullets(text)
    quantified = [b for b in bullets if NUMBER_RE.search(b)]
    contact = {
        "email": bool(EMAIL_RE.search(text)),
        "phone": any(sum(c.isdigit() for c in m.group()) >= 10 for m in PHONE_RE.finditer(text)),
        "linkedin": bool(LINKEDIN_RE.search(text)),
        "github": bool(GITHUB_RE.search(text)),
    }
    orgs = nlpmod.entities(sections.get("experience", text)).get("ORG", [])[:10]
    return ResumeProfile(
        text=text, name=guess_name(text), sections=sections, skills=extract_skills(text),
        experience_years=years, experience_spans=spans, internships=internships,
        education_level=level, education_label=label, cgpa=cgpa, bullets=bullets,
        quantified_bullets=quantified, contact=contact, organizations=orgs, word_count=len(text.split()),
        experiences=entries, cgpa_raw=cgpa_raw,
    )


def redact_pii(text: str) -> str:
    """Mask emails, phone numbers and URLs before sending text to any external API."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[URL]", text)
    text = LINKEDIN_RE.sub("[LINKEDIN]", text)
    text = GITHUB_RE.sub("[GITHUB]", text)
    text = PHONE_RE.sub(lambda m: "[PHONE]" if sum(c.isdigit() for c in m.group()) >= 10 else m.group(), text)
    return text