"""Rule-based + NLP resume improvement suggestions and a resume quality score."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .extractor import ResumeProfile
from .nlp import passive_sentences

WEAK_PHRASES = {
    "responsible for": "Led / Owned / Delivered",
    "worked on": "Built / Implemented / Developed",
    "helped": "Contributed to / Enabled / Supported X by …",
    "assisted": "Partnered with … to deliver …",
    "involved in": "Drove / Executed",
    "duties included": "Start with the result you produced",
    "tasked with": "Delivered / Shipped",
    "participated in": "Contributed … resulting in …",
    "familiar with": "Move to Skills, or show it in a project bullet",
}
ACTION_VERBS = {
    "built", "developed", "designed", "implemented", "led", "created", "optimized", "optimised", "improved",
    "reduced", "increased", "automated", "deployed", "trained", "analyzed", "analysed", "architected", "launched",
    "engineered", "delivered", "integrated", "migrated", "scaled", "achieved", "streamlined", "fine-tuned",
    "evaluated", "researched", "published", "mentored", "won", "established", "accelerated", "refactored",
}
BUZZWORDS = {"hardworking", "hard-working", "go-getter", "synergy", "detail-oriented", "team player",
             "self-motivated", "passionate", "dynamic", "results-driven", "think outside the box"}


@dataclass
class Suggestion:
    severity: str  # high | medium | low
    area: str
    title: str
    detail: str
    example: str = ""


def _starts_with_action(b: str) -> bool:
    first = re.sub(r"[^a-z\-]", "", b.split()[0].lower()) if b.split() else ""
    return first in ACTION_VERBS or first.endswith("ed")


def analyze_resume(profile: ResumeProfile, missing_keywords: list[str] | None = None) -> tuple[float, list[Suggestion]]:
    s: list[Suggestion] = []
    text_l = profile.text.lower()
    secs = profile.sections

    # --- Structure
    for key, label, sev in (("experience", "Experience / Internships", "high"), ("projects", "Projects", "high"),
                            ("skills", "Skills", "high"), ("education", "Education", "medium"),
                            ("summary", "Summary", "low")):
        if key not in secs:
            s.append(Suggestion(sev, "Structure", f"Add a clear '{label}' section",
                                f"ATS parsers and recruiters look for a standard '{label}' heading. None was detected."))
    # --- Contact
    if not profile.contact["email"]:
        s.append(Suggestion("high", "Contact", "Add a professional email", "No email address was detected."))
    if not profile.contact["linkedin"]:
        s.append(Suggestion("medium", "Contact", "Add your LinkedIn URL", "Recruiters routinely check LinkedIn."))
    if not profile.contact["github"] and any(k in profile.skills for k in ("Python", "Machine Learning", "JavaScript", "Java")):
        s.append(Suggestion("medium", "Contact", "Add a GitHub/portfolio link",
                            "For technical roles, linked code is strong evidence of skill."))
    # --- Length
    wc = profile.word_count
    if wc < 250:
        s.append(Suggestion("high", "Length", "Resume is too thin",
                            f"Only {wc} words. Aim for 400–800 words for a student/early-career one-page resume."))
    elif wc > 1100:
        s.append(Suggestion("medium", "Length", "Resume may be too long",
                            f"{wc} words. Keep it to one page (≈ 400–800 words) unless you have 5+ years of experience."))
    # --- Bullets: quantification, verbs, weak phrases, length
    bullets = profile.bullets
    if bullets:
        q_ratio = len(profile.quantified_bullets) / len(bullets)
        if q_ratio < 0.4:
            s.append(Suggestion("high", "Impact", "Quantify more achievements",
                                f"Only {q_ratio:.0%} of bullets contain numbers. Target ≥ 50%: users, %, latency, accuracy, money, time saved.",
                                "Built a churn model → Built a churn model (XGBoost) that lifted recall from 0.61 to 0.78 on 50k customers."))
        non_action = [b for b in bullets if not _starts_with_action(b)]
        if len(non_action) / len(bullets) > 0.4:
            s.append(Suggestion("medium", "Language", "Start bullets with strong action verbs",
                                f"{len(non_action)} of {len(bullets)} bullets don't start with an action verb.",
                                f"“{non_action[0][:80]}” → start with Built / Designed / Automated …"))
        long_b = [b for b in bullets if len(b.split()) > 32]
        if long_b:
            s.append(Suggestion("low", "Readability", "Shorten long bullets",
                                f"{len(long_b)} bullet(s) exceed 32 words. Aim for 1–2 lines each.", long_b[0][:120] + "…"))
    for phrase, better in WEAK_PHRASES.items():
        if phrase in text_l:
            m = re.search(rf"[^\n]*{re.escape(phrase)}[^\n]*", profile.text, re.I)
            s.append(Suggestion("medium", "Language", f"Replace weak phrase “{phrase}”",
                                f"Passive duty statements hide impact. Try: {better}.", m.group(0).strip()[:120] if m else ""))
    # --- Pronouns, buzzwords, passive voice
    pronouns = re.findall(r"\b(I|me|my)\b", profile.text)
    if len(pronouns) >= 3:
        s.append(Suggestion("low", "Language", "Remove first-person pronouns",
                            f"Found {len(pronouns)} uses of I/me/my. Resumes use implied first person."))
    buzz = sorted(b for b in BUZZWORDS if b in text_l)
    if buzz:
        s.append(Suggestion("low", "Language", "Swap buzzwords for evidence",
                            f"Found: {', '.join(buzz)}. Show these traits through outcomes instead."))
    passive = passive_sentences("\n".join(bullets[:40]))
    if len(passive) >= 2:
        s.append(Suggestion("low", "Language", "Reduce passive voice",
                            f"{len(passive)} bullet(s) use passive voice (detected by spaCy dependency parsing).",
                            passive[0][:120]))
    # --- Skills section depth
    if len(profile.skills) < 6:
        s.append(Suggestion("high", "Skills", "List more concrete, relevant skills",
                            f"Only {len(profile.skills)} recognised skills. Name tools, frameworks and libraries explicitly."))
    unevidenced = [n for n, h in profile.skills.items() if h.count == 1 and "skills" in secs
                   and any(n.lower() in ln.lower() for ln in secs["skills"].split("\n"))]
    if len(unevidenced) >= 5:
        s.append(Suggestion("medium", "Skills", "Back up listed skills with evidence",
                            f"{len(unevidenced)} skills appear only in the Skills list (e.g. {', '.join(unevidenced[:5])}). "
                            "Mention them in project/experience bullets to prove usage."))
    # --- Job-specific keywords
    if missing_keywords:
        s.append(Suggestion("high", "Targeting", "Address job keywords you genuinely have",
                            "These job requirements were not found in your resume: " + ", ".join(missing_keywords[:10])
                            + ". Add them only if you have real experience; otherwise see the learning plan."))
    if profile.education_level == 0:
        s.append(Suggestion("medium", "Education", "State your degree clearly",
                            "No degree was recognised. Write it as e.g. 'B.Tech in Computer Science, 2025 — CGPA 8.4/10'."))

    # --- Quality score: start at 100, subtract by severity
    penalty = {"high": 12, "medium": 6, "low": 3}
    score = max(0.0, 100.0 - sum(penalty[x.severity] for x in s))
    order = {"high": 0, "medium": 1, "low": 2}
    s.sort(key=lambda x: order[x.severity])
    return round(score, 1), s
