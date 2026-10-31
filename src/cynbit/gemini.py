"""Optional Gemini integration for natural-language coaching. The core app never depends on it.

Scores are always computed locally; Gemini only rewrites/explains. PII is redacted before any request.
"""
from __future__ import annotations

import logging

from .config import settings
from .extractor import redact_pii

log = logging.getLogger(__name__)


def is_available() -> bool:
    if not settings.gemini_api_key:
        return False
    try:
        from google import genai  # noqa: F401  (package: google-genai)
        return True
    except ImportError:
        return False


def coach(resume_text: str, job_text: str, missing: list[str], score: float) -> str:
    """Ask Gemini for rewritten bullets and a short coaching note. Returns '' if unavailable or on error."""
    if not is_available():
        return ""
    from google import genai

    prompt = (
        "You are a career coach for students. The candidate's resume and a job description follow. A local NLP model "
        f"scored the match at {score:.0f}/100 and found these missing skills: {', '.join(missing) or 'none'}.\n"
        "1) Rewrite the 3 weakest resume bullets to be specific and quantified (do NOT invent facts; use [X] placeholders "
        "for numbers the candidate must fill in).\n2) Give 3 concise, honest tips to improve fit.\n"
        "Keep the answer under 250 words, in Markdown.\n\n"
        f"RESUME:\n{redact_pii(resume_text)[:6000]}\n\nJOB DESCRIPTION:\n{job_text[:4000]}"
    )
    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        resp = client.models.generate_content(model=settings.gemini_model, contents=prompt)
        return resp.text or ""
    except Exception as exc:  # network / quota / auth errors must never break the app
        log.warning("Gemini request failed: %s", exc)
        return f"_Gemini request failed: {type(exc).__name__}. Local analysis above is unaffected._"
