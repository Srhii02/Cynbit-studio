"""Downloadable per-candidate PDF report (ReportLab)."""
from __future__ import annotations

import io
from datetime import date
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BRAND = colors.HexColor("#4F46E5")


def _styles():
    from .fonts import register_fonts

    reg, bold, _ = register_fonts()
    ss = getSampleStyleSheet()
    for st in ss.byName.values():
        if hasattr(st, "fontName"):
            st.fontName = bold if "Bold" in st.fontName else reg
    ss.add(ParagraphStyle("H1x", parent=ss["Title"], textColor=BRAND, fontSize=20, spaceAfter=4))
    ss.add(ParagraphStyle("H2x", parent=ss["Heading2"], textColor=colors.HexColor("#111827"), spaceBefore=10))
    ss.add(ParagraphStyle("Small", parent=ss["BodyText"], fontSize=8.5, textColor=colors.HexColor("#6B7280")))
    return ss


def _table(rows, widths, header=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    from .fonts import register_fonts

    reg, bold, _ = register_fonts()
    style = [("FONTNAME", (0, 0), (-1, -1), reg), ("FONTNAME", (0, 0), (-1, 0), bold), ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E5E7EB")),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")])]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), BRAND), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(style))
    return t


def build_candidate_report(profile, results, recs, readiness, quality, suggestions) -> bytes:
    ss = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title="Cynbit Studio – Career Match Report")
    P = lambda t, s="BodyText": Paragraph(escape(str(t)), ss[s])  # noqa: E731
    el = [P("Cynbit Studio · Career Match Report", "H1x"),
          P(f"Generated {date.today():%d %b %Y} · Engine: {results[0].engine if results else '—'}", "Small"),
          Spacer(1, 6)]
    el += [P("Career readiness", "H2x"),
           P(f"Readiness index {readiness.index:.0f}/100 — {readiness.level}. Resume quality {quality:.0f}/100. "
             f"Experience ≈ {profile.experience_years:g} yrs, {profile.internships} internship mention(s), "
             f"education: {profile.education_label}.")]
    el.append(_table([["Component", "Score"]] + [[k, f"{v:.0f}"] for k, v in readiness.components.items()],
                     [120 * mm, 40 * mm]))
    if results:
        el.append(P("Job ranking", "H2x"))
        rows = [["#", "Role", "Overall", "Semantic", "Skills", "Verdict"]]
        for i, r in enumerate(results, 1):
            rows.append([i, Paragraph(escape(r.job_title[:60]), ss["BodyText"]), f"{r.overall:.0f}",
                         f"{r.semantic:.0f}", f"{r.skills:.0f}", r.verdict])
        el.append(_table(rows, [8 * mm, 72 * mm, 20 * mm, 20 * mm, 18 * mm, 30 * mm]))
        best = results[0]
        el.append(P(f"Why “{best.job_title}” ranks first", "H2x"))
        for e in best.explanations:
            el.append(P("• " + e))
    if recs:
        el.append(P("Personalised learning plan", "H2x"))
        rows = [["Skill", "Priority", "Hours", "Weeks", "Start here"]]
        for r in recs:
            res = r.resources[0]["title"] if r.resources else "—"
            rows.append([r.skill, r.priority_label, r.est_hours, f"{r.week_start}–{r.week_end}",
                         Paragraph(escape(res), ss["BodyText"])])
        el.append(_table(rows, [38 * mm, 20 * mm, 16 * mm, 18 * mm, 76 * mm]))
    if suggestions:
        el.append(P("Resume improvement suggestions", "H2x"))
        for s in suggestions[:12]:
            el.append(P(f"[{s.severity.upper()}] {s.title} — {s.detail}"))
    el += [Spacer(1, 8), P("Scores are computed locally from NLP similarity and skill coverage; they are guidance, "
                           "not a hiring decision.", "Small")]
    doc.build(el)
    return buf.getvalue()
