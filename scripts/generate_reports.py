"""Generate docs/AI_Opportunity_Report.pdf and docs/Technical_Approach_Report.pdf.

The evaluation section is computed live by running the real pipeline on the synthetic samples, so the numbers in
the Technical Approach Report always reflect the embedding engine available on the machine that generated it.

    python scripts/generate_reports.py
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

BRAND = colors.HexColor("#4F46E5")
TEAL = colors.HexColor("#0D9488")
INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#6B7280")
LIGHT = colors.HexColor("#EEF2FF")

from cynbit.fonts import register_fonts  # noqa: E402

REG, BOLD, MONO = register_fonts()
ss = getSampleStyleSheet()
for _st in ss.byName.values():
    if hasattr(_st, "fontName"):
        _st.fontName = BOLD if "Bold" in _st.fontName else REG
H0 = ParagraphStyle("H0", parent=ss["Title"], fontSize=24, leading=29, textColor=BRAND, alignment=0, spaceAfter=6)
SUB = ParagraphStyle("SUB", parent=ss["BodyText"], fontSize=11, textColor=MUTED, spaceAfter=14)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=INK, spaceBefore=12, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, textColor=BRAND, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.8, leading=14, spaceAfter=5)
BUL = ParagraphStyle("BL", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)
CODE = ParagraphStyle("C", parent=BODY, fontName=MONO, fontSize=8.2, leading=11.5, backColor=LIGHT,
                      borderPadding=6, spaceBefore=4, spaceAfter=8)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, textColor=MUTED)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.3, leading=10.5, spaceAfter=0)


def p(t, st=BODY):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(rows, widths, header=True, font=8.3):
    data = [[c if not isinstance(c, str) else Paragraph(c, CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
          ("FONTSIZE", (0, 0), (-1, -1), font), ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), BRAND), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(st))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(REG, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, f"Cynbit Studio – {doc.title}")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build(path: Path, title: str, story):
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=title, author="Cynbit Studio prototype")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# --------------------------------------------------------------------------------------------- diagrams
def _box(d, x, y, w, h, title, sub="", fill=LIGHT, stroke=BRAND):
    d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=fill, strokeColor=stroke, strokeWidth=1))
    d.add(String(x + w / 2, y + h - 14, title, fontName=BOLD, fontSize=8.5, fillColor=INK,
                 textAnchor="middle"))
    for i, line in enumerate(sub.split("\n")):
        d.add(String(x + w / 2, y + h - 26 - 10 * i, line, fontName=REG, fontSize=7, fillColor=MUTED,
                     textAnchor="middle"))


def _arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=MUTED, strokeWidth=1))
    import math

    a = math.atan2(y2 - y1, x2 - x1)
    s = 5
    d.add(Polygon([x2, y2, x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4),
                   x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4)], fillColor=MUTED, strokeColor=MUTED))


def architecture_diagram() -> Drawing:
    d = Drawing(500, 290)
    W, H = 110, 52
    # row 1 inputs
    _box(d, 10, 225, W, H, "Resume upload", "PDF · DOCX · TXT")
    _box(d, 10, 150, W, H, "Job descriptions", "paste · upload · samples")
    # row 2 parsing / understanding
    _box(d, 140, 225, W, H, "Parser", "PyMuPDF · python-docx\nclean_text()")
    _box(d, 140, 150, W, H, "JD Analyzer", "required / preferred\nyears · degree · seniority")
    _box(d, 270, 225, W, H, "Resume Extractor", "spaCy NER · sections\nskills · dates · CGPA")
    # semantic core
    _box(d, 270, 130, W, 70, "Semantic Engine", "all-MiniLM-L6-v2 (384-d)\nchunk + mean-pool\nTF-IDF+LSA fallback",
         fill=colors.HexColor("#CCFBF1"), stroke=TEAL)
    _box(d, 395, 175, 100, 70, "Matcher", "doc cosine + requirement\nalignment + skills\n+ exp + edu → score",
         fill=colors.HexColor("#CCFBF1"), stroke=TEAL)
    # outputs
    _box(d, 10, 20, 90, H, "Ranking", "multi-job table\nskill heatmap")
    _box(d, 110, 20, 90, H, "Recommender", "learning plan\nresources · weeks")
    _box(d, 210, 20, 90, H, "Suggestions", "quality score\nspaCy passive voice")
    _box(d, 310, 20, 90, H, "Readiness", "index · radar\ncoverage")
    _box(d, 410, 20, 85, H, "SQLite · PDF", "history (no raw text)\nreport export")
    _arrow(d, 120, 251, 140, 251)
    _arrow(d, 120, 176, 140, 176)
    _arrow(d, 250, 251, 270, 251)
    _arrow(d, 380, 240, 395, 225)
    _arrow(d, 250, 170, 270, 168)
    _arrow(d, 380, 165, 395, 195)
    _arrow(d, 325, 225, 325, 200)
    for x in (55, 155, 255, 355, 452):
        _arrow(d, 445, 175, x, 72)
    return d


# --------------------------------------------------------------------------------------------- evaluation
def run_evaluation():
    from cynbit import analyze
    from cynbit.embeddings import get_engine
    from cynbit.extractor import build_profile
    from cynbit.jd_analyzer import analyze_job
    from cynbit.matcher import match
    from cynbit.parser import parse_file

    engine = get_engine()
    resumes = {p.stem: parse_file(p).text for p in sorted((ROOT / "samples/resumes").glob("*"))
               if p.suffix in {".pdf", ".docx"}}
    jobs = {p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "samples/jobs").glob("*.txt"))}
    expected = {"ananya_rao_ml_student": {"ai_ml_engineer_intern", "nlp_engineer_junior"},
                "karthik_menon_fullstack": {"full_stack_developer"},
                "meera_iyer_data_analyst": {"data_analyst_intern"}}
    matrix, hits, timings = {}, 0, []
    job_profiles = {j: analyze_job(t) for j, t in jobs.items()}
    for rn, rt in resumes.items():
        prof = build_profile(rt)
        row = {}
        for jn, jp in job_profiles.items():
            t0 = time.perf_counter()
            row[jn] = match(prof, jp, engine).overall
            timings.append(time.perf_counter() - t0)
        matrix[rn] = row
        if rn in expected and max(row, key=row.get) in expected[rn]:
            hits += 1
    full = {rn: analyze(rt, list(jobs.values()), engine=engine) for rn, rt in resumes.items()}
    return engine.name, matrix, hits, len(expected), sum(timings) / len(timings), full


# --------------------------------------------------------------------------------------------- reports
def opportunity_report(path: Path):
    s = [p("AI Opportunity Report", H0),
         p(f"Cynbit Studio – AI Resume &amp; Job Matching Intelligence · Prepared for the Cynbit Technologies AI/ML "
           f"Engineer Intern assessment · {date.today():%d %B %Y}", SUB)]
    s += [p("1. Executive summary", H1),
          p("Students and early-career candidates apply widely but rarely know <b>which roles they genuinely fit</b>, "
            "<b>which skills are missing</b>, and <b>what to learn next</b>. Cynbit Studio is a working prototype that "
            "turns a resume and a set of job descriptions into an explainable match score, a ranked job list, a "
            "prioritised skills gap, a week-by-week learning plan, resume fixes and a career-readiness index. It runs "
            "locally with open models (Sentence Transformers <i>all-MiniLM-L6-v2</i>, spaCy), so it is cheap to operate "
            "and privacy-friendly, with optional Gemini coaching on top."),
          p("For Cynbit Technologies, the prototype is a reusable <b>AI module</b> that can be offered to colleges, "
            "placement cells, training institutes and recruiting clients, embedded into client websites, and used to "
            "generate useful, search-friendly career content.")]
    s += [p("2. The problem", H1),
          p("The India Skills Report 2025 estimates graduate employability at about <b>54.8%</b>, meaning that "
            "roughly 45% of assessed graduates are not yet considered job-ready. The gap is less about effort than "
            "about direction: candidates lack clear, personalised feedback."),
          *bullets(["<b>Opaque screening:</b> keyword-based ATS filters reject resumes without telling candidates why.",
                    "<b>Unclear targeting:</b> students apply to roles that don't match their profile and miss ones that do.",
                    "<b>Generic learning:</b> course recommendations aren't tied to the skills specific jobs require.",
                    "<b>Weak resumes:</b> duty-style bullets with no measurable impact hide real ability.",
                    "<b>Limited placement-cell capacity:</b> one officer may serve hundreds of students."]),
          p("Source: India Skills Report 2025 (Wheebox / CII / AICTE), as reported by Deccan Herald and Swarajya, "
            "Nov 2024.", SMALL)]
    s += [p("3. Target users and value", H1),
          table([["Segment", "Pain point", "Value delivered"],
                 ["Students &amp; freshers", "Don't know fit or gaps", "Ranked jobs, missing skills, learning plan, resume fixes"],
                 ["Colleges / placement cells", "Too many students, little time", "Batch readiness dashboards, cohort skill-gap analytics"],
                 ["Training institutes", "Generic curricula", "Data on which skills are demanded vs. missing"],
                 ["Recruiters / SMEs", "Noisy applicant pools", "Explainable shortlisting aid (human-in-the-loop)"],
                 ["Cynbit's web clients", "Static career pages", "Embeddable 'check your fit' widget on job pages"]],
                [38 * mm, 55 * mm, 81 * mm])]
    s += [p("4. Solution delivered in this prototype", H1),
          table([["Capability", "How it works (real NLP, no hard-coded scores)"],
                 ["Resume parsing", "PyMuPDF (reading-order text) and python-docx (incl. tables); unicode/bullet clean-up"],
                 ["Skill &amp; experience extraction", "90-skill taxonomy (~430 surface forms) with special-character-safe matching; "
                                                     "date-range parsing with overlap merging; degree &amp; CGPA detection; spaCy NER"],
                 ["Job description analysis", "Required vs. preferred skills by section and cue phrases; min. years, degree, seniority; "
                                              "spaCy noun-chunk key phrases"],
                 ["Semantic matching", "MiniLM embeddings: whole-document cosine + per-requirement best-evidence alignment"],
                 ["Explainable score", "40% semantic · 35% skills · 15% experience · 10% education, each shown with reasons"],
                 ["Gap &amp; learning plan", "Missing / related skills; priority × effort scheduling; curated free resources"],
                 ["Multi-job ranking", "Ranking table, grouped chart, skill-demand heatmap, CSV export"],
                 ["Readiness dashboard", "Composite index, radar, category coverage, downloadable PDF report"],
                 ["Resume improvement", "Impact, action verbs, weak phrases, passive voice (spaCy), structure, keywords"]],
                [45 * mm, 129 * mm])]
    s += [p("5. Why this approach", H1),
          *bullets(["<b>Explainable:</b> every score decomposes into components and evidence lines a student can act on.",
                    "<b>Low cost:</b> MiniLM (~23M parameters) runs on CPU in milliseconds per sentence; no per-request LLM fees.",
                    "<b>Private by design:</b> resumes stay local; history stores only a fingerprint and scores; PII is "
                    "redacted before any optional LLM call.",
                    "<b>Robust:</b> automatic offline fallback (TF-IDF + LSA) keeps the product usable without model downloads."])]
    s += [p("6. Integration with Cynbit Studio", H1),
          p("Cynbit Technologies' public profile lists SEO, web design, UX/UI and custom software development. The "
            "plan below assumes Cynbit Studio is (or becomes) the company's product/delivery platform for client web "
            "solutions; it requires <b>no change to Cynbit's current website</b> and can be adopted incrementally."),
          table([["Phase", "Deliverable", "Technical approach", "Effort"],
                 ["1 · Service", "Matching API", "FastAPI wrapper around cynbit.analyze(); Docker; /parse /match /rank /recommend", "2–3 wks"],
                 ["2 · Widget", "Embeddable 'Check your fit'", "JS widget calling the API; JWT/SSO; brandable per client", "2–4 wks"],
                 ["3 · Scale", "Job index", "PostgreSQL + pgvector, job embeddings computed once, nightly feeds", "3–4 wks"],
                 ["4 · Analytics", "Placement-cell dashboard", "Cohort readiness, skill-demand trends, exports", "3–4 wks"],
                 ["5 · Learning loop", "Calibrated scoring", "Fit component weights to recruiter shortlist outcomes", "ongoing"],
                 ["6 · GenAI", "Coaching &amp; content", "Gemini/LLM bullet rewrites, cover letters, interview prep, "
                                                       "SEO-friendly skill-gap articles", "2–3 wks"]],
                [24 * mm, 36 * mm, 92 * mm, 22 * mm]),
          Spacer(1, 4),
          p("<b>Business models:</b> (a) B2B SaaS licence for colleges and training institutes (per-student/year); "
            "(b) white-label module bundled into Cynbit web projects for HR, staffing and education clients; "
            "(c) lead generation and SEO value from public, privacy-safe skill-gap insights; (d) recruiter "
            "add-on for explainable pre-screening.")]
    s += [p("7. Success metrics", H1),
          table([["Metric", "Target for a pilot (one college, one semester)"],
                 ["Activation", "≥ 60% of final-year students upload a resume"],
                 ["Engagement", "≥ 2 job comparisons per active student; ≥ 30% revisit within 30 days"],
                 ["Outcome", "Readiness index +10 points median after 8 weeks; ↑ interview-call rate vs. previous batch"],
                 ["Quality", "Top-1 job ranking agrees with placement officer judgement in ≥ 80% of sampled cases"],
                 ["Trust", "Zero raw resumes stored server-side; documented consent under India's DPDP Act, 2023"]],
                [35 * mm, 139 * mm])]
    s += [p("8. Risks and responsible AI", H1),
          table([["Risk", "Mitigation"],
                 ["Bias against non-standard resumes or regional phrasing", "Skill-alias taxonomy, semantic matching, human review; audit scores by college/region"],
                 ["Over-reliance on a single number", "Component scores + evidence; UI states scores are guidance, not decisions"],
                 ["Keyword gaming", "Requirement-level semantic evidence and 'back up listed skills' checks"],
                 ["Privacy / data protection", "Local processing, fingerprint-only history, PII redaction, consent, retention limits"],
                 ["LLM hallucination (optional coaching)", "LLM never sets scores; prompts forbid inventing facts and use [X] placeholders"]],
                [62 * mm, 112 * mm])]
    s += [p("9. Next steps", H1),
          *bullets(["Validate on 100–200 real, consented resume–job pairs labelled by placement officers; calibrate weights.",
                    "Ship the FastAPI service and widget (Phases 1–2) for a single pilot institution.",
                    "Extend the taxonomy with Cynbit's client domains and regional job titles; add OCR for scanned resumes."])]
    build(path, "AI Opportunity Report", s)


def technical_report(path: Path, ev):
    engine_name, matrix, hits, n_expected, avg_t, full = ev
    s = [p("Technical Approach Report", H0),
         p(f"Cynbit Studio – AI Resume &amp; Job Matching Intelligence · {date.today():%d %B %Y} · evaluation engine: "
           f"<b>{escape(engine_name)}</b>", SUB)]
    s += [p("1. Objectives", H1),
          p("Build a modular, testable system that (1) parses resumes, (2) extracts skills and experience, (3) analyses "
            "job descriptions, (4) performs real semantic matching with Sentence Transformers, (5) explains the score, "
            "(6) identifies missing skills, (7) recommends learning, (8) compares and ranks jobs, (9) visualises career "
            "readiness and (10) suggests resume improvements.")]
    s += [p("2. Architecture", H1), architecture_diagram(),
          p("All AI logic lives in the framework-independent package <font name='DejaVuMono'>src/cynbit</font>; "
            "<font name='DejaVuMono'>app.py</font> is a thin Streamlit layer. The same <font name='DejaVuMono'>analyze()</font> "
            "function powers the UI, the CLI and the tests, which makes a later FastAPI service trivial.", SMALL)]
    s += [p("3. Pipeline", H1),
          p("3.1 Parsing", H2),
          p("PDFs are read with PyMuPDF using <font name='DejaVuMono'>get_text('text', sort=True)</font> for natural reading "
            "order in multi-column layouts; DOCX paragraphs and table cells are read with python-docx. Text is NFKC-"
            "normalised, ligatures (ﬁ, ﬂ) expanded, bullet glyphs unified, whitespace collapsed. Empty, encrypted, "
            "corrupted, unsupported or image-only files raise a user-friendly ParsingError."),
          p("3.2 Resume information extraction", H2),
          *bullets(["<b>Sections:</b> header detection against ~60 canonical aliases (Experience, Internships, Projects…).",
                    "<b>Skills:</b> taxonomy of 90 canonical skills / ~430 surface forms in 13 categories. Custom regex boundaries "
                    "<font name='DejaVuMono'>(?&lt;![\\w+#])alias(?![\\w+#]|\\.\\w)</font> match C++, C#, Node.js, CI/CD while "
                    "rejecting 'reactive'→React or 'javascript'→Java. Ambiguous names (R, Go) count only in list context.",
                    "<b>Experience:</b> date ranges (Jan 2023 – Present, 06/2021 – 08/2022, 2019 – 2021) parsed to fractional "
                    "years; overlapping intervals merged so concurrent roles are not double-counted.",
                    "<b>Education:</b> degree level (Diploma→PhD) with false-positive-safe patterns; CGPA normalised to /10.",
                    "<b>spaCy (en_core_web_sm):</b> sentence segmentation, ORG entities, noun-chunk key phrases and "
                    "dependency-based passive-voice detection (nsubjpass/auxpass)."]),
          p("3.3 Job description analysis", H2),
          p("Lines are bucketed into requirements, preferred and responsibilities by headers (\"Nice to have\", "
            "\"Requirements\"…) and, per sentence, by cue phrases (\"is a plus\", \"familiarity with\"). Skills found only "
            "in preferred text become <i>preferred</i>; all others are <i>required</i>. Minimum years, minimum degree and "
            "seniority are extracted with targeted patterns.")]
    s += [p("3.4 Semantic matching with all-MiniLM-L6-v2", H2),
          p("<i>all-MiniLM-L6-v2</i> is a 6-layer, 384-dimensional sentence-embedding model (~22.7M parameters) trained "
            "contrastively on over a billion sentence pairs. Because it truncates input at 256 word-pieces, documents are "
            "split into ~90-word chunks, embedded, mean-pooled and re-normalised. Two signals are combined:"),
          p("doc_sim  = cos( mean_pool(E(resume_chunks)), mean_pool(E(job_chunks)) )<br/>"
            "req_cov  = mean over job requirements r_i of  max_j cos(E(r_i), E(resume_line_j))<br/>"
            "semantic = 0.5·cal(doc_sim) + 0.5·cal(req_cov)", CODE),
          p("The requirement-level alignment is what makes the score explainable: for each requirement the UI shows "
            "the single resume line that best supports it, and flags weakly-covered requirements. "
            "<font name='DejaVuMono'>cal()</font> linearly maps the model's typical cosine range (≈0.20 unrelated → 0.75 "
            "strongly related) to 0–1; the raw cosine is always shown alongside."),
          p("<b>Offline fallback.</b> If the model cannot be loaded, a TF-IDF (1–2-grams, sublinear tf) + LSA "
            "(TruncatedSVD, 60 components, background corpus built from the skill taxonomy) engine is used. Canonical "
            "skill tokens are appended to each text so that aliases such as 'k8s' and 'Kubernetes' coincide. It has its "
            "own calibration range (0.05–0.55). The UI warns when the fallback is active.")]
    s += [p("3.5 Scoring", H2),
          p("overall    = 0.40·semantic + 0.35·skills + 0.15·experience + 0.10·education<br/>"
            "skills     = Σ w·credit / Σ w ;  w = 2 (required), 1 (preferred)<br/>"
            "credit     = 1 matched · 0.5 related · 0 missing<br/>"
            "experience = min(1, years / required_years)   (internships add 0.25 yr each if no dated roles)<br/>"
            "education  = 1 if degree ≥ required; 0.6 one level below; 0.3 otherwise; 0.5 if not detected", CODE),
          p("<b>Related skills</b> receive partial credit through taxonomy relations (TensorFlow ↔ PyTorch, FastAPI → "
            "REST APIs) and, with MiniLM, through embedding similarity ≥ 0.62 between skill names in the same "
            "category. Verdicts: ≥75 Strong, ≥60 Good, ≥45 Moderate, else Low."),
          p("3.6 Recommendations, suggestions and readiness", H2),
          *bullets(["<b>Learning plan:</b> gaps are aggregated over the 3 best-matching jobs; priority = Σ importance × "
                    "gap-type × job-fit weight; effort from taxonomy hours (halved for related skills); scheduled into "
                    "weeks at the student's chosen hours/week with curated free resources and a portfolio project idea.",
                    "<b>Resume quality:</b> 15+ checks (structure, contact links, length, quantified-bullet ratio, action "
                    "verbs, weak phrases, pronouns, buzzwords, passive voice, unevidenced skills, job keywords, degree); "
                    "score = 100 − 12/6/3 per high/medium/low issue.",
                    "<b>Readiness index:</b> 35% market fit (mean of top-3 matches), 20% skill depth, 20% resume quality, "
                    "15% practical experience, 10% profile completeness; plus per-category skill coverage."])]
    s += [p("4. Data, storage and privacy", H1),
          p("SQLite tables <font name='DejaVuMono'>resumes</font> (SHA-256 fingerprint, filename, derived skills, quality) "
            "and <font name='DejaVuMono'>analyses</font> (scores, verdict, engine, missing skills). Raw resume text is never "
            "persisted — a unit test asserts this. Gemini calls (optional) receive text with emails, phones and URLs "
            "redacted. Secrets live only in a git-ignored <font name='DejaVuMono'>.env</font>.")]
    # Evaluation
    jobs = list(next(iter(matrix.values())).keys())
    short = {j: j.replace("_", " ").title().replace("Ai Ml", "AI/ML").replace("Nlp", "NLP").replace("Mlops", "MLOps")
             for j in jobs}
    rows = [["Resume \\ Job"] + [short[j] for j in jobs]]
    for rn, row in matrix.items():
        best = max(row, key=row.get)
        rows.append([rn.replace("_", " ")] + [f"<b>{row[j]:.0f}</b>" if j == best else f"{row[j]:.0f}" for j in jobs])
    w = (174 * mm - 38 * mm) / len(jobs)
    s += [p("5. Evaluation on the synthetic dataset", H1),
          p(f"Four synthetic resumes (ML student, full-stack developer, data analyst, and a deliberately weak resume) "
            f"were scored against six synthetic job descriptions using <b>{escape(engine_name)}</b>. Bold marks each "
            f"resume's top-ranked job."),
          table(rows, [38 * mm] + [w] * len(jobs)),
          Spacer(1, 6),
          p(f"<b>Top-1 domain accuracy:</b> {hits}/{n_expected} resumes with a known target domain ranked a job from "
            f"that domain first. <b>Mean match latency:</b> {avg_t * 1000:.0f} ms per resume–job pair on CPU "
            f"(including requirement-level alignment)."),
          table([["Resume", "Readiness", "Level", "Resume quality", "#Suggestions"]] +
                [[rn.replace("_", " "), f"{a.readiness.index:.0f}", a.readiness.level, f"{a.quality:.0f}",
                  str(len(a.suggestions))] for rn, a in full.items()],
                [60 * mm, 25 * mm, 30 * mm, 30 * mm, 29 * mm]),
          Spacer(1, 4),
          p("The weak resume receives the lowest readiness and quality scores and the most suggestions, and every "
            "domain resume scores its own domain far above unrelated roles — evidence that scores respond to content "
            "rather than being fixed. A synthetic set this small validates behaviour, not accuracy; Section 8 describes "
            "the labelled evaluation needed before production.", SMALL)]
    s += [p("6. Testing", H1),
          p("The pytest suite (44 tests) covers parsing (PDF/DOCX/TXT, corrupted/empty/unsupported files), skill "
            "extraction edge cases (C++, Node.js, false positives, R/Go), date-range merging, education detection, JD "
            "required/preferred separation, similarity properties, domain-ranking sanity, related-skill credit, "
            "recommendations scheduling, suggestions, readiness ordering, SQLite privacy, PDF export, and a headless "
            "Streamlit <font name='DejaVuMono'>AppTest</font> that renders all eight pages in empty and demo states. A "
            "MiniLM-specific test is marked <font name='DejaVuMono'>transformer</font> and skips automatically when the model "
            "is not cached. A GitHub Actions workflow runs the suite on every push.")]
    s += [p("7. Technology choices", H1),
          table([["Concern", "Choice", "Reason"],
                 ["Embeddings", "Sentence Transformers all-MiniLM-L6-v2", "Strong quality/speed trade-off; CPU-friendly; 384-d"],
                 ["Linguistics", "spaCy en_core_web_sm", "Fast sentence splitting, NER, dependency parse for passive voice"],
                 ["Parsing", "PyMuPDF, python-docx", "Accurate text order, table support, no external services"],
                 ["Fallback", "scikit-learn TF-IDF + TruncatedSVD", "Works fully offline, deterministic"],
                 ["UI / charts", "Streamlit, Plotly", "Rapid, interactive, testable with AppTest"],
                 ["Storage", "SQLite (stdlib)", "Zero-config; swap for PostgreSQL + pgvector in production"],
                 ["Optional GenAI", "Google Gemini (google-genai)", "Coaching only; never affects scores"]],
                [30 * mm, 58 * mm, 86 * mm])]
    s += [p("8. Limitations and future work", H1),
          *bullets(["Calibration ranges and weights are heuristic → fit on labelled resume–job pairs (logistic regression "
                    "over components) and report precision@k / NDCG.",
                    "Taxonomy-bound skill extraction → add a spaCy span-categorizer trained on job postings, with embedding "
                    "nearest-neighbour mapping to canonical skills.",
                    "No OCR for scanned PDFs → add Tesseract/PaddleOCR fallback.",
                    "English only → multilingual MiniLM (paraphrase-multilingual-MiniLM-L12-v2) for Indian-language resumes.",
                    "Single-user local app → FastAPI service, authentication, PostgreSQL + pgvector, background workers."])]
    s += [p("9. Reproducibility", H1),
          p("pip install -r requirements-dev.txt<br/>python scripts/download_models.py<br/>pytest<br/>"
            "python scripts/generate_reports.py   # regenerates both PDFs with live evaluation<br/>streamlit run app.py",
            CODE)]
    build(path, "Technical Approach Report", s)


def main():
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    opportunity_report(out / "AI_Opportunity_Report.pdf")
    ev = run_evaluation()
    technical_report(out / "Technical_Approach_Report.pdf", ev)
    print(f"Reports written to {out} (engine: {ev[0]})")


if __name__ == "__main__":
    main()
