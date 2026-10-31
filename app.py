"""Cynbit Studio – AI Resume & Job Matching Intelligence (Streamlit UI).

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from cynbit import analyze, charts, gemini  # noqa: E402
from cynbit.config import SAMPLES_DIR, settings  # noqa: E402
from cynbit.database import HistoryStore  # noqa: E402
from cynbit.embeddings import get_engine  # noqa: E402
from cynbit.nlp import get_nlp, has_full_model  # noqa: E402
from cynbit.parser import ParsingError, parse_bytes, parse_file  # noqa: E402
from cynbit.ranking import ranking_table, skill_matrix  # noqa: E402
from cynbit.report_export import build_candidate_report  # noqa: E402
from cynbit.skills import get_taxonomy  # noqa: E402

st.set_page_config(page_title="Cynbit Studio · Resume & Job Intelligence", page_icon="🎯", layout="wide")

CSS = """
<style>
:root { --brand:#4F46E5; --brand2:#0D9488; --muted:#64748B; }
.block-container { padding-top: 1.6rem; max-width: 1250px; }
.hero { background: linear-gradient(120deg,#4F46E5 0%,#7C3AED 55%,#0D9488 100%); color:#fff; border-radius:18px;
        padding: 26px 30px; margin-bottom: 18px; }
.hero h1 { color:#fff; font-size: 2rem; margin: 0 0 4px 0; }
.hero p { color: rgba(255,255,255,.88); margin:0; font-size: 1.02rem; }
.card { border:1px solid rgba(148,163,184,.35); border-radius:14px; padding:16px 18px; margin-bottom:12px;
        background: rgba(148,163,184,.06); }
.card h4 { margin: 0 0 6px 0; }
.chip { display:inline-block; padding:3px 10px; margin:3px 4px 3px 0; border-radius:999px; font-size:.82rem;
        border:1px solid transparent; }
.chip.ok { background:#CCFBF1; color:#115E59; } .chip.rel { background:#FEF3C7; color:#92400E; }
.chip.miss { background:#FFE4E6; color:#9F1239; } .chip.neu { background:#E0E7FF; color:#3730A3; }
.sev-high { border-left: 4px solid #E11D48; } .sev-medium { border-left: 4px solid #D97706; }
.sev-low { border-left: 4px solid #64748B; }
.small { color: var(--muted); font-size: .86rem; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading NLP models (first run downloads all-MiniLM-L6-v2)…")
def load_models():
    engine = get_engine()
    get_nlp()
    return engine


@st.cache_resource
def store() -> HistoryStore:
    return HistoryStore()


@st.cache_data(show_spinner="Running semantic analysis…", max_entries=64)
def run_analysis(resume_text: str, job_texts: tuple[str, ...], hours: int):
    return analyze(resume_text, list(job_texts), hours_per_week=hours, engine=load_models())


def sample_jobs() -> dict[str, str]:
    return {p.stem.replace("_", " ").title(): p.read_text(encoding="utf-8")
            for p in sorted((SAMPLES_DIR / "jobs").glob("*.txt"))}


def sample_resumes() -> dict[str, Path]:
    return {p.name: p for p in sorted((SAMPLES_DIR / "resumes").glob("*")) if p.suffix in {".pdf", ".docx", ".txt"}}


def chips(items, kind="neu") -> str:
    return "".join(f'<span class="chip {kind}">{i}</span>' for i in items) or '<span class="small">—</span>'


# ----------------------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("resume_text", "")
ss.setdefault("resume_name", "")
ss.setdefault("jobs", {})  # name -> text
ss.setdefault("hours", 8)
ss.setdefault("saved_keys", set())


def set_resume(text: str, name: str):
    ss.resume_text, ss.resume_name = text, name


def load_demo():
    p = sample_resumes()["ananya_rao_ml_student.pdf"]
    set_resume(parse_file(p).text, p.name)
    ss.jobs = sample_jobs()


def current():
    if not ss.resume_text or not ss.jobs:
        return None
    return run_analysis(ss.resume_text, tuple(ss.jobs.values()), ss.hours)


def save_history(a):
    if not settings.store_history:
        return
    key = (hash(ss.resume_text), tuple(sorted(ss.jobs)))
    if key in ss.saved_keys:
        return
    db = store()
    rid = db.upsert_resume(ss.resume_text, ss.resume_name, a.profile.name, a.profile.skill_names,
                           a.profile.experience_years, a.profile.education_label, a.quality)
    for job, r in a.pairs:
        db.add_analysis(rid, job.text, r)
    ss.saved_keys.add(key)


def need_inputs(resume=True, jobs=True) -> bool:
    missing = []
    if resume and not ss.resume_text:
        missing.append("a resume (📄 Resume)")
    if jobs and not ss.jobs:
        missing.append("at least one job description (🎯 Job Match or 📊 Compare Jobs)")
    if missing:
        st.info("Add " + " and ".join(missing) + " — or click **Load demo data** in the sidebar.")
        return True
    return False


# ----------------------------------------------------------------------------- sidebar
engine = load_models()
with st.sidebar:
    st.markdown("### 🎯 Cynbit Studio")
    st.caption("AI Resume & Job Matching Intelligence")
    page = st.radio("Navigate", ["🏠 Overview", "📄 Resume", "🎯 Job Match", "📊 Compare Jobs",
                                 "🚀 Readiness Dashboard", "✍️ Resume Improvement", "🕘 History",
                                 "🔗 Cynbit Integration"], label_visibility="collapsed")
    st.divider()
    if st.button("⚡ Load demo data", use_container_width=True):
        load_demo()
        st.rerun()
    ss.hours = st.slider("Study hours / week", 2, 30, ss.hours)
    st.markdown(f"**Resume:** {ss.resume_name or '—'}  \n**Jobs loaded:** {len(ss.jobs)}")
    st.divider()
    st.caption(f"Embedding engine: **{engine.name}**")
    st.caption(f"spaCy: **{settings.spacy_model if has_full_model() else 'blank en (no NER)'}**")
    st.caption(f"Gemini coaching: **{'on' if gemini.is_available() else 'off (optional)'}**")
    if not engine.is_transformer:
        st.warning("all-MiniLM-L6-v2 could not be loaded (offline?). Using the TF-IDF + LSA fallback. "
                   "Run `python scripts/download_models.py` with internet access.", icon="⚠️")


# ----------------------------------------------------------------------------- pages
def page_overview():
    st.markdown('<div class="hero"><h1>Cynbit Studio</h1><p>Upload a resume, add job descriptions, and get an '
                'explainable semantic match score, the exact skills you are missing, a personalised learning plan '
                'and a career-readiness dashboard.</p></div>', unsafe_allow_html=True)
    c = st.columns(4)
    steps = [("1 · Parse", "PDF/DOCX/TXT parsed with PyMuPDF & python-docx; sections, skills, experience and "
                           "education extracted with spaCy + a 90-skill taxonomy."),
             ("2 · Understand the job", "Required vs preferred skills, seniority, years and degree requirements."),
             ("3 · Match semantically", "all-MiniLM-L6-v2 embeddings: document similarity + requirement-by-requirement "
                                        "evidence alignment, blended with skill coverage."),
             ("4 · Act", "Ranked jobs, gap analysis, learning roadmap, resume fixes and a PDF report.")]
    for col, (h, t) in zip(c, steps):
        col.markdown(f'<div class="card"><h4>{h}</h4><div class="small">{t}</div></div>', unsafe_allow_html=True)
    st.markdown("#### Quick start")
    st.write("Click **⚡ Load demo data** in the sidebar to analyse a synthetic student resume against six sample "
             "job descriptions, or start with **📄 Resume**.")
    a = current()
    if a:
        best_job, best = a.pairs[0]
        k = st.columns(4)
        k[0].metric("Best match", f"{best.overall:.0f}/100", best.job_title[:24])
        k[1].metric("Career readiness", f"{a.readiness.index:.0f}/100", a.readiness.level)
        k[2].metric("Resume quality", f"{a.quality:.0f}/100")
        k[3].metric("Skills detected", len(a.profile.skills))


def page_resume():
    st.header("📄 Resume")
    left, right = st.columns([3, 2])
    with left:
        up = st.file_uploader("Upload your resume (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
        if up is not None:
            if up.size > settings.max_upload_mb * 1024 * 1024:
                st.error(f"File is larger than {settings.max_upload_mb} MB.")
            else:
                try:
                    doc = parse_bytes(up.getvalue(), up.name)
                    if doc.text != ss.resume_text:
                        set_resume(doc.text, up.name)
                    st.success(f"Parsed **{up.name}** · {doc.pages} page(s) · {doc.word_count} words")
                except ParsingError as e:
                    st.error(str(e))
    with right:
        samples = sample_resumes()
        choice = st.selectbox("…or use a synthetic sample", ["—"] + list(samples))
        if st.button("Load sample", disabled=choice == "—"):
            set_resume(parse_file(samples[choice]).text, choice)
            st.rerun()
    if not ss.resume_text:
        return
    from cynbit.extractor import build_profile

    p = build_profile(ss.resume_text)
    m = st.columns(5)
    m[0].metric("Skills found", len(p.skills))
    total_months = round(p.experience_years * 12)
    m[1].metric("Experience", f"{p.experience_years:g} yrs" if total_months >= 12 or not total_months
                else f"{total_months} months")
    m[2].metric("Internships", p.internships,
                f"{p.internship_months:g} months" if p.internship_months else None, delta_color="off")
    m[3].metric("Education", p.education_label)
    m[4].metric("CGPA (/10)", f"{p.cgpa:.2f}" if p.cgpa else "Not stated",
                p.cgpa_raw if p.cgpa_raw and not p.cgpa else None, delta_color="off")
    st.markdown("#### Skills by category")
    tax = get_taxonomy()
    for cat, names in sorted(tax.by_category(p.skills).items()):
        st.markdown(f"**{cat}** &nbsp; {chips(names)}", unsafe_allow_html=True)
    st.markdown("#### Experience & internships")
    if p.experiences:
        st.dataframe(pd.DataFrame([{
            "Company": e.company or "Not stated", "Role": e.title or "Not stated",
            "Type": "Internship" if e.is_internship else "Work",
            "Start": e.start_label or "Not stated", "End": e.end_label or "Not stated",
            "Duration": e.duration_label, "As written": e.date_text or "—"} for e in p.experiences]),
            hide_index=True, use_container_width=True)
        st.caption("Durations are month-inclusive (May–Aug = 4 months); overlapping roles are counted once. "
                   "Roles without dates are listed but add no time.")
    else:
        st.caption("No work experience or internships detected. Use a heading such as 'Experience' or "
                   "'Internships' and include dates like 'May 2025 – Aug 2025'.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Detected sections")
        st.markdown(chips([s.title() for s in p.sections if s != "header"]), unsafe_allow_html=True)
        st.markdown("#### Contact signals")
        st.markdown(chips([k for k, v in p.contact.items() if v], "ok") + chips([k for k, v in p.contact.items() if not v], "miss"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown("#### Organisations (spaCy NER)")
        st.markdown(chips(p.organizations), unsafe_allow_html=True)
    with st.expander("Skill evidence (where each skill appears)"):
        st.dataframe(pd.DataFrame([{"Skill": h.name, "Category": h.category, "Mentions": h.count,
                                    "Evidence": " | ".join(h.evidence)} for h in p.skills.values()]),
                     hide_index=True, use_container_width=True)
    with st.expander("Extracted text"):
        st.text(ss.resume_text[:8000])


def _job_input(key: str) -> tuple[str, str] | None:
    tabs = st.tabs(["Paste", "Upload", "Sample"])
    with tabs[0]:
        txt = st.text_area("Job description", height=220, key=f"{key}_txt",
                           placeholder="Paste the full job description here…")
        name = st.text_input("Label (optional)", key=f"{key}_name")
        if st.button("Add job", key=f"{key}_add", disabled=len(txt.split()) < 15):
            return (name or txt.strip().split("\n")[0][:50], txt)
    with tabs[1]:
        files = st.file_uploader("Upload JD files", type=["txt", "pdf", "docx"], accept_multiple_files=True,
                                 key=f"{key}_files")
        if files and st.button("Add uploaded", key=f"{key}_addf"):
            for f in files:
                try:
                    ss.jobs[Path(f.name).stem] = parse_bytes(f.getvalue(), f.name).text
                except ParsingError as e:
                    st.error(f"{f.name}: {e}")
            st.rerun()
    with tabs[2]:
        sj = sample_jobs()
        picks = st.multiselect("Sample job descriptions", list(sj), key=f"{key}_samples")
        if st.button("Add samples", key=f"{key}_adds", disabled=not picks):
            for p in picks:
                ss.jobs[p] = sj[p]
            st.rerun()
    return None


def _skill_section(r):
    c = st.columns(3)
    c[0].markdown("**✅ Matched**<br>" + chips([s.skill for s in r.matched], "ok"), unsafe_allow_html=True)
    c[1].markdown("**🟡 Related (partial credit)**<br>" + chips([f"{s.skill} ← {s.via}" for s in r.related], "rel"),
                  unsafe_allow_html=True)
    c[2].markdown("**❌ Missing**<br>" + chips([f"{s.skill}{' *' if s.importance == 'required' else ''}"
                                               for s in r.missing], "miss")
                  + '<div class="small">* required</div>', unsafe_allow_html=True)


def page_match():
    st.header("🎯 Job Match")
    with st.expander("➕ Add a job description", expanded=not ss.jobs):
        added = _job_input("single")
    if added:
        ss.jobs[added[0]] = added[1]
        st.rerun()
    if need_inputs():
        return
    a = current()
    save_history(a)
    titles = [f"{r.job_title} — {r.overall:.0f}/100" for _, r in a.pairs]
    idx = st.selectbox("Job to inspect", range(len(titles)), format_func=lambda i: titles[i])
    job, r = a.pairs[idx]
    g, b = st.columns([2, 3])
    g.plotly_chart(charts.gauge(r.overall, r.verdict), use_container_width=True)
    b.plotly_chart(charts.component_bars(r), use_container_width=True)
    st.markdown("#### Why this score")
    for e in r.explanations:
        st.markdown(f"- {e}")
    st.markdown("#### Skills")
    _skill_section(r)
    with st.expander("Job analysis", expanded=False):
        jc = st.columns(4)
        jc[0].metric("Seniority", job.seniority)
        jc[1].metric("Min. experience", f"{job.min_years:g} yrs" if job.min_years else "—")
        jc[2].metric("Degree", job.degree_label)
        jc[3].metric("Skills requested", len(job.all_skills))
        st.markdown("**Key phrases (spaCy noun chunks):** " + chips(job.key_phrases), unsafe_allow_html=True)
    st.markdown("#### Requirement-by-requirement evidence")
    st.dataframe(pd.DataFrame([{"Job requirement": al.requirement, "Best resume evidence": al.evidence,
                                "Similarity": round(al.similarity * 100)} for al in r.alignments]),
                 hide_index=True, use_container_width=True,
                 column_config={"Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=100,
                                                                              format="%d%%")})
    from cynbit.recommender import recommend

    recs = recommend([r], hours_per_week=ss.hours, top_n=8)
    if recs:
        st.markdown("#### What to learn for this role")
        for rec in recs:
            res = " · ".join(f"[{x['title']}]({x['url']})" for x in rec.resources[:2])
            st.markdown(f"- **{rec.skill}** ({rec.priority_label}, ~{rec.est_hours} h) — {res}")
    if gemini.is_available():
        if st.button("✨ Ask Gemini for rewritten bullets"):
            with st.spinner("Asking Gemini (PII redacted)…"):
                st.markdown(gemini.coach(ss.resume_text, job.text, [s.skill for s in r.missing], r.overall))
    if st.button("🗑️ Remove this job"):
        key = [k for k, v in ss.jobs.items() if v == job.text]
        for k in key:
            ss.jobs.pop(k)
        st.rerun()


def page_compare():
    st.header("📊 Compare & Rank Jobs")
    with st.expander("Add job descriptions", expanded=not ss.jobs):
        added = _job_input("multi")
        if added:
            ss.jobs[added[0]] = added[1]
            st.rerun()
    if ss.jobs:
        st.markdown("**Loaded:** " + chips(list(ss.jobs)), unsafe_allow_html=True)
        if st.button("Clear all jobs"):
            ss.jobs = {}
            st.rerun()
    if need_inputs():
        return
    a = current()
    save_history(a)
    df = ranking_table(a.pairs)
    st.plotly_chart(charts.ranking_bars(df), use_container_width=True)
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"Overall": st.column_config.ProgressColumn("Overall", min_value=0, max_value=100,
                                                                           format="%.0f")})
    st.download_button("⬇️ Download ranking (CSV)", df.to_csv(index=False).encode(), "job_ranking.csv", "text/csv")
    mat = skill_matrix(a.pairs)
    if not mat.empty:
        st.plotly_chart(charts.skill_heatmap(mat), use_container_width=True)


def page_dashboard():
    st.header("🚀 Career Readiness Dashboard")
    if need_inputs():
        return
    a = current()
    save_history(a)
    rd = a.readiness
    k = st.columns(5)
    k[0].metric("Readiness index", f"{rd.index:.0f}/100", rd.level)
    k[1].metric("Best match", f"{a.pairs[0][1].overall:.0f}", a.pairs[0][1].job_title[:22])
    k[2].metric("Resume quality", f"{a.quality:.0f}")
    k[3].metric("Skills", len(a.profile.skills))
    total_h = sum(r.est_hours for r in a.recommendations)
    k[4].metric("Learning plan", f"{total_h} h", f"≈ {a.recommendations[-1].week_end if a.recommendations else 0} weeks")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.radar(rd.components), use_container_width=True)
    if rd.category_coverage:
        c2.plotly_chart(charts.coverage_bars(rd.category_coverage), use_container_width=True)
    s1, s2 = st.columns(2)
    s1.markdown("**Strengths** " + chips(rd.strengths, "ok"), unsafe_allow_html=True)
    s2.markdown("**Focus areas** " + chips(rd.focus_areas, "miss"), unsafe_allow_html=True)
    if a.recommendations:
        st.plotly_chart(charts.roadmap(a.recommendations), use_container_width=True)
        st.caption("Bars: 🟥 high · 🟧 medium · ⬜ low priority. Plan targets your 3 best-matching jobs.")
        st.markdown("#### Personalised learning plan")
        for rec in a.recommendations:
            with st.container(border=True):
                st.markdown(f"**{rec.skill}** · {rec.category} · <span class='chip {'miss' if rec.priority_label == 'High' else 'rel'}'>"
                            f"{rec.priority_label} priority</span> · ~{rec.est_hours} h · weeks {rec.week_start}–{rec.week_end}",
                            unsafe_allow_html=True)
                note = f"you already know **{rec.via}**, so this is a shorter bridge. " if rec.status == "related" else ""
                st.markdown(f"Needed by: {', '.join(rec.demanded_by[:4])}. {note}")
                st.markdown(" · ".join(f"[{x['title']}]({x['url']})" for x in rec.resources[:3]))
                if rec.project:
                    st.caption(f"Portfolio project: {rec.project}")
    pdf = build_candidate_report(a.profile, [r for _, r in a.pairs], a.recommendations, rd, a.quality, a.suggestions)
    st.download_button("⬇️ Download my career report (PDF)", pdf, "cynbit_career_report.pdf", "application/pdf",
                       type="primary")


def page_improve():
    st.header("✍️ Resume Improvement")
    if need_inputs(jobs=False):
        return
    if ss.jobs:
        a = current()
        quality, sugg = a.quality, a.suggestions
        st.caption(f"Targeting keywords from your best-matching job: **{a.pairs[0][1].job_title}**")
    else:
        from cynbit.extractor import build_profile
        from cynbit.suggestions import analyze_resume

        quality, sugg = analyze_resume(build_profile(ss.resume_text))
    c1, c2 = st.columns([1, 2])
    c1.plotly_chart(charts.gauge(quality, "Resume quality"), use_container_width=True)
    counts = pd.Series([s.severity for s in sugg]).value_counts()
    c2.markdown(f"**{len(sugg)} suggestions** · 🔴 {counts.get('high', 0)} high · 🟠 {counts.get('medium', 0)} "
                f"medium · ⚪ {counts.get('low', 0)} low")
    c2.caption("Checks: section structure, contact links, length, quantified impact, action verbs, weak phrases, "
               "pronouns, buzzwords, passive voice (spaCy), skill evidence and job keywords.")
    for s in sugg:
        ex = f"<div class='small'>Example: {s.example}</div>" if s.example else ""
        st.markdown(f"<div class='card sev-{s.severity}'><b>{s.title}</b> <span class='small'>· {s.area} · "
                    f"{s.severity}</span><div>{s.detail}</div>{ex}</div>", unsafe_allow_html=True)


def page_history():
    st.header("🕘 History")
    st.caption("Stored locally in SQLite. Only a fingerprint of your resume is saved — never the raw text.")
    df = store().history()
    if df.empty:
        st.info("No analyses yet.")
        return
    if ss.resume_text:
        tr = store().resume_trend(ss.resume_text)
        if len(tr) > 1:
            st.plotly_chart(charts.trend(tr), use_container_width=True)
    st.dataframe(df.drop(columns=["missing_json"]), hide_index=True, use_container_width=True)
    if st.button("Delete all history"):
        store().clear()
        ss.saved_keys = set()
        st.rerun()


def page_integration():
    st.header("🔗 Integrating with Cynbit Studio")
    st.markdown("""
This prototype is deliberately built as a **framework-agnostic Python package (`src/cynbit`) with a thin Streamlit UI**,
so the same engine can be embedded in a production Cynbit Studio product without rewriting the AI logic.

| Phase | What | How |
|---|---|---|
| 1 · API | Wrap `cynbit.analyze()` in a FastAPI service (`/parse`, `/match`, `/rank`, `/recommend`) | Docker image, model loaded once per worker |
| 2 · Embed | Add a “Resume Match” widget to Cynbit Studio pages / client portals | REST + JWT/SSO, results rendered by the existing web front end |
| 3 · Data | Replace SQLite with PostgreSQL + pgvector; store job embeddings once, match in milliseconds | Nightly re-indexing of job feeds |
| 4 · Intelligence | Learn weights from recruiter feedback (shortlisted / rejected) | Logistic regression or LightGBM over the score components |
| 5 · GenAI | Optional Gemini / LLM coaching, cover-letter drafts, interview prep | PII redaction already built in (`redact_pii`) |

**Value for Cynbit:** a reusable AI module for placement cells, colleges, recruiters and HR-tech clients; SEO-friendly
career content (skills-gap insights); and a showcase of explainable, privacy-aware AI.
See `docs/AI_Opportunity_Report.pdf` for the full proposal.
""")


PAGES = {"🏠 Overview": page_overview, "📄 Resume": page_resume, "🎯 Job Match": page_match,
         "📊 Compare Jobs": page_compare, "🚀 Readiness Dashboard": page_dashboard,
         "✍️ Resume Improvement": page_improve, "🕘 History": page_history,
         "🔗 Cynbit Integration": page_integration}
PAGES[page]()
st.markdown("<div class='small' style='margin-top:30px'>Cynbit Studio prototype · scores are explainable guidance, "
            "not hiring decisions · sample data is synthetic.</div>", unsafe_allow_html=True)