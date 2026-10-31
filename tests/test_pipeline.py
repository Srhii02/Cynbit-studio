"""Ranking, recommendations, suggestions, readiness, persistence and PDF export."""
import pymupdf as fitz

from cynbit import analyze
from cynbit.database import HistoryStore
from cynbit.extractor import build_profile
from cynbit.ranking import ranking_table, skill_matrix
from cynbit.recommender import recommend, resources_for
from cynbit.report_export import build_candidate_report
from cynbit.suggestions import analyze_resume


def test_full_pipeline_ranking(resume_texts, job_texts, engine):
    a = analyze(resume_texts["ananya_rao_ml_student"], list(job_texts.values()), engine=engine)
    scores = [r.overall for _, r in a.pairs]
    assert scores == sorted(scores, reverse=True)
    df = ranking_table(a.pairs)
    assert list(df["Rank"]) == list(range(1, len(job_texts) + 1))
    mat = skill_matrix(a.pairs)
    assert not mat.empty and set(mat.columns) <= {r.job_title[:28] for _, r in a.pairs}


def test_recommendations_prioritised_and_scheduled(resume_texts, job_texts, engine):
    a = analyze(resume_texts["meera_iyer_data_analyst"], list(job_texts.values()), engine=engine, hours_per_week=10)
    recs = a.recommendations
    assert recs
    assert [r.priority for r in recs] == sorted([r.priority for r in recs], reverse=True)
    assert all(r.resources for r in recs)
    assert all(r.resources[0]["url"].startswith("https://") for r in recs)
    assert recs[0].week_start == 1
    assert all(r.week_end >= r.week_start for r in recs)


def test_resources_fallback_to_category():
    res, project = resources_for("Unknown Skill", "Web Backend")
    assert res and project


def test_suggestions_flag_weak_resume(resume_texts):
    weak_q, weak = analyze_resume(build_profile(resume_texts["rohan_das_needs_work"]))
    strong_q, _ = analyze_resume(build_profile(resume_texts["ananya_rao_ml_student"]))
    titles = " ".join(s.title for s in weak)
    assert "weak phrase" in titles.lower()
    assert "too thin" in titles.lower()
    assert any("buzzword" in s.title.lower() for s in weak)
    assert strong_q > weak_q + 30


def test_readiness_ordering(resume_texts, job_texts, engine):
    strong = analyze(resume_texts["ananya_rao_ml_student"], list(job_texts.values()), engine=engine)
    weak = analyze(resume_texts["rohan_das_needs_work"], list(job_texts.values()), engine=engine)
    assert 0 <= weak.readiness.index < strong.readiness.index <= 100
    assert set(strong.readiness.components) == {"Market fit", "Skill depth", "Resume quality",
                                                "Practical experience", "Profile completeness"}


def test_history_store(tmp_path, resume_texts, job_texts, engine):
    db = HistoryStore(tmp_path / "t.db")
    text = resume_texts["karthik_menon_fullstack"]
    a = analyze(text, [job_texts["full_stack_developer"]], engine=engine)
    rid = db.upsert_resume(text, "k.docx", a.profile.name, a.profile.skill_names, 1.5, "Bachelor's", a.quality)
    assert db.upsert_resume(text, "k.docx", "x", [], 1.5, "", 0) == rid  # idempotent on fingerprint
    db.add_analysis(rid, job_texts["full_stack_developer"], a.pairs[0][1])
    hist = db.history()
    assert len(hist) == 1 and hist.iloc[0]["overall"] == a.pairs[0][1].overall
    # privacy: raw resume text must not be stored anywhere in the database
    raw = (tmp_path / "t.db").read_bytes()
    assert b"BrightCart" not in raw
    assert len(db.resume_trend(text)) == 1
    db.clear()
    assert db.history().empty


def test_pdf_report(resume_texts, job_texts, engine):
    a = analyze(resume_texts["ananya_rao_ml_student"], list(job_texts.values())[:3], engine=engine)
    pdf = build_candidate_report(a.profile, [r for _, r in a.pairs], a.recommendations, a.readiness, a.quality,
                                 a.suggestions)
    assert pdf[:4] == b"%PDF"
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        text = "".join(p.get_text() for p in doc)
    assert "Career Match Report" in text and "learning plan" in text.lower()


def test_recommend_empty():
    assert recommend([]) == []
