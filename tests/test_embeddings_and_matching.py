"""Checks that similarity and scores are real (data-dependent), bounded and sensible."""
import numpy as np
import pytest

from cynbit.extractor import build_profile
from cynbit.jd_analyzer import analyze_job
from cynbit.matcher import match, verdict_for


def test_tfidf_similarity_properties(tfidf_engine):
    a = "Built machine learning models in Python with scikit-learn"
    sim = tfidf_engine.similarity_matrix([a, "Baked sourdough bread and pastries"], [a, "Trained ML models using sklearn"])
    assert sim.shape == (2, 2)
    assert sim[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert sim[0, 1] > sim[1, 1]  # ML sentence closer to ML sentence than baking is
    assert np.all(sim >= -1e-9) and np.all(sim <= 1 + 1e-9)


def test_tfidf_alias_concepts(tfidf_engine):
    sim = tfidf_engine.similarity_matrix(["experience with k8s"], ["Kubernetes clusters", "Excel spreadsheets"])
    assert sim[0, 0] > sim[0, 1]


@pytest.mark.transformer
def test_minilm_semantics(transformer_engine):
    e = transformer_engine
    v = e.encode(["hello world"])
    assert v.shape == (1, 384)
    sim = e.similarity_matrix(["I trained neural networks for image recognition"],
                              ["Built deep learning computer vision models", "Managed a retail cash register"])
    assert sim[0, 0] > sim[0, 1] + 0.2


def test_scores_bounded_and_explained(resume_texts, job_texts, engine):
    p = build_profile(resume_texts["ananya_rao_ml_student"])
    r = match(p, analyze_job(job_texts["ai_ml_engineer_intern"]), engine)
    for v in (r.overall, r.semantic, r.skills, r.experience, r.education):
        assert 0 <= v <= 100
    assert r.explanations and r.alignments
    assert r.verdict == verdict_for(r.overall)


def test_relevant_job_outscores_irrelevant(resume_texts, job_texts, engine):
    """The ML student must match the ML internship better than the Java backend role (real semantics)."""
    p = build_profile(resume_texts["ananya_rao_ml_student"])
    ml = match(p, analyze_job(job_texts["ai_ml_engineer_intern"]), engine)
    java = match(p, analyze_job(job_texts["backend_java_developer"]), engine)
    assert ml.overall > java.overall + 15
    assert ml.semantic > java.semantic


def test_each_sample_resume_best_fits_its_domain(resume_texts, job_texts, engine):
    expected = {"ananya_rao_ml_student": {"ai_ml_engineer_intern", "nlp_engineer_junior"},
                "karthik_menon_fullstack": {"full_stack_developer"},
                "meera_iyer_data_analyst": {"data_analyst_intern"}}
    for rname, good in expected.items():
        p = build_profile(resume_texts[rname])
        scores = {j: match(p, analyze_job(t), engine).overall for j, t in job_texts.items()}
        assert max(scores, key=scores.get) in good, (rname, scores)


def test_missing_and_related_skills(resume_texts, job_texts, engine):
    p = build_profile(resume_texts["meera_iyer_data_analyst"])
    r = match(p, analyze_job(job_texts["mlops_engineer"]), engine)
    missing = {s.skill for s in r.missing}
    assert {"Kubernetes", "Terraform", "Docker"} <= missing
    assert r.overall < 50


def test_related_credit():
    p = build_profile("Skills\nPython, TensorFlow, Keras, Pandas\n\nExperience\n- Trained TensorFlow models for 2 years "
                      "on image data and deployed them.\n" * 2)
    job = analyze_job("Requirements\n- Strong PyTorch experience\n- Python")
    from cynbit.embeddings import get_engine

    r = match(p, job, get_engine("tfidf"))
    statuses = {s.skill: s.status for s in r.skill_matches}
    assert statuses["Python"] == "matched"
    assert statuses["PyTorch"] == "related"


def test_experience_shortfall_lowers_score(resume_texts, job_texts, engine):
    p = build_profile(resume_texts["ananya_rao_ml_student"])
    r = match(p, analyze_job(job_texts["mlops_engineer"]), engine)  # needs 2+ years
    assert r.experience < 60
