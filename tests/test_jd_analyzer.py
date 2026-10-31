from cynbit.jd_analyzer import analyze_job


def test_title_company_seniority(job_texts):
    job = analyze_job(job_texts["ai_ml_engineer_intern"])
    assert job.title == "AI/ML Engineer Intern"
    assert job.company.startswith("Nimbus Labs")
    assert job.seniority == "Intern"


def test_required_vs_preferred(job_texts):
    job = analyze_job(job_texts["ai_ml_engineer_intern"])
    assert {"Python", "PyTorch", "spaCy", "FastAPI", "Docker"} <= set(job.required_skills)
    assert {"LangChain", "MLflow", "AWS"} <= set(job.preferred_skills)
    assert not set(job.required_skills) & set(job.preferred_skills)


def test_min_years_and_degree(job_texts):
    mlops = analyze_job(job_texts["mlops_engineer"])
    assert mlops.min_years == 2
    assert mlops.degree_label == "Bachelor's"
    intern = analyze_job(job_texts["ai_ml_engineer_intern"])
    assert intern.min_years == 0


def test_unstructured_jd():
    job = analyze_job("We need someone who knows Python and SQL and has 3 years of experience building dashboards "
                      "in Tableau. Docker is a plus.")
    assert {"Python", "SQL", "Tableau"} <= set(job.required_skills)
    assert "Docker" in job.preferred_skills
    assert job.min_years == 3


def test_key_phrases(job_texts):
    job = analyze_job(job_texts["data_analyst_intern"])
    assert len(job.key_phrases) > 3
