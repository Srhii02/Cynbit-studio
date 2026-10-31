from cynbit.extractor import (build_profile, detect_education, find_date_ranges, merge_spans, redact_pii,
                              split_sections)
from cynbit.skills import extract_skills, get_taxonomy


def test_taxonomy_loads():
    tax = get_taxonomy()
    assert len(tax.skills) >= 80
    assert "NLP & LLMs" in tax.categories


def test_alias_normalisation():
    hits = extract_skills("Used sklearn, k8s and HuggingFace with Node.js and ReactJS.")
    for s in ("Scikit-learn", "Kubernetes", "Transformers", "Node.js", "React"):
        assert s in hits, s


def test_special_character_skills():
    hits = extract_skills("Languages: C++, C#, Python. Set up CI/CD pipelines.")
    assert {"C++", "C#", "Python", "CI/CD"} <= set(hits)


def test_no_substring_false_positives():
    hits = extract_skills("A reactive, javascriptless approach; we rested and expressed interest in springs.")
    assert "React" not in hits
    assert "JavaScript" not in hits
    assert "Java" not in hits


def test_ambiguous_short_names_need_list_context():
    assert "R" in extract_skills("Skills: Python, R, SQL")
    assert "Go" in extract_skills("Languages: Java, Go, Rust")
    assert "Go" not in extract_skills("Go to the website and register. R is a letter.")


def test_skill_evidence_lines():
    hits = extract_skills("- Built a FastAPI service\n- Deployed it with Docker on AWS")
    assert hits["Docker"].evidence and "Docker" in hits["Docker"].evidence[0]


def test_date_ranges_and_overlap_merge():
    text = "ML Intern | May 2023 - Aug 2023\nEngineer | Jan 2024 - Dec 2024\nTA | Jun 2024 - Sep 2024"
    spans = find_date_ranges(text)
    assert len(spans) == 3
    total = merge_spans(spans)
    assert 1.2 <= total <= 1.35  # 4 months + 12 months, overlap counted once


def test_year_only_and_present():
    spans = find_date_ranges("Analyst, 2019 - 2021\nEngineer, 2022 - Present")
    assert len(spans) == 2
    assert abs(spans[0].years - 2.0) < 1e-6


def test_sections_detected(resume_texts):
    secs = split_sections(resume_texts["ananya_rao_ml_student"])
    for s in ("summary", "education", "experience", "projects", "skills"):
        assert s in secs


def test_education_detection():
    lvl, label, cgpa = detect_education("B.Tech in CSE, CGPA: 8.7/10", {})
    assert lvl == 2 and label == "Bachelor's" and cgpa == 8.7
    lvl, label, _ = detect_education("M.Tech in AI", {})
    assert label == "Master's"
    lvl, _, _ = detect_education("Proficient in MS Excel and able to be on time", {})
    assert lvl == 0  # no false positives from 'MS' / 'be'


def test_profile_from_sample(resume_texts):
    p = build_profile(resume_texts["ananya_rao_ml_student"])
    assert p.name == "Ananya Rao"
    assert {"Python", "PyTorch", "NLP", "Sentence Transformers"} <= set(p.skills)
    assert 0.4 <= p.experience_years <= 1.0  # two internships in 2025
    assert p.education_level == 2
    assert p.contact == {"email": True, "phone": True, "linkedin": True, "github": True}
    assert len(p.quantified_bullets) >= 4


def test_redact_pii():
    out = redact_pii("Mail a.b@example.com, call +91 55500 11223, see github.com/someone")
    assert "example.com" not in out and "55500" not in out and "github.com/someone" not in out
