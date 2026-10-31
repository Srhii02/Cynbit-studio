"""Experience / internship extraction: company, title, dates, duration, and no fabrication."""
from datetime import date

import pytest

from cynbit.experience import extract_entries, find_ranges, merged_months, stated_duration_months
from cynbit.extractor import build_profile, extract_cgpa

TODAY = date(2026, 10, 30)

# Mirrors the reported resume: company BLP Industry.AI, role Data Analyst, May–August 2025, internship work.
BLP_RESUME = """Test Candidate
candidate@example.com | +91 55500 12345

EDUCATION
B.Tech in Computer Science and Engineering, Example Institute (fictional), 2022 - 2026

PROFESSIONAL EXPERIENCE
Company: BLP Industry.AI
Role: Data Analyst
Duration: May–August 2025
Internship responsibilities:
- Cleaned and analysed 50,000 sales records using Python, Pandas and SQL.
- Built Power BI dashboards used for weekly reporting.
Project details:
- Customer churn analysis with scikit-learn; presented findings to the analytics lead.

SKILLS
Python, SQL, Pandas, Power BI, Excel
"""


# ------------------------------------------------------------------------------------------- the reported bug
def test_reported_resume_experience_and_internship():
    p = build_profile(BLP_RESUME)
    assert p.internships == 1
    assert len(p.experiences) == 1
    e = p.experiences[0]
    assert e.company == "BLP Industry.AI"
    assert e.title == "Data Analyst"
    assert (e.start_label, e.end_label) == ("May 2025", "Aug 2025")
    assert e.months == 4 and e.duration_label == "4 months"
    assert e.is_internship
    assert p.experience_years == pytest.approx(4 / 12, abs=0.01)
    assert len(p.experience_spans) == 1
    # skills and education still work
    assert {"Python", "SQL", "Power BI"} <= set(p.skills)
    assert p.education_label == "Bachelor's"


@pytest.mark.parametrize("layout", [
    "WORK EXPERIENCE & INTERNSHIPS\nData Analyst Intern | BLP Industry.AI | May–August 2025\n- Built dashboards.",
    "EXPERIENCE\nData Analyst, BLP Industry.AI (May 2025 – Aug 2025)\n- Internship: built dashboards.",
    "INTERNSHIPS\nData Analyst\nBLP Industry.AI, Hyderabad\n05/2025 - 08/2025\n- Analysed data.",
    "Experience (Internship)\nBLP Industry.AI — Data Analyst — May 2025 to August 2025\n- Analysed data.",
    "Internship Experience:\nData Analyst at BLP Industry.AI | 2025-05 – 2025-08\n- Analysed data.",
])
def test_layout_variants(layout):
    p = build_profile("Test Candidate\n" + layout + "\n\nSKILLS\nPython, SQL")
    assert p.internships == 1, layout
    e = p.experiences[0]
    assert e.company.startswith("BLP Industry.AI"), (layout, e)
    assert e.title.startswith("Data Analyst"), (layout, e)
    assert e.months == 4, (layout, e)


# ------------------------------------------------------------------------------------------- date formats
@pytest.mark.parametrize("text,start,end,months", [
    ("May 2025 – Aug 2025", (2025, 5), (2025, 8), 4),
    ("May–August 2025", (2025, 5), (2025, 8), 4),
    ("May - Aug, 2025", (2025, 5), (2025, 8), 4),
    ("Nov–Feb 2025", (2024, 11), (2025, 2), 4),
    ("Dec 2024 - Feb 2025", (2024, 12), (2025, 2), 3),
    ("May'25 – Aug'25", (2025, 5), (2025, 8), 4),
    ("May, 2025 to Aug, 2025", (2025, 5), (2025, 8), 4),
    ("05/2025 - 08/2025", (2025, 5), (2025, 8), 4),
    ("2025-05 to 2025-08", (2025, 5), (2025, 8), 4),
    ("01/05/2025 – 31/08/2025", (2025, 5), (2025, 8), 4),
    ("May 1, 2025 – August 31, 2025", (2025, 5), (2025, 8), 4),
    ("1st May 2025 - 31st Aug 2025", (2025, 5), (2025, 8), 4),
    ("June 2024 – Present", (2024, 6), (2026, 10), 29),
    ("Jan 2026 - Current", (2026, 1), (2026, 10), 10),
    ("Mar 2026 till date", (2026, 3), (2026, 10), 8),
    ("Jan 2024 to date", (2024, 1), (2026, 10), 34),
])
def test_date_formats(text, start, end, months):
    rs = find_ranges(text, TODAY)
    assert len(rs) == 1, text
    assert rs[0].start == start and rs[0].end == end and rs[0].months == months, (text, rs[0])


def test_year_only_range():
    r = find_ranges("Analyst, 2019 - 2021", TODAY)[0]
    assert r.year_precision and r.months == 24


def test_future_end_capped_and_future_start_ignored():
    assert find_ranges("May 2026 - Dec 2026", TODAY)[0].end == (2026, 10)
    assert find_ranges("Jan 2027 - Jun 2027", TODAY) == []


@pytest.mark.parametrize("text", [
    "CGPA: 8.7/10", "Phone: 98-765-43210", "Version 2.1 - 3.4", "Built 3 dashboards - saved 20 hours",
    "Aug 2025 - Jun 2025",  # end before start
])
def test_not_dates(text):
    assert find_ranges(text, TODAY) == []


def test_overlapping_roles_counted_once():
    entries = extract_entries(
        "Data Analyst Intern | Alpha Labs | Jan 2025 - Jun 2025\n- Did work.\n"
        "Research Assistant | Beta University | Apr 2025 - Aug 2025\n- Did research.", today=TODAY)
    assert len(entries) == 2
    assert merged_months(entries) == 8  # Jan..Aug


def test_explicit_duration_without_dates():
    entries = extract_entries("Data Analyst Intern, BLP Industry.AI\nDuration: 3 months\n- Built reports.", today=TODAY)
    assert entries[0].months == 3 and entries[0].date_range is None
    assert stated_duration_months("8 weeks") == pytest.approx(1.8, abs=0.1)
    assert stated_duration_months("6-month internship") == 6


# ------------------------------------------------------------------------------------------- no fabrication
def test_missing_dates_are_not_fabricated():
    p = build_profile("Test Candidate\nEXPERIENCE\nData Analyst Intern\nBLP Industry.AI\n- Built dashboards in Excel.\n"
                      "\nSKILLS\nExcel, SQL")
    assert p.internships == 1
    e = p.experiences[0]
    assert e.company == "BLP Industry.AI" and e.title == "Data Analyst Intern"
    assert e.date_range is None and e.start_label is None and e.end_label is None
    assert e.months == 0 and e.duration_label == "Not stated"
    assert p.experience_years == 0


def test_education_dates_not_counted_as_experience():
    p = build_profile("Test Candidate\nEDUCATION\nB.Tech in CSE, 2021 - 2025\nSKILLS\nPython, SQL, Excel, Git")
    assert p.experience_years == 0 and p.internships == 0 and p.experiences == []


def test_objective_mention_is_not_an_internship():
    p = build_profile("Test Candidate\nOBJECTIVE\nSeeking an internship in data analytics.\nSKILLS\nPython, SQL")
    assert p.internships == 0


def test_full_time_job_in_mixed_section_is_not_internship():
    p = build_profile("Test Candidate\nWORK EXPERIENCE & INTERNSHIPS\nSoftware Engineer | Acme Corp | Jan 2024 - Dec 2024"
                      "\n- Built APIs.\nData Analyst Intern | Beta Labs | May 2023 - Jul 2023\n- Analysed data.")
    kinds = {e.company: e.is_internship for e in p.experiences}
    assert kinds == {"Acme Corp": False, "Beta Labs": True}
    assert p.internships == 1
    assert p.experience_years == pytest.approx(15 / 12, abs=0.01)


# ------------------------------------------------------------------------------------------- CGPA
@pytest.mark.parametrize("text,expected", [
    ("CGPA: 8.7/10", 8.7), ("CGPA 9.1 out of 10", 9.1), ("8.2 CGPA", 8.2), ("GPA: 3.6/4.0", 9.0),
    ("SGPA - 7.9", 7.9),
])
def test_cgpa_formats(text, expected):
    assert extract_cgpa(text)[0] == expected


@pytest.mark.parametrize("text", ["B.Tech in CSE, 2022 - 2026", "CGPA: N/A", "Percentage: 85%", "CGPA: 12/10"])
def test_cgpa_not_fabricated(text):
    assert extract_cgpa(text) == (None, None)


def test_cgpa_ambiguous_scale_keeps_raw_only():
    value, raw = extract_cgpa("GPA: 3.6")
    assert value is None and "scale not stated" in raw


def test_profile_without_cgpa():
    p = build_profile(BLP_RESUME)
    assert p.cgpa is None and p.cgpa_raw is None