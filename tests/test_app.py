"""Headless UI tests using Streamlit's AppTest: every page renders without exceptions."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["🏠 Overview", "📄 Resume", "🎯 Job Match", "📊 Compare Jobs", "🚀 Readiness Dashboard",
         "✍️ Resume Improvement", "🕘 History", "🔗 Cynbit Integration"]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("CYNBIT_DB_PATH", str(tmp_path / "ui.db"))
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    return at


def test_empty_state_pages(app):
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page


def test_demo_flow_all_pages(app):
    app.sidebar.button[0].click().run()  # Load demo data
    assert not app.exception
    assert app.session_state["resume_name"] == "ananya_rao_ml_student.pdf"
    assert len(app.session_state["jobs"]) == 6
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
    app.sidebar.radio[0].set_value("🚀 Readiness Dashboard").run()
    labels = [m.label for m in app.metric]
    assert "Readiness index" in labels


def test_resume_page_shows_experience_and_internship(app):
    from test_experience import BLP_RESUME

    app.session_state["resume_text"] = BLP_RESUME
    app.session_state["resume_name"] = "blp_resume.txt"
    app.sidebar.radio[0].set_value("📄 Resume").run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Experience"] == "4 months"
    assert metrics["Internships"] == "1"
    assert metrics["CGPA (/10)"] == "Not stated"
    table = app.dataframe[0].value
    row = table.iloc[0]
    assert (row["Company"], row["Role"], row["Type"], row["Duration"]) == (
        "BLP Industry.AI", "Data Analyst", "Internship", "4 months")
    assert (row["Start"], row["End"]) == ("May 2025", "Aug 2025")