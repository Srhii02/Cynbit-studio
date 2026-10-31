"""SQLite persistence for analysis history.

Privacy: raw resume text is NOT stored, only a SHA-256 fingerprint, the file name and derived scores/skills.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS resumes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT UNIQUE NOT NULL,
    filename TEXT,
    candidate_label TEXT,
    skills_json TEXT,
    experience_years REAL,
    education TEXT,
    quality_score REAL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id INTEGER NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    job_title TEXT,
    company TEXT,
    job_fingerprint TEXT,
    overall REAL, semantic REAL, skills REAL, experience REAL, education REAL,
    verdict TEXT,
    engine TEXT,
    missing_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_analyses_resume ON analyses(resume_id);
"""


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class HistoryStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("CYNBIT_DB_PATH") or settings.db_path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_resume(self, text: str, filename: str, label: str, skills: list[str], years: float,
                      education: str, quality: float) -> int:
        fp = fingerprint(text)
        with self._conn() as c:
            c.execute(
                """INSERT INTO resumes (fingerprint, filename, candidate_label, skills_json, experience_years, education, quality_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(fingerprint) DO UPDATE SET filename=excluded.filename, skills_json=excluded.skills_json,
                   experience_years=excluded.experience_years, education=excluded.education, quality_score=excluded.quality_score""",
                (fp, filename, label, json.dumps(skills), years, education, quality),
            )
            return c.execute("SELECT id FROM resumes WHERE fingerprint = ?", (fp,)).fetchone()[0]

    def add_analysis(self, resume_id: int, job_text: str, result) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO analyses (resume_id, job_title, company, job_fingerprint, overall, semantic, skills,
                   experience, education, verdict, engine, missing_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (resume_id, result.job_title, result.company, fingerprint(job_text), result.overall, result.semantic,
                 result.skills, result.experience, result.education, result.verdict, result.engine,
                 json.dumps([s.skill for s in result.missing])),
            )
            return cur.lastrowid

    def history(self, limit: int = 200) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query(
                """SELECT a.id, a.created_at, r.filename, a.job_title, a.company, a.overall, a.semantic, a.skills,
                          a.experience, a.education, a.verdict, a.engine, a.missing_json
                   FROM analyses a JOIN resumes r ON r.id = a.resume_id ORDER BY a.id DESC LIMIT ?""",
                c, params=(limit,))

    def resume_trend(self, text: str) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query(
                """SELECT a.created_at, a.job_title, a.overall FROM analyses a JOIN resumes r ON r.id = a.resume_id
                   WHERE r.fingerprint = ? ORDER BY a.id""", c, params=(fingerprint(text),))

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM analyses")
            c.execute("DELETE FROM resumes")
