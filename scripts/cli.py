"""Command-line analysis (no UI). Useful for quick checks and CI.

    python scripts/cli.py samples/resumes/ananya_rao_ml_student.pdf samples/jobs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cynbit import analyze  # noqa: E402
from cynbit.parser import parse_file  # noqa: E402
from cynbit.ranking import ranking_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("resume", help="PDF/DOCX/TXT resume")
    ap.add_argument("jobs", nargs="+", help="JD files or a folder of .txt JDs")
    ap.add_argument("--hours", type=int, default=8)
    args = ap.parse_args()

    job_paths: list[Path] = []
    for j in args.jobs:
        p = Path(j)
        job_paths += sorted(p.glob("*.txt")) if p.is_dir() else [p]
    jobs = [parse_file(p).text for p in job_paths]
    a = analyze(parse_file(args.resume).text, jobs, hours_per_week=args.hours)
    print(f"Engine: {a.pairs[0][1].engine}\n")
    print(ranking_table(a.pairs).to_string(index=False))
    best = a.pairs[0][1]
    print(f"\nWhy #{1} – {best.job_title}:")
    for e in best.explanations:
        print("  •", e)
    print(f"\nReadiness {a.readiness.index}/100 ({a.readiness.level}) · resume quality {a.quality}/100")
    print("\nLearning plan:")
    for r in a.recommendations:
        print(f"  [{r.priority_label:6}] {r.skill:22} ~{r.est_hours:>3}h  weeks {r.week_start}-{r.week_end}")


if __name__ == "__main__":
    main()
