"""Taxonomy-driven skill extraction with alias normalisation and evidence sentences."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import DATA_DIR


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    aliases: tuple[str, ...]
    hours: int = 20
    related: tuple[str, ...] = ()
    case_sensitive: bool = False


@dataclass
class SkillHit:
    name: str
    category: str
    count: int = 0
    evidence: list[str] = field(default_factory=list)


class SkillTaxonomy:
    def __init__(self, skills: list[Skill]):
        self.skills = {s.name: s for s in skills}
        self._patterns: list[tuple[Skill, re.Pattern]] = []
        for s in skills:
            forms = sorted({a.lower() for a in s.aliases} | ({s.name.lower()} if not s.case_sensitive else set()),
                           key=len, reverse=True)
            if forms:
                alt = "|".join(re.escape(f) for f in forms)
                # Custom boundaries so that "C++", "C#", "Node.js", "CI/CD" match but "reactive" != "react".
                self._patterns.append((s, re.compile(rf"(?<![\w+#])(?:{alt})(?![\w+#]|\.\w)", re.IGNORECASE)))
            if s.case_sensitive:
                # Ambiguous short names ("R", "Go") only count inside list context: "Python, R, SQL".
                self._patterns.append((s, re.compile(
                    rf"(?:(?<=[,:;|/(•])|(?<=[,:;|/(•] ))({re.escape(s.name)})(?=\s?[,;|/)]|\s*$)", re.MULTILINE)))

    @classmethod
    def from_json(cls, path: Path) -> "SkillTaxonomy":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        skills = [
            Skill(
                name=d["name"], category=d["category"], aliases=tuple(d.get("aliases", [])),
                hours=int(d.get("hours", 20)), related=tuple(d.get("related", [])),
                case_sensitive=bool(d.get("case_sensitive", False)),
            )
            for d in raw["skills"]
        ]
        return cls(skills)

    @property
    def categories(self) -> list[str]:
        return sorted({s.category for s in self.skills.values()})

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def extract(self, text: str, with_evidence: bool = True) -> dict[str, SkillHit]:
        """Return canonical skills found in text, with mention counts and evidence lines."""
        hits: dict[str, SkillHit] = {}
        lines = [ln.strip("- ").strip() for ln in text.split("\n") if ln.strip()]
        for skill, pat in self._patterns:
            matches = list(pat.finditer(text))
            if not matches:
                continue
            hit = hits.setdefault(skill.name, SkillHit(skill.name, skill.category))
            hit.count += len(matches)
            if with_evidence and len(hit.evidence) < 3:
                for ln in lines:
                    if pat.search(ln) and ln not in hit.evidence:
                        hit.evidence.append(ln[:220])
                        if len(hit.evidence) >= 3:
                            break
        return dict(sorted(hits.items(), key=lambda kv: (-kv[1].count, kv[0])))

    def by_category(self, names) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for n in names:
            s = self.skills.get(n)
            if s:
                out.setdefault(s.category, []).append(n)
        return out


@lru_cache(maxsize=1)
def get_taxonomy() -> SkillTaxonomy:
    return SkillTaxonomy.from_json(DATA_DIR / "skills_taxonomy.json")


def extract_skills(text: str) -> dict[str, SkillHit]:
    return get_taxonomy().extract(text)
