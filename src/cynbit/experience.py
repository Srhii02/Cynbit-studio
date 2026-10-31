"""Work-experience / internship extraction: date ranges, durations, company names and job titles.

Design rules
* Never fabricate: a role without a parsable date range or an explicitly stated duration contributes 0 months,
  and fields that are not present stay ``None``.
* Month-inclusive arithmetic: "May – Aug 2025" = 4 months (May, Jun, Jul, Aug).
* Overlapping roles are merged so concurrent positions are not double-counted.

Supported date formats (case-insensitive, any of - – — to till until through as separator):
    May 2025 – Aug 2025      May–August 2025 (shared year)   Nov 2024 – Feb 2025      Dec–Feb 2025 (year wrap)
    May'25 – Aug'25          May, 2025 – Aug, 2025           05/2025 – 08/2025        2025-05 – 2025-08
    01/05/2025 – 31/08/2025  May 1, 2025 – Aug 31, 2025      1st May 2025 – 31st Aug 2025
    Jan 2024 – Present/Current/Now/Till date/Ongoing          2019 – 2021 (year precision)
    Explicit durations when no range exists: "Duration: 3 months", "(8 weeks)", "6-month internship".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

_MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
    "nov": 11, "dec": 12,
}
MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
         r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_Y4 = r"(?:19|20)\d{2}"
_ORD = r"(?:st|nd|rd|th)?"

# Alternatives are tried in order at each position, so the most specific come first.
_TOKEN_RE = re.compile(
    "|".join([
        rf"(?P<iso>(?<!\d)(?P<iso_y>{_Y4})[-/.](?P<iso_m>0?[1-9]|1[0-2])(?:[-/.](?:0?[1-9]|[12]\d|3[01]))?(?!\d))",
        rf"(?P<dmy>(?<![\d.])(?P<dmy_a>\d{{1,2}})[/.-](?P<dmy_b>\d{{1,2}})[/.-](?P<dmy_y>{_Y4}|\d{{2}})(?![\d.]))",
        rf"(?P<mdy>\b(?P<mdy_m>{MONTH})\s+\d{{1,2}}{_ORD},?\s+(?P<mdy_y>{_Y4})\b)",
        rf"(?P<dmyt>\b\d{{1,2}}{_ORD}\s+(?P<dmyt_m>{MONTH}),?\s+(?P<dmyt_y>{_Y4})\b)",
        rf"(?P<my>\b(?P<my_m>{MONTH})\s*(?:['’`]\s*(?P<my_yy>\d{{2}})\b|,?\s*(?P<my_y>{_Y4})\b))",
        rf"(?P<num>(?<![\d.])(?P<num_m>0?[1-9]|1[0-2])[/.-](?P<num_y>{_Y4})(?!\d))",
        rf"(?P<mon>\b(?P<mon_m>{MONTH})(?![a-z]))",
        rf"(?P<year>(?<![\d.])(?P<year_y>{_Y4})(?![\d.]))",
        r"(?P<present>\b(?:present|current(?:ly)?|now|till\s+date|to\s+date|till\s+now|today|ongoing)\b)",
    ]),
    re.IGNORECASE,
)
_SEP_RE = re.compile(r"^\s*(?:-|–|—|−|~|→|to|till|until|through|thru)\s*$", re.IGNORECASE)
_DURATION_RE = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d)?)\s*[- ]?\s*(months?|mos?|weeks?|wks?|years?|yrs?)\b", re.I)

TITLE_WORDS = re.compile(
    r"\b(intern|internship|trainee|apprentice|analyst|engineer|developer|scientist|assistant|associate|manager|"
    r"consultant|lead|designer|researcher|specialist|architect|fellow|coordinator|executive|officer|"
    r"administrator|programmer|technician|tester|volunteer|tutor|teaching|head|director|founder|co-founder|"
    r"representative|member|ambassador|mentor|instructor|strategist|owner)\b", re.I)
INTERN_RE = re.compile(r"\b(intern|interns|internship|internships|trainee|apprentice(?:ship)?|summer\s+analyst)\b", re.I)
_LABEL_RE = re.compile(
    r"^(?P<label>company(?:\s+name)?|organi[sz]ation|employer|firm|client|role|position|title|job\s+title|"
    r"designation|duration|dates?|period|tenure|timeline|location)(?:\s*:\s*|\s+[-–—]\s+)(?P<value>.+)$", re.I)
_LOCATION_WORDS = re.compile(r"^(remote|on-?site|hybrid|wfh|work from home|india|usa|uk)$", re.I)


@dataclass
class DateRange:
    start: tuple[int, int] | None  # (year, month)
    end: tuple[int, int] | None
    raw: str
    is_present: bool = False
    year_precision: bool = False  # "2019 - 2021": only years known

    @property
    def months(self) -> int:
        if not self.start or not self.end:
            return 0
        if self.year_precision:
            return max(0, (self.end[0] - self.start[0]) * 12) or 6
        return max(0, (self.end[0] * 12 + self.end[1]) - (self.start[0] * 12 + self.start[1]) + 1)

    @property
    def start_index(self) -> int:  # absolute month index, used for merging overlaps
        return self.start[0] * 12 + self.start[1] - 1

    @property
    def end_index(self) -> int:  # exclusive
        return self.start_index + self.months


@dataclass
class ExperienceEntry:
    company: str | None = None
    title: str | None = None
    date_text: str | None = None
    date_range: DateRange | None = None
    stated_months: float | None = None  # explicit "Duration: 3 months" when no date range exists
    is_internship: bool = False
    location: str | None = None
    description: list[str] = field(default_factory=list)

    @property
    def months(self) -> float:
        if self.date_range and self.date_range.months:
            return float(self.date_range.months)
        return float(self.stated_months or 0)

    @property
    def start_label(self) -> str | None:
        r = self.date_range
        return f"{date(r.start[0], r.start[1], 1):%b %Y}" if r and r.start else None

    @property
    def end_label(self) -> str | None:
        r = self.date_range
        if not r or not r.end:
            return None
        return "Present" if r.is_present else f"{date(r.end[0], r.end[1], 1):%b %Y}"

    @property
    def duration_label(self) -> str:
        m = self.months
        if not m:
            return "Not stated"
        if m < 12:
            return f"{m:g} month" + ("s" if m != 1 else "")
        y, rem = divmod(int(round(m)), 12)
        return f"{y} yr" + (f" {rem} mo" if rem else "")


# ----------------------------------------------------------------------------------------------- dates
def _month(tok: str) -> int:
    return _MONTH_NAMES[tok.lower().rstrip(".")[:3]]


def _year2(yy: str) -> int:
    y = int(yy)
    return 2000 + y if y <= (date.today().year % 100) + 1 else 1900 + y


def _parse_token(m: re.Match) -> dict:
    """Return {'y': int|None, 'm': int|None, 'present': bool, 'kind': str}."""
    k = m.lastgroup
    g = m.groupdict()
    if k == "iso":
        return {"y": int(g["iso_y"]), "m": int(g["iso_m"]), "kind": k}
    if k == "dmy":
        a, b = int(g["dmy_a"]), int(g["dmy_b"])
        y = int(g["dmy_y"]) if len(g["dmy_y"]) == 4 else _year2(g["dmy_y"])
        # dd/mm/yyyy (Indian/European default); if the first number can't be a day-of-month-ambiguous month, use it
        month = b if b <= 12 else a
        if a > 12 and b > 12:
            return {"y": None, "m": None, "kind": "invalid"}
        return {"y": y, "m": month, "kind": k}
    if k == "mdy":
        return {"y": int(g["mdy_y"]), "m": _month(g["mdy_m"]), "kind": k}
    if k == "dmyt":
        return {"y": int(g["dmyt_y"]), "m": _month(g["dmyt_m"]), "kind": k}
    if k == "my":
        y = int(g["my_y"]) if g["my_y"] else _year2(g["my_yy"])
        return {"y": y, "m": _month(g["my_m"]), "kind": k}
    if k == "num":
        return {"y": int(g["num_y"]), "m": int(g["num_m"]), "kind": k}
    if k == "mon":
        return {"y": None, "m": _month(g["mon_m"]), "kind": k}
    if k == "year":
        return {"y": int(g["year_y"]), "m": None, "kind": k}
    if k == "present":
        return {"y": None, "m": None, "present": True, "kind": k}
    return {"y": None, "m": None, "kind": "invalid"}


def find_ranges(text: str, today: date | None = None) -> list[DateRange]:
    """Find all date ranges in a piece of text (works line by line)."""
    today = today or date.today()
    out: list[DateRange] = []
    for line in text.split("\n"):
        toks = list(_TOKEN_RE.finditer(line))
        i = 0
        while i < len(toks) - 1:
            a, b = toks[i], toks[i + 1]
            between = line[a.end():b.start()]
            sep_ok = bool(_SEP_RE.match(between)) or (
                b.lastgroup == "present" and not between.strip() and re.match(r"(to|till)\b", b.group(), re.I))
            if not sep_ok:
                i += 1
                continue
            s, e = _parse_token(a), _parse_token(b)
            r = _build_range(s, e, line[a.start():b.end()], today)
            if r:
                out.append(r)
                i += 2
            else:
                i += 1
    return out


def _build_range(s: dict, e: dict, raw: str, today: date) -> DateRange | None:
    if s.get("present") or s["kind"] == "invalid" or e["kind"] == "invalid":
        return None
    is_present = bool(e.get("present"))
    if is_present:
        e = {"y": today.year, "m": today.month, "kind": "present"}
    # Year-only on both sides -> year precision ("2019 - 2021")
    if s["kind"] == "year" and e["kind"] in {"year"}:
        if not (1970 <= s["y"] <= e["y"] <= today.year + 1):
            return None
        return DateRange((s["y"], 1), (e["y"], 1), raw.strip(), False, True)
    sy, sm, ey, em = s["y"], s["m"], e["y"], e["m"]
    if sm is None and sy is not None:   # "2024 – Mar 2025": start month unknown -> year precision only
        return DateRange((sy, 1), (ey or sy, 1), raw.strip(), is_present, True) if ey and ey >= sy else None
    if em is None and ey is not None:   # "Mar 2024 – 2025"
        return DateRange((sy or ey, 1), (ey, 1), raw.strip(), is_present, True) if sy and ey >= sy else None
    if sm is None or em is None:
        return None
    # Shared year: "May–August 2025" (start borrows end year), "May 2025 – Aug" (end borrows start year)
    if sy is None and ey is None:
        return None
    if sy is None:
        sy = ey if sm <= em else ey - 1          # "Nov–Feb 2025" -> Nov 2024
    if ey is None:
        ey = sy if em >= sm else sy + 1
    if not (1970 <= sy <= today.year + 1 and 1970 <= ey <= today.year + 2):
        return None
    if (ey, em) < (sy, sm):
        return None
    # Do not count the future: cap the end at the current month; skip roles that haven't started
    if (sy, sm) > (today.year, today.month):
        return None
    if (ey, em) > (today.year, today.month):
        ey, em = today.year, today.month
    if ey - sy > 45:
        return None
    return DateRange((sy, sm), (ey, em), raw.strip(), is_present, False)


def stated_duration_months(text: str) -> float | None:
    m = _DURATION_RE.search(text)
    if not m:
        return None
    n, unit = float(m.group(1)), m.group(2).lower()
    if unit.startswith(("week", "wk")):
        return round(n / 4.345, 1)
    if unit.startswith(("year", "yr")):
        return n * 12
    return n


def merged_months(entries: list[ExperienceEntry]) -> float:
    """Total months across entries, counting overlapping date ranges once."""
    intervals = sorted((e.date_range.start_index, e.date_range.end_index) for e in entries
                       if e.date_range and e.date_range.months)
    total, cur = 0, None
    for s, e in intervals:
        if cur and s <= cur[1]:
            cur[1] = max(cur[1], e)
        else:
            if cur:
                total += cur[1] - cur[0]
            cur = [s, e]
    if cur:
        total += cur[1] - cur[0]
    total += sum(e.stated_months or 0 for e in entries if not (e.date_range and e.date_range.months))
    return float(total)


# ----------------------------------------------------------------------------------------------- entries
def _strip_dates(line: str, ranges: list[DateRange]) -> str:
    for r in ranges:
        line = line.replace(r.raw, " ")
    line = _DURATION_RE.sub(" ", line)
    line = re.sub(r"\(\s*\)", " ", line)
    return re.sub(r"\s{2,}", " ", line).strip(" |,–—-:")


def _split_parts(line: str) -> list[str]:
    parts = re.split(r"\s+[|•·]\s+|\s*\|\s*|\s+[–—-]\s+|\s+at\s+|\s+@\s+|,\s+", line)
    return [p.strip(" |,–—-:") for p in parts if p and p.strip(" |,–—-:")]


def _is_location(part: str) -> bool:
    return bool(_LOCATION_WORDS.match(part.strip())) or bool(re.match(r"^[A-Z][a-z]+,\s*[A-Z][a-z]+$", part))


def _parse_block(lines: list[str], section_is_internship: bool, today: date) -> ExperienceEntry | None:
    entry = ExperienceEntry()
    header_lines, bullets = [], []
    for ln in lines:
        (bullets if ln.startswith("- ") else header_lines).append(ln[2:].strip() if ln.startswith("- ") else ln)

    free_header = []
    for ln in header_lines:
        m = _LABEL_RE.match(ln)
        if not m:
            free_header.append(ln)
            continue
        label, value = m.group("label").lower(), m.group("value").strip()
        if label.startswith(("company", "organ", "employer", "firm", "client")):
            entry.company = entry.company or value
        elif label in {"role", "position", "title", "job title", "designation"} or label.startswith("job"):
            entry.title = entry.title or value
        elif label == "location":
            entry.location = value
        else:  # duration / dates / period / tenure / timeline
            rs = find_ranges(value, today)
            if rs and not entry.date_range:
                entry.date_range, entry.date_text = rs[0], rs[0].raw
            elif not entry.date_range and entry.stated_months is None:
                entry.stated_months = stated_duration_months(value)
                entry.date_text = value if entry.stated_months else entry.date_text

    for ln in free_header:
        rs = find_ranges(ln, today)
        if rs and not entry.date_range:
            entry.date_range, entry.date_text = rs[0], rs[0].raw
        rest = _strip_dates(ln, rs)
        if not rest or re.fullmatch(r"(key\s+)?(internship\s+)?(responsibilities|projects?|achievements|"
                                    r"project details|highlights|work done)\s*:?", rest, re.I):
            continue
        for part in _split_parts(rest):
            if _is_location(part):
                entry.location = entry.location or part
            elif TITLE_WORDS.search(part) and not entry.title and len(part.split()) <= 7:
                entry.title = part
            elif (not entry.company and not TITLE_WORDS.search(part) and len(part.split()) <= 6
                  and not part.endswith(".") and re.search(r"\b[A-Z0-9]", part)):
                entry.company = part

    if not entry.date_range and entry.stated_months is None:
        for ln in header_lines + bullets[:2]:
            sm = stated_duration_months(ln) if re.search(r"intern|duration|month|week", ln, re.I) else None
            if sm:
                entry.stated_months, entry.date_text = sm, ln
                break
        if not entry.date_range:  # date range could be on a bullet line
            for ln in bullets:
                rs = find_ranges(ln, today)
                if rs:
                    entry.date_range, entry.date_text = rs[0], rs[0].raw
                    break

    entry.description = bullets
    header_text = " ".join(header_lines)
    entry.is_internship = bool(INTERN_RE.search(header_text) or INTERN_RE.search(entry.title or "")
                               or section_is_internship
                               or any(INTERN_RE.search(b) for b in bullets[:1]))
    if not (entry.date_range or entry.stated_months or (entry.company and entry.title)):
        return None
    return entry


def _blocks(lines: list[str]) -> list[list[str]]:
    """Group lines into one block per role. A new block starts at a non-bullet line that follows bullets, at a
    'Company:' label when the current block already has a company, or at a second dated header line."""
    blocks, cur = [], []
    cur_has_company, cur_has_dates, prev_bullet = False, False, False

    def flush():
        nonlocal cur, cur_has_company, cur_has_dates
        if cur:
            blocks.append(cur)
        cur, cur_has_company, cur_has_dates = [], False, False

    for raw in lines:
        ln = raw.strip()
        if not ln:
            continue
        is_bullet = ln.startswith("- ")
        lab = _LABEL_RE.match(ln)
        is_company_label = bool(lab and lab.group("label").lower().startswith(("company", "organ", "employer", "firm")))
        has_dates = bool(find_ranges(ln)) and not is_bullet
        if not is_bullet and (prev_bullet or (is_company_label and cur_has_company) or (has_dates and cur_has_dates)):
            flush()
        cur.append(ln)
        cur_has_company |= is_company_label
        cur_has_dates |= has_dates
        prev_bullet = is_bullet
    flush()
    return blocks


def extract_entries(section_text: str, section_is_internship: bool = False, today: date | None = None,
                    require_dates: bool = False) -> list[ExperienceEntry]:
    today = today or date.today()
    entries = []
    for block in _blocks(section_text.split("\n")):
        e = _parse_block(block, section_is_internship, today)
        if e is None:
            continue
        if require_dates and not (e.date_range and (e.title or e.is_internship)):
            continue
        entries.append(e)
    return entries