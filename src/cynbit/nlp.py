"""spaCy pipeline loader with a graceful fallback to a blank English pipeline."""
from __future__ import annotations

import logging
import re
from collections import Counter
from functools import lru_cache

from .config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_nlp():
    """Return a spaCy Language object. Prefers en_core_web_sm (NER, parser, POS)."""
    import spacy

    try:
        nlp = spacy.load(settings.spacy_model)
        nlp.max_length = 2_000_000
        return nlp
    except Exception as exc:  # model not downloaded
        log.warning("spaCy model '%s' unavailable (%s); using blank 'en' + sentencizer.", settings.spacy_model, exc)
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


def has_full_model() -> bool:
    nlp = get_nlp()
    return "ner" in nlp.pipe_names


def sentences(text: str) -> list[str]:
    """Sentence / bullet segmentation. Bullets and lines are respected before spaCy splitting."""
    nlp = get_nlp()
    out: list[str] = []
    for block in re.split(r"\n+", text):
        block = block.strip().lstrip("- ").strip()
        if not block:
            continue
        if len(block) < 160:
            out.append(block)
            continue
        out.extend(s.text.strip() for s in nlp(block).sents if s.text.strip())
    return out


def entities(text: str, labels: tuple[str, ...] = ("ORG", "GPE", "DATE")) -> dict[str, list[str]]:
    """Named entities grouped by label (empty if only the blank pipeline is available)."""
    nlp = get_nlp()
    if "ner" not in nlp.pipe_names:
        return {lab: [] for lab in labels}
    doc = nlp(text[:100_000])
    grouped: dict[str, list[str]] = {lab: [] for lab in labels}
    for ent in doc.ents:
        if ent.label_ in grouped and ent.text.strip() not in grouped[ent.label_]:
            grouped[ent.label_].append(ent.text.strip())
    return grouped


def key_phrases(text: str, top_n: int = 15) -> list[str]:
    """Most frequent meaningful noun chunks (falls back to frequent non-stopword tokens)."""
    nlp = get_nlp()
    doc = nlp(text[:100_000])
    counts: Counter[str] = Counter()
    if doc.has_annotation("DEP"):
        for chunk in doc.noun_chunks:
            toks = [t for t in chunk if not t.is_stop and not t.is_punct and not t.like_num]
            if not toks:
                continue
            phrase = " ".join(t.lemma_.lower() for t in toks)
            if len(phrase) > 2:
                counts[phrase] += 1
    else:
        for t in doc:
            if t.is_alpha and not t.is_stop and len(t) > 3:
                counts[t.lower_] += 1
    return [p for p, _ in counts.most_common(top_n)]


def passive_sentences(text: str) -> list[str]:
    """Sentences using passive voice (needs dependency parser)."""
    nlp = get_nlp()
    if "parser" not in nlp.pipe_names:
        return []
    out = []
    for sent in sentences(text):
        doc = nlp(sent)
        if any(t.dep_ in ("nsubjpass", "auxpass") for t in doc):
            out.append(sent)
    return out
