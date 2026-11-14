"""Deterministic extraction of clauses, defined terms and cross-references.

Pure functions: no LLM, no database. Everything here is driven by surface-form
regexes, so every result can be traced back to doc_id + section + source text.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

PREAMBLE = "0"
MAX_PLAIN_SECTION = 40  # plain integers above this are years, amounts, addresses

NUM = r"(\d{1,2}(?:\.\d{1,2}){0,3})"
# Headings also accept one space or tab around the dots ("5. 1.", "5 .1"); the
# number is normalised by stripping that whitespace.
HEADING_NUM = r"(\d{1,2}(?:[ \t]?\.[ \t]?\d{1,2}){0,3})"
HEADING_RE = re.compile(
    r"(?:^|(?<=[.:;]\s)|(?<=\s{2}))[ \t]*"
    r"(?:(?:Section|SECTION|Article|ARTICLE|Clause|CLAUSE)\s*-?\s*)?"
    + HEADING_NUM
    + r"\.?[ \t]+(?=[A-Z][A-Za-z])",
    re.M,
)
# Unnumbered ALL-CAPS heading on its own line start ("WRITTEN AGREEMENT", "PRODUCTS,
# QUANTITY, PRICE"), followed by a line break or a capitalised sentence. Letters
# only (no digits), at most 8 words, so all-caps disclaimers and "ARTICLE 5" do not match.
CAPS_HEADING_RE = re.compile(
    r"^[ \t]*([A-Z][A-Z&,'/\-]*(?:[ \t]+[A-Z&][A-Z&,'/\-]*){0,7})"
    r"(?=[ \t]*\n|[ \t]+[A-Z][a-z])",
    re.M,
)
CAPS_STOPWORDS = {"BETWEEN", "AMONG", "WITNESSETH", "WHEREAS", "RECITALS", "BY", "AND"}
MAX_CAPS_REPEATS = 2
CAPS_PREFIX = "H"  # synthetic section ids: H1, H2, ... in document order
DEFINITION_RE = re.compile(
    r'\(\s*(?:the\s+|each\s+(?:a|an)\s+|individually\s+(?:a|an)\s+'
    r'|collectively,?\s+(?:the\s+)?|hereinafter(?:\s+referred\s+to\s+as)?\s+(?:the\s+)?'
    r'|herein\s+)?["“]([^"”\n]{2,60}?)["”]'
    r'|["“]([^"”\n]{2,60}?)["”]\s+(?:shall\s+)?'
    r"(?:mean|means|has the meaning|shall have the meaning|refers to|includes)"
)
REFERENCE_RE = re.compile(
    r"\b(?i:sections?|articles?|clauses?|paragraphs?)\s+"
    r"(\d{1,2}(?:\.\d{1,2}){0,3}"
    r"(?:\s*(?:,|(?i:and|or|through|to))\s*\d{1,2}(?:\.\d{1,2}){0,3})*)"
)
NUM_RE = re.compile(NUM)


@dataclass(frozen=True)
class Section:
    number: str
    text: str


@dataclass
class Clause:
    uid: str
    doc_id: str
    section: str
    text: str
    position: int  # document order; synthetic ids (H1, ...) do not sort numerically


@dataclass
class ContractGraph:
    doc_id: str
    clauses: list[Clause]
    terms: list[str]  # unique per contract
    defines: list[tuple[str, str]]  # (clause uid, term name)
    uses: list[tuple[str, str]]  # (clause uid, term name)
    references: list[tuple[str, str, str]]  # (from uid, to uid, number as written)
    subsections: list[tuple[str, str]]  # (child uid, parent uid)
    unresolved: list[tuple[str, str]]  # (from uid, number as written)
    hub_terms: list[str]
    metrics: dict = field(default_factory=dict)


def clause_uid(doc_id: str, section: str) -> str:
    return f"{doc_id}#{section}"


def term_uid(doc_id: str, name: str) -> str:
    return f"{doc_id}#{name}"


def _caps_heading_starts(text: str) -> list[int]:
    """Start offsets of ALL-CAPS headings. A heading repeated more than
    MAX_CAPS_REPEATS times is a page header/footer, not a section boundary."""
    found: list[tuple[int, str]] = []
    for m in CAPS_HEADING_RE.finditer(text):
        words = m.group(1).replace(",", " ").split()
        letters = sum(c.isalpha() for c in m.group(1))
        if letters < 4 or all(w in CAPS_STOPWORDS for w in words):
            continue
        found.append((m.start(1), " ".join(words)))
    counts = Counter(key for _, key in found)
    return [pos for pos, key in found if counts[key] <= MAX_CAPS_REPEATS]


def split_sections(text: str, caps_headings: bool = True) -> list[Section]:
    """Cut text at section headings; text before the first heading is section "0".

    Only the first occurrence of a number counts (numbers repeat in tables and
    lists), and plain integers above MAX_PLAIN_SECTION are ignored. With
    caps_headings, unnumbered ALL-CAPS headings also start a section, numbered
    H1, H2, ... in document order.
    """
    starts: list[tuple[int, str]] = []
    seen: set[str] = set()
    for m in HEADING_RE.finditer(text):
        number = re.sub(r"[ \t]", "", m.group(1))
        if "." not in number and not 0 < int(number) <= MAX_PLAIN_SECTION:
            continue  # "0" is the preamble; large plain integers are years, amounts
        if number in seen:
            continue
        seen.add(number)
        starts.append((m.start(1), number))
    if caps_headings:
        taken = {pos for pos, _ in starts}
        caps = [pos for pos in _caps_heading_starts(text) if pos not in taken]
        # The first ALL-CAPS line before any numbered heading is the document
        # title ("SUPPLY AGREEMENT"); it belongs to the preamble.
        if caps and (not starts or caps[0] < starts[0][0]):
            caps = caps[1:]
        starts += [(pos, f"{CAPS_PREFIX}{i}") for i, pos in enumerate(caps, 1)]
        starts.sort()

    sections: list[Section] = []
    preamble_end = starts[0][0] if starts else len(text)
    if text[:preamble_end].strip():
        sections.append(Section(PREAMBLE, text[:preamble_end].strip()))
    for i, (start, number) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        sections.append(Section(number, text[start:end].strip()))
    return sections


def find_parent(number: str, section_numbers: set[str]) -> str | None:
    """Nearest existing ancestor: "9.1.2" -> "9.1", else "9.1.0"... up to "9" or "9.0".

    "9.0" is a heading-only form of "9", so "9.1" falls back to it, and "9.0"
    itself has parent "9" only if both exist. Synthetic ids have no parent.
    """
    parts = number.split(".")
    if not all(p.isdigit() for p in parts):
        return None
    if parts[-1] == "0":
        parts = parts[:-1]  # "9.0" starts its search at "9"
        candidates = [".".join(parts)] if parts else []
    else:
        candidates = []
    for k in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:k])
        candidates += [prefix, prefix + ".0"]
    for c in candidates:
        if c != number and c in section_numbers:
            return c
    return None


def find_definitions(text: str) -> list[str]:
    """Defined term names in order of first appearance, deduplicated."""
    names: list[str] = []
    for m in DEFINITION_RE.finditer(text):
        name = (m.group(1) or m.group(2)).strip()
        if name and name not in names:
            names.append(name)
    return names


def find_references(text: str) -> list[str]:
    """Section numbers referenced as "Section 12.3", "ARTICLE 5", "Sections 2.7, 8 and 10"."""
    numbers: list[str] = []
    for m in REFERENCE_RE.finditer(text):
        for number in NUM_RE.findall(m.group(1)):
            if number not in numbers:
                numbers.append(number)
    return numbers


def resolve_reference(number: str, section_numbers: list[str]) -> str | None:
    """Exact section, else its first subsection ("12" -> "12.1"), else None."""
    if number in section_numbers:
        return number
    prefix = number + "."
    for candidate in section_numbers:
        if candidate.startswith(prefix):
            return candidate
    return None


def _term_pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")


def extract(
    doc_id: str, text: str, hub_threshold: float = 0.4, caps_headings: bool = True
) -> ContractGraph:
    sections = split_sections(text, caps_headings=caps_headings)
    clauses = [
        Clause(clause_uid(doc_id, s.number), doc_id, s.number, s.text, i)
        for i, s in enumerate(sections)
    ]
    numbers = [c.section for c in clauses]

    # SUBSECTION_OF: 9.1 -> 9 (or 9.0 when the article exists only as a heading).
    number_set = set(numbers)
    subsections = [
        (c.uid, clause_uid(doc_id, parent))
        for c in clauses
        if c.section != PREAMBLE and (parent := find_parent(c.section, number_set))
    ]

    # DEFINES: a term may be defined in more than one clause; the node is unique.
    terms: list[str] = []
    defines: list[tuple[str, str]] = []
    defined_in: dict[str, set[str]] = {}
    for c in clauses:
        for name in find_definitions(c.text):
            defines.append((c.uid, name))
            defined_in.setdefault(name, set()).add(c.uid)
            if name not in terms:
                terms.append(name)

    # USES_TERM: whole-word, case-sensitive occurrence in a clause that does not define it.
    candidate_uses: dict[str, list[str]] = {}
    for name in terms:
        pattern = _term_pattern(name)
        candidate_uses[name] = [
            c.uid
            for c in clauses
            if c.uid not in defined_in[name] and pattern.search(c.text)
        ]
    hub_terms = [
        name
        for name, users in candidate_uses.items()
        if clauses and len(users) / len(clauses) > hub_threshold
    ]
    uses = [
        (uid, name)
        for name, users in candidate_uses.items()
        if name not in hub_terms
        for uid in users
    ]

    # REFERENCES: resolved to an existing clause, no self-references.
    references: list[tuple[str, str, str]] = []
    unresolved: list[tuple[str, str]] = []
    for c in clauses:
        targets: set[str] = set()
        for number in find_references(c.text):
            target = resolve_reference(number, numbers)
            if target is None:
                unresolved.append((c.uid, number))
                continue
            target_uid = clause_uid(doc_id, target)
            if target_uid == c.uid or target_uid in targets:
                continue
            targets.add(target_uid)
            references.append((c.uid, target_uid, number))

    total_refs = len(references) + len(unresolved)
    synthetic = sum(1 for c in clauses if c.section.startswith(CAPS_PREFIX))
    metrics = {
        # Numbered sections only, so a contract whose numbering failed stays at 0
        # even when ALL-CAPS headings split it.
        "sections": sum(1 for c in clauses if c.section != PREAMBLE) - synthetic,
        "caps_sections": synthetic,
        # Under-splitting signals: a long preamble or one huge clause means missed headings.
        "preamble_chars": sum(len(c.text) for c in clauses if c.section == PREAMBLE),
        "max_clause_chars": max((len(c.text) for c in clauses), default=0),
        "definitions": len(terms),
        "term_uses": len(uses),
        "hub_terms": len(hub_terms),
        "references": len(references),
        "subsection_links": len(subsections),
        "unresolved_refs": len(unresolved),
        "unresolved_share": round(len(unresolved) / total_refs, 3) if total_refs else 0.0,
    }
    return ContractGraph(
        doc_id=doc_id,
        clauses=clauses,
        terms=terms,
        defines=defines,
        uses=uses,
        references=references,
        subsections=subsections,
        unresolved=unresolved,
        hub_terms=hub_terms,
        metrics=metrics,
    )
