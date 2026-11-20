"""Step 6: LLM regime extraction - which clauses form one contractual regime, and how.

Pipeline per (contract, regime), see ADR 0004:
1. candidates()  - no LLM: vector top-k chunks for the regime description, restricted
                   to the contract, aggregated to clauses, plus one graph hop
                   (definitions, references, subsections, parent). Data minimisation:
                   only these excerpts are ever sent to the LLM.
2. llm.complete_json() with prompts/<prompt_version>.md and ASSIGNMENT_SCHEMA.
3. validate()    - deterministic checks; anything failed or low-confidence is stored
                   with status "needs_review", never silently accepted or dropped.
4. write_regime() - (:Clause)-[:PART_OF]->(:Regime), (:Clause)-[:RELIES_ON]->(:Clause),
                   idempotent per (contract, regime, model, prompt_version).
Validation is pure (no API, no database) so it is tested with fake responses.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from neo4j import Driver
from pydantic import BaseModel, ValidationError

from clause_graph import queries
from clause_graph.chunking import collapse

ROOT = Path(__file__).resolve().parents[2]
REGIMES_FILE = Path(__file__).with_name("regimes.json")
PROMPTS = ROOT / "prompts"
CACHE = ROOT / "runs" / "llm_cache"
PROMPT_VERSION = "assign_regime_v1"

DEFAULT_K = 15  # top-k chunks per regime description
MAX_CANDIDATES = 25
CANDIDATE_CHARS = 4000  # per candidate excerpt
DEFINITION_CHARS = 600  # per defined-term excerpt in a definitions clause
MAX_DEFINITIONS = 4  # term excerpts per definitions clause
MIN_CONFIDENCE = 0.6
# Graph hop used for candidates: no siblings (too broad), no LLM edges (no feedback loop).
CANDIDATE_KINDS = frozenset(
    {"definition", "reference_out", "subsection_of_reference", "subsection", "parent",
     "reference_in"}
)
ROLES = ("rule", "cap", "exclusion", "definition", "procedure", "remedy", "other")
ELISIONS = ("...", "…")
REDACTION_RE = re.compile(r"\[\s*\*+\s*\]|\*\*\*")


# ---------------------------------------------------------------- config


@dataclass(frozen=True)
class Regime:
    name: str
    description: str
    cuad_categories: tuple[str, ...]


def load_regimes() -> list[Regime]:
    data = json.loads(REGIMES_FILE.read_text(encoding="utf-8"))
    return [
        Regime(r["name"], r["description"], tuple(r["cuad_categories"])) for r in data["regimes"]
    ]


def match_regime(name: str, regimes: list[Regime] | None = None) -> Regime:
    """Case-insensitive name or unique prefix ("liab" -> Liability)."""
    regimes = regimes or load_regimes()
    n = (name or "").strip().lower()
    exact = [r for r in regimes if r.name.lower() == n]
    if exact:
        return exact[0]
    starts = [r for r in regimes if n and r.name.lower().startswith(n)]
    if len(starts) == 1:
        return starts[0]
    valid = ", ".join(f"'{r.name}'" for r in regimes)
    raise queries.QueryError(f"Unknown regime '{name}'. Valid regimes: {valid}.")


def regime_uid(doc_id: str, regime: str) -> str:
    return f"{doc_id}#{regime}"


def load_prompt(version: str = PROMPT_VERSION) -> str:
    return (PROMPTS / f"{version}.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- 1. candidates


@dataclass
class Candidate:
    section: str
    uid: str
    text: str  # full clause text (validation runs against this)
    excerpt: str  # what is sent to the LLM
    why: list[str]
    truncated: bool


TOP_CHUNKS = """
MATCH (ch:Chunk {doc_id: $doc}) WHERE ch.embedding IS NOT NULL
WITH ch, vector.similarity.cosine(ch.embedding, $vec) AS score
ORDER BY score DESC LIMIT $k
MATCH (ch)-[:CHUNK_OF]->(c:Clause)
WITH c, ch, score ORDER BY score DESC
WITH c, collect({score: score, idx: ch.idx, start: ch.start}) AS hits
RETURN c.section AS section, c.uid AS uid, hits[0] AS best, hits
ORDER BY best.score DESC
"""
MAX_HITS = 3  # chunk excerpts per seed clause (oversized clauses, sections=0 contracts)


def _excerpt(item: dict) -> tuple[str, bool]:
    text = item["text"] or ""
    if len(text) <= CANDIDATE_CHARS:
        return text, False
    if item.get("best") is None and item["terms"]:
        # Definitions clause: one excerpt per defined term the seeds use.
        parts = []
        for term in item["terms"][:MAX_DEFINITIONS]:
            anchor = queries._definition_anchor(text, term)
            if anchor is not None:
                parts.append(queries.bounded_text(text, DEFINITION_CHARS, anchor)["text"])
        if parts:
            return "\n[...]\n".join(parts), True
    if item.get("hits"):
        # Seed: an excerpt around each matching chunk (best first), merged in text order.
        starts = sorted({h["start"] for h in item["hits"][:MAX_HITS]})
        budget = CANDIDATE_CHARS if len(starts) == 1 else CANDIDATE_CHARS * 2 // len(starts)
        spans: list[list[int]] = []
        for start in starts:
            b = queries.bounded_text(text, budget, start)
            lo = b.get("excerpt_start", 0)
            hi = lo + len(b["text"])
            if spans and lo <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], hi)
            else:
                spans.append([lo, hi])
        return "\n[...]\n".join(text[lo:hi] for lo, hi in spans), True
    return queries.bounded_text(text, CANDIDATE_CHARS)["text"], True


def candidates(
    driver: Driver,
    doc_id: str,
    vector: list[float],
    k: int = DEFAULT_K,
    max_candidates: int = MAX_CANDIDATES,
) -> tuple[list[Candidate], dict]:
    """Candidate clauses for one regime in one contract, in document order."""
    seeds = queries._read(driver, TOP_CHUNKS, doc=doc_id, vec=vector, k=k)
    kept, omitted = queries.collect_context(driver, seeds, max_candidates, CANDIDATE_KINDS)
    out = []
    for item in sorted(kept, key=queries.clause_order):
        excerpt, truncated = _excerpt(item)
        out.append(
            Candidate(item["section"], item["uid"], item["text"] or "", excerpt,
                      item["why"][:queries.MAX_REASONS], truncated)
        )
    return out, omitted


def contract_chars(driver: Driver, doc_id: str) -> int:
    rows = queries._read(
        driver,
        "MATCH (:Contract {doc_id: $doc})-[:HAS_CLAUSE]->(c:Clause) "
        "RETURN sum(size(c.text)) AS n",
        doc=doc_id,
    )
    return rows[0]["n"] or 0


# ---------------------------------------------------------------- 2. prompt + schema

ASSIGNMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "members": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "role": {"type": "string", "enum": list(ROLES)},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number", "description": "0 to 1"},
                },
                "required": ["section", "role", "evidence", "confidence"],
                "additionalProperties": False,
            },
        },
        "relies_on": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_section": {"type": "string"},
                    "to_section": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["from_section", "to_section", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["members", "relies_on"],
    "additionalProperties": False,
}


def build_messages(regime: Regime, cands: list[Candidate], prompt: str) -> tuple[str, str]:
    system = prompt.replace("{{regime_name}}", regime.name).replace(
        "{{regime_description}}", regime.description
    )
    blocks = []
    for c in cands:
        note = ' excerpt="true"' if c.truncated else ""
        blocks.append(f'<clause section="{c.section}"{note}>\n{c.excerpt}\n</clause>')
    user = (
        f"Candidate clauses ({len(cands)}), in document order:\n\n" + "\n\n".join(blocks)
        + f"\n\nAssign the members of the regime '{regime.name}'."
    )
    return system, user


def input_hash(system: str, user: str, model: str, effort: str) -> str:
    payload = json.dumps([system, user, ASSIGNMENT_SCHEMA, model, effort], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def cache_path(doc_id: str, regime: str, model: str, digest: str,
               prompt_version: str = PROMPT_VERSION) -> Path:
    safe = re.sub(r"[^\w.,&()-]+", "_", f"{doc_id}__{regime}")
    return CACHE / prompt_version / model / f"{safe}__{digest}.json"


# ---------------------------------------------------------------- 3. validation


class Member(BaseModel):
    section: str
    role: Literal["rule", "cap", "exclusion", "definition", "procedure", "remedy", "other"]
    evidence: str
    confidence: float


class Reliance(BaseModel):
    from_section: str
    to_section: str
    evidence: str


class Assignment(BaseModel):
    members: list[Member]
    relies_on: list[Reliance]


@dataclass
class Checked:
    data: dict
    checks: list[str] = field(default_factory=list)  # failed checks; empty = accepted

    @property
    def status(self) -> str:
        return "needs_review" if self.checks else "accepted"


@dataclass
class Validation:
    members: list[Checked]  # stored (accepted or needs_review)
    relies_on: list[Checked]
    rejected: list[dict]  # not storable: section does not exist in the contract, self-loop
    duplicates: int = 0
    error: str | None = None  # call-level failure: schema_error, refusal, max_tokens, ...

    def counts(self) -> dict:
        return {
            "members": len(self.members),
            "accepted": sum(m.status == "accepted" for m in self.members),
            "needs_review": sum(m.status == "needs_review" for m in self.members),
            "relies_on": len(self.relies_on),
            "relies_on_needs_review": sum(r.status == "needs_review" for r in self.relies_on),
            "rejected": len(self.rejected),
        }


def norm(text: str) -> str:
    return collapse(text)


def evidence_checks(evidence: str, clause_text: str | None) -> list[str]:
    checks = []
    e = norm(evidence)
    if not e:
        return ["evidence_empty"]
    if any(x in e for x in ELISIONS):
        checks.append("evidence_elided")
    if REDACTION_RE.search(e):
        checks.append("evidence_redacted")
    if clause_text is None or e not in norm(clause_text):
        checks.append("evidence_not_verbatim")
    return checks


def validate(
    data: dict | None,
    candidate_sections: set[str],
    clause_texts: dict[str, str],
    error: str | None = None,
) -> Validation:
    """Deterministic checks on one LLM answer.

    candidate_sections: sections that were sent; clause_texts: section -> full text for
    every clause of the contract (evidence is checked against the full clause text)."""
    if error or data is None:
        return Validation([], [], [], error=error or "no_output")
    try:
        parsed = Assignment.model_validate(data)
    except ValidationError as e:
        return Validation([], [], [], error=f"schema_error: {e.error_count()} errors")

    members: dict[tuple[str, str], Checked] = {}
    rejected: list[dict] = []
    duplicates = 0
    for m in parsed.members:
        section = queries.normalize_section(m.section)
        row = {**m.model_dump(), "section": section}
        if section not in clause_texts:
            rejected.append({**row, "reason": "section_not_in_contract"})
            continue
        checks = []
        if section not in candidate_sections:
            checks.append("section_not_candidate")
        checks += evidence_checks(m.evidence, clause_texts[section])
        if not 0 <= m.confidence <= 1:
            checks.append("confidence_out_of_range")
        elif m.confidence < MIN_CONFIDENCE:
            checks.append("low_confidence")
        key = (section, m.role)
        if key in members:
            duplicates += 1
            if m.confidence <= members[key].data["confidence"]:
                continue
        members[key] = Checked(row, checks)

    relies: list[Checked] = []
    seen: set[tuple[str, str]] = set()
    for r in parsed.relies_on:
        a, b = queries.normalize_section(r.from_section), queries.normalize_section(r.to_section)
        row = {**r.model_dump(), "from_section": a, "to_section": b}
        if a not in clause_texts or b not in clause_texts:
            rejected.append({**row, "reason": "section_not_in_contract"})
            continue
        if a == b:
            rejected.append({**row, "reason": "self_reference"})
            continue
        if (a, b) in seen:
            duplicates += 1
            continue
        seen.add((a, b))
        checks = [
            f"{side}_not_candidate"
            for side, s in (("from_section", a), ("to_section", b))
            if s not in candidate_sections
        ]
        checks += evidence_checks(r.evidence, clause_texts[a])
        relies.append(Checked(row, checks))

    return Validation(list(members.values()), relies, rejected, duplicates)


# ---------------------------------------------------------------- 4. graph

DELETE_RUN = """
MATCH (r:Regime {uid: $uid})
OPTIONAL MATCH (:Clause)-[e:PART_OF]->(r)
  WHERE e.method = 'llm' AND e.model = $model AND e.prompt_version = $pv
DELETE e
WITH DISTINCT r
OPTIONAL MATCH (:Clause {doc_id: $doc})-[x:RELIES_ON]->(:Clause)
  WHERE x.regime = $regime AND x.model = $model AND x.prompt_version = $pv
DELETE x
"""
MERGE_REGIME = """
MERGE (r:Regime {uid: $uid})
SET r.doc_id = $doc, r.name = $regime,
    r.runs = CASE WHEN $run IN coalesce(r.runs, []) THEN r.runs
                  ELSE coalesce(r.runs, []) + $run END
"""
CREATE_PART_OF = """
MATCH (r:Regime {uid: $uid})
UNWIND $rows AS row
MATCH (c:Clause {uid: row.clause})
CREATE (c)-[:PART_OF {role: row.role, evidence: row.evidence, confidence: row.confidence,
                      status: row.status, checks: row.checks, method: 'llm', model: $model,
                      prompt_version: $pv, created_at: $at}]->(r)
"""
CREATE_RELIES_ON = """
UNWIND $rows AS row
MATCH (a:Clause {uid: row.src}), (b:Clause {uid: row.dst})
CREATE (a)-[:RELIES_ON {evidence: row.evidence, status: row.status, checks: row.checks,
                        regime: $regime, method: 'llm', model: $model, prompt_version: $pv,
                        created_at: $at}]->(b)
"""


def write_regime(tx, doc_id: str, regime: str, v: Validation, model: str,
                 prompt_version: str = PROMPT_VERSION) -> None:
    """Replace this (contract, regime, model, prompt_version)'s LLM edges."""
    uid = regime_uid(doc_id, regime)
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    params = {"uid": uid, "doc": doc_id, "regime": regime, "model": model, "pv": prompt_version}
    tx.run(MERGE_REGIME, **params, run=f"{model}/{prompt_version}")
    tx.run(DELETE_RUN, **params)
    clause = lambda s: f"{doc_id}#{s}"  # noqa: E731
    tx.run(
        CREATE_PART_OF, **params, at=at,
        rows=[
            {"clause": clause(m.data["section"]), "role": m.data["role"],
             "evidence": m.data["evidence"], "confidence": m.data["confidence"],
             "status": m.status, "checks": m.checks}
            for m in v.members
        ],
    )
    tx.run(
        CREATE_RELIES_ON, **params, at=at,
        rows=[
            {"src": clause(r.data["from_section"]), "dst": clause(r.data["to_section"]),
             "evidence": r.data["evidence"], "status": r.status, "checks": r.checks}
            for r in v.relies_on
        ],
    )


def validation_dict(v: Validation) -> dict:
    return {
        "error": v.error,
        "counts": v.counts(),
        "duplicates": v.duplicates,
        "members": [{**m.data, "status": m.status, "checks": m.checks} for m in v.members],
        "relies_on": [{**r.data, "status": r.status, "checks": r.checks} for r in v.relies_on],
        "rejected": v.rejected,
    }

