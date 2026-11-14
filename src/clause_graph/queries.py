"""Read-only, parameterised Neo4j queries over the contract graph.

Plain functions that take a neo4j Driver, so the MCP server, the A2A agent and
n8n can share them. Every returned item carries provenance (doc_id, section, uid).
Clause text is bounded to max_chars; any cut is reported via "truncated".
User-facing problems raise QueryError with a message meant for an LLM.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from neo4j import Driver, RoutingControl

DEFAULT_MAX_CHARS = 1500
DEFAULT_DEFINITION_CHARS = 600
MAX_LIMIT = 50
MAX_RELATED = 20  # cap on each list in get_clause_context
SUGGESTIONS = 5
HUB_NOTE = (
    "Hub terms (defined terms used in more than 40% of the contract's clauses, "
    "e.g. 'Agreement', 'Parties') have no USES_TERM edges and are not expanded."
)


class QueryError(Exception):
    """A problem the caller can fix (unknown contract, term or section)."""


# ---------------------------------------------------------------- helpers


def _read(driver: Driver, query: str, **params: Any) -> list[dict]:
    records, _, _ = driver.execute_query(query, params, routing_=RoutingControl.READ)
    return [r.data() for r in records]


def section_key(section: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in section.split("."))
    except ValueError:
        return (10**6,)


def clause_order(row: dict) -> tuple[int, ...]:
    """Document order: Clause.position when loaded, else the numeric section key
    (synthetic ALL-CAPS ids like "H3" have no numeric order)."""
    if row.get("position") is not None:
        return (row["position"],)
    return section_key(row["section"])


def provenance(doc_id: str, section: str, uid: str) -> dict:
    return {"doc_id": doc_id, "section": section, "uid": uid}


def bounded_text(text: str, max_chars: int, anchor: int | None = None) -> dict:
    """Cut text to max_chars. With an anchor (char offset of the interesting part)
    the excerpt starts at the sentence containing it, so the match stays visible."""
    if max_chars < 1:
        raise QueryError("max_chars must be a positive integer.")
    if len(text) <= max_chars:
        return {"text": text, "truncated": False}
    start = 0
    if anchor is not None and anchor > max_chars // 2:
        window = text[max(0, anchor - 300) : anchor]
        cut = max(window.rfind(". "), window.rfind("\n"), window.rfind("; "))
        start = anchor - len(window) + cut + 1 if cut >= 0 else max(0, anchor - 300)
        while start < anchor and text[start].isspace():
            start += 1
    excerpt = text[start : start + max_chars]
    out = {"text": excerpt, "truncated": True, "full_length": len(text)}
    if start:
        out["excerpt_start"] = start
    return out


def match_doc_id(prefix: str, doc_ids: list[str]) -> str:
    """Resolve a doc_id from a full id or a unique (case-insensitive) prefix."""
    p = (prefix or "").strip()
    if not p:
        raise QueryError("doc_id is empty. Call list_contracts to see available doc_ids.")
    if p in doc_ids:
        return p
    starts = [d for d in doc_ids if d.lower().startswith(p.lower())]
    if len(starts) == 1:
        return starts[0]
    if len(starts) > 1:
        shown = "\n".join(f"- {d}" for d in starts[:10])
        more = f"\n(and {len(starts) - 10} more)" if len(starts) > 10 else ""
        raise QueryError(
            f"doc_id prefix '{p}' is ambiguous: {len(starts)} contracts match. "
            f"Use a longer prefix:\n{shown}{more}"
        )
    contains = [d for d in doc_ids if p.lower() in d.lower()]
    if contains:
        shown = "\n".join(f"- {d}" for d in contains[:10])
        raise QueryError(
            f"No contract doc_id starts with '{p}'. These contain it - did you mean:\n{shown}"
        )
    shown = "\n".join(f"- {d}" for d in doc_ids[:20])
    more = " (list truncated; call list_contracts)" if len(doc_ids) > 20 else ""
    raise QueryError(f"No contract matches '{p}'. Available doc_ids{more}:\n{shown}")


def resolve_doc_id(driver: Driver, prefix: str) -> str:
    rows = _read(driver, "MATCH (k:Contract) RETURN k.doc_id AS doc_id ORDER BY doc_id")
    if not rows:
        raise QueryError("The graph is empty. Load contracts with scripts/load_graph.py.")
    return match_doc_id(prefix, [r["doc_id"] for r in rows])


def normalize_section(section: str | int) -> str:
    s = str(section).strip()
    if s.lower() in {"preamble", "0"}:
        return "0"
    s = re.sub(r"^(?:section|article|clause|§)\s*", "", s, flags=re.I)
    s = s.rstrip(".").strip()
    return s.upper() if re.fullmatch(r"h\d+", s, re.I) else s  # synthetic ALL-CAPS ids


def _definition_anchor(text: str, term: str) -> int | None:
    m = re.search(r'["“]' + re.escape(term) + r'["”]', text)
    return m.start() if m else None


# ---------------------------------------------------------------- queries


def list_contracts(driver: Driver) -> list[dict]:
    rows = _read(driver, "MATCH (k:Contract) RETURN k {.*} AS k ORDER BY k.doc_id")
    return [r["k"] for r in rows]


def count_contracts(driver: Driver) -> int:
    # Single attempt, no retry loop (unlike _read): used at server startup, which
    # must not stall for ~30s when Neo4j is down.
    with driver.session(default_access_mode="READ") as session:
        return session.run("MATCH (k:Contract) RETURN count(k) AS n").single()["n"]


def find_clauses(
    driver: Driver,
    doc_id: str,
    keyword: str,
    limit: int = 10,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict:
    doc = resolve_doc_id(driver, doc_id)
    kw = (keyword or "").strip()
    if not kw:
        raise QueryError("keyword is empty.")
    limit = max(1, min(int(limit), MAX_LIMIT))
    rows = _read(
        driver,
        """
        MATCH (c:Clause {doc_id: $doc})
        WHERE toLower(c.text) CONTAINS toLower($kw)
        RETURN c.section AS section, c.uid AS uid, c.text AS text, c.position AS position
        """,
        doc=doc,
        kw=kw,
    )
    rows.sort(key=clause_order)
    clauses = []
    for r in rows[:limit]:
        anchor = r["text"].lower().find(kw.lower())
        clauses.append(
            {**provenance(doc, r["section"], r["uid"]), **bounded_text(r["text"], max_chars, anchor)}
        )
    return {
        "doc_id": doc,
        "keyword": kw,
        "total_matches": len(rows),
        "returned": len(clauses),
        "clauses": clauses,
    }


def get_definition(
    driver: Driver, doc_id: str, term: str, max_chars: int = DEFAULT_MAX_CHARS
) -> dict:
    doc = resolve_doc_id(driver, doc_id)
    wanted = (term or "").strip().strip('"“”\'')
    if not wanted:
        raise QueryError("term is empty.")
    rows = _read(
        driver,
        """
        MATCH (t:DefinedTerm {doc_id: $doc})
        OPTIONAL MATCH (c:Clause)-[:DEFINES]->(t)
        RETURN t.name AS name, t.uid AS uid, t.hub AS hub,
               collect(c {.section, .uid, .text, .position}) AS clauses
        """,
        doc=doc,
    )
    by_name = {r["name"]: r for r in rows}
    row = by_name.get(wanted)
    note = None
    if row is None:
        ci = [n for n in by_name if n.lower() == wanted.lower()]
        if len(ci) == 1:
            row = by_name[ci[0]]
            note = f"No exact match for '{wanted}'; matched '{ci[0]}' case-insensitively."
    if row is None:
        names = list(by_name)
        lower = {n.lower(): n for n in names}
        similar = [lower[n] for n in difflib.get_close_matches(wanted.lower(), list(lower), n=SUGGESTIONS, cutoff=0.6)]
        similar += [n for n in names if wanted.lower() in n.lower() and n not in similar]
        similar = similar[:SUGGESTIONS]
        hint = f" Similar defined terms: {', '.join(similar)}." if similar else ""
        raise QueryError(
            f"'{wanted}' is not a defined term in {doc} ({len(names)} terms are defined).{hint} "
            "Terms are matched as written in quotes in the contract, e.g. \"Losses\"."
        )

    defining = sorted(row["clauses"], key=clause_order)
    definitions = [
        {
            **provenance(doc, c["section"], c["uid"]),
            **bounded_text(c["text"], max_chars, _definition_anchor(c["text"], row["name"])),
        }
        for c in defining
    ]
    sections = [c["section"] for c in defining]
    notes = [note] if note else []
    if len(sections) > 1:
        notes.append(
            f"'{row['name']}' is defined in {len(sections)} sections ({', '.join(sections)}); "
            "all are returned - check which definition governs."
        )
    return {
        "doc_id": doc,
        "term": row["name"],
        "term_uid": row["uid"],
        "hub": bool(row["hub"]),
        "defined_in_sections": sections,
        "definitions": definitions,
        "notes": notes,
    }


def _capped(items: list, label: str, notes: list[str]) -> list:
    if len(items) > MAX_RELATED:
        notes.append(f"{label}: showing {MAX_RELATED} of {len(items)}.")
        return items[:MAX_RELATED]
    return items


def get_clause_context(
    driver: Driver,
    doc_id: str,
    section: str | int,
    max_chars: int = DEFAULT_MAX_CHARS,
    definition_chars: int = DEFAULT_DEFINITION_CHARS,
) -> dict:
    doc = resolve_doc_id(driver, doc_id)
    sec = normalize_section(section)
    uid = f"{doc}#{sec}"
    rows = _read(
        driver,
        """
        MATCH (c:Clause {uid: $uid})
        OPTIONAL MATCH (c)-[:DEFINES]->(own:DefinedTerm)
        RETURN c.text AS text, coalesce(c.unresolved_refs, []) AS unresolved,
               collect(own.name) AS defines
        """,
        uid=uid,
    )
    if not rows:
        all_secs = [
            r["section"]
            for r in _read(
                driver, "MATCH (c:Clause {doc_id: $doc}) RETURN c.section AS section", doc=doc
            )
        ]
        all_secs.sort(key=section_key)
        top = sec.split(".")[0]
        near = [s for s in all_secs if s.split(".")[0] == top]
        shown = near or all_secs[:60]
        label = "Sections with the same number" if near else "Available sections"
        more = "" if near or len(all_secs) <= 60 else f" (first 60 of {len(all_secs)})"
        raise QueryError(
            f"Section '{sec}' not found in {doc}. {label}{more}: {', '.join(shown) or 'none'}. "
            "Section '0' is the preamble."
        )
    clause_row = rows[0]

    uses = _read(
        driver,
        """
        MATCH (c:Clause {uid: $uid})-[:USES_TERM]->(t:DefinedTerm)<-[:DEFINES]-(d:Clause)
        RETURN t.name AS term, t.uid AS term_uid,
               collect(d {.section, .uid, .text, .position}) AS defs
        ORDER BY toLower(t.name)
        """,
        uid=uid,
    )
    outgoing = _read(
        driver,
        """
        MATCH (c:Clause {uid: $uid})-[r:REFERENCES]->(o:Clause)
        RETURN o.section AS section, o.uid AS uid, o.text AS text, o.position AS position,
               r.number AS number
        """,
        uid=uid,
    )
    incoming = _read(
        driver,
        """
        MATCH (i:Clause)-[r:REFERENCES]->(c:Clause {uid: $uid})
        RETURN i.section AS section, i.uid AS uid, i.text AS text, i.position AS position,
               r.number AS number
        """,
        uid=uid,
    )
    # Subsections (any depth) of the clause and of every clause it cites or is cited by:
    # a reference to "Section 9.0" means the heading plus 9.1, 9.2, ...
    parents = [uid] + [r["uid"] for r in outgoing + incoming]
    sub_rows = _read(
        driver,
        """
        UNWIND $parents AS parent
        MATCH (s:Clause)-[:SUBSECTION_OF*1..3]->(:Clause {uid: parent})
        RETURN parent, s.section AS section, s.uid AS uid, s.text AS text,
               s.position AS position
        """,
        parents=parents,
    )
    subs_of: dict[str, list[dict]] = {}
    for r in sub_rows:
        subs_of.setdefault(r["parent"], []).append(r)

    notes: list[str] = [HUB_NOTE]
    definitions = []
    for u in _capped(uses, "definitions", notes):
        defs = sorted(u["defs"], key=clause_order)
        definitions.append(
            {
                "term": u["term"],
                "term_uid": u["term_uid"],
                "defined_in": [
                    {
                        **provenance(doc, d["section"], d["uid"]),
                        **bounded_text(
                            d["text"], definition_chars, _definition_anchor(d["text"], u["term"])
                        ),
                    }
                    for d in defs
                ],
            }
        )

    def subsections(parent_uid: str, label: str) -> dict:
        rows = sorted(subs_of.get(parent_uid, []), key=clause_order)
        if not rows:
            return {}
        shown = _capped(rows, f"subsections of {label}", notes)
        return {
            "subsections": [
                {**provenance(doc, r["section"], r["uid"]), **bounded_text(r["text"], max_chars)}
                for r in shown
            ]
        }

    def related(rows: list[dict], direction: str) -> list[dict]:
        rows = sorted(rows, key=clause_order)
        return [
            {
                "direction": direction,
                "number_as_written": r["number"],
                **provenance(doc, r["section"], r["uid"]),
                **bounded_text(r["text"], max_chars),
                **subsections(r["uid"], f"section {r['section']}"),
            }
            for r in rows
        ]

    references = _capped(
        related(outgoing, "outgoing"), "outgoing references", notes
    ) + _capped(related(incoming, "incoming"), "incoming references", notes)

    return {
        "clause": {
            **provenance(doc, sec, uid),
            **bounded_text(clause_row["text"], max_chars),
            "defines": sorted(clause_row["defines"]),
            **subsections(uid, f"section {sec}"),
        },
        "definitions": definitions,
        "references": references,
        "unresolved_references": clause_row["unresolved"],
        "notes": notes
        + [
            "direction 'outgoing' = this clause cites that one; "
            "'incoming' = that clause cites this one."
        ],
    }
