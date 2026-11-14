"""Integration tests for queries.py against the running Neo4j.

Expect all 75 contracts loaded by: uv run scripts/load_graph.py --reset
Skipped when Neo4j is down or the graph is empty.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from clause_graph import queries
from clause_graph.queries import QueryError, bounded_text, match_doc_id, normalize_section

ROOT = Path(__file__).resolve().parent.parent
WEST = "WestPharma"
DYNAMEX = "DYNAMEX"


@pytest.fixture(scope="module")
def driver():
    neo4j = pytest.importorskip("neo4j")
    load_dotenv(ROOT / ".env")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        pytest.skip("NEO4J_PASSWORD not set")
    drv = neo4j.GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"), auth=("neo4j", password)
    )
    try:
        drv.verify_connectivity()
    except Exception as e:
        drv.close()
        pytest.skip(f"Neo4j not reachable: {e}")
    if not queries.list_contracts(drv):
        drv.close()
        pytest.skip("graph is empty; run scripts/load_graph.py --reset")
    yield drv
    drv.close()


# ------------------------------------------------------------- pure helpers


def test_match_doc_id_prefix_and_errors():
    ids = ["ACME_SUPPLY", "ACME_LICENCE", "BETA_SERVICES"]
    assert match_doc_id("beta", ids) == "BETA_SERVICES"
    assert match_doc_id("ACME_SUPPLY", ids) == "ACME_SUPPLY"
    with pytest.raises(QueryError, match="(?s)ambiguous.*ACME_LICENCE"):
        match_doc_id("acme", ids)
    with pytest.raises(QueryError, match="(?s)did you mean.*ACME_LICENCE"):
        match_doc_id("LICENCE", ids)
    with pytest.raises(QueryError, match="Available doc_ids"):
        match_doc_id("zzz", ids)


def test_bounded_text_keeps_anchor_visible():
    text = "A" * 2000 + ". The cap applies here. " + "B" * 2000
    out = bounded_text(text, 100, anchor=text.index("cap"))
    assert out["truncated"] and out["text"].startswith("The cap")
    assert bounded_text("short", 100) == {"text": "short", "truncated": False}


def test_normalize_section():
    assert normalize_section("Section 12.3.") == "12.3"
    assert normalize_section("preamble") == "0"
    assert normalize_section("h3") == "H3"
    assert normalize_section(7) == "7"


# ------------------------------------------------------------- against Neo4j


def test_list_contracts_has_metrics(driver):
    contracts = queries.list_contracts(driver)
    assert len(contracts) >= 10
    for c in contracts:
        assert {"doc_id", "sections", "definitions", "references", "unresolved_refs"} <= c.keys()
    leclanche = [c for c in contracts if c["doc_id"].startswith("LECLANCH")]
    assert leclanche and leclanche[0]["sections"] == 0  # known miss stays visible


def test_resolve_doc_id_prefix(driver):
    doc = queries.resolve_doc_id(driver, WEST)
    assert doc.startswith("WestPharmaceutical")
    with pytest.raises(QueryError):
        queries.resolve_doc_id(driver, "NoSuchContract")


def test_find_clauses_case_insensitive_with_provenance(driver):
    upper = queries.find_clauses(driver, WEST, "LIABILITY")
    lower = queries.find_clauses(driver, WEST, "liability")
    assert upper["total_matches"] == lower["total_matches"] > 0
    for c in lower["clauses"]:
        assert c["uid"] == f"{c['doc_id']}#{c['section']}"
        assert "liability" in c["text"].lower()


def test_find_clauses_limit_and_truncation(driver):
    out = queries.find_clauses(driver, WEST, "the", limit=2, max_chars=200)
    assert out["returned"] == 2 and out["total_matches"] > 2
    assert all(len(c["text"]) <= 200 for c in out["clauses"])
    assert any(c["truncated"] for c in out["clauses"])


def test_get_definition_multiple_sections(driver):
    out = queries.get_definition(driver, WEST, "Agreement")
    # Master agreement preamble + the seller's Standard Terms, which start at an
    # unnumbered ALL-CAPS heading (synthetic id H<n>).
    first, second = out["defined_in_sections"]
    assert first == "0" and second.startswith("H")
    assert "Standard Terms and Conditions" in out["definitions"][1]["text"]
    assert any("2 sections" in n for n in out["notes"])
    assert all(d["uid"].endswith("#" + d["section"]) for d in out["definitions"])


def test_get_definition_unknown_term_suggests(driver):
    with pytest.raises(QueryError, match="Similar defined terms:.*Agreement"):
        queries.get_definition(driver, WEST, "Agrement")


def test_get_clause_context_definitions(driver):
    ctx = queries.get_clause_context(driver, WEST, "13")
    assert ctx["clause"]["section"] == "13"
    terms = {d["term"] for d in ctx["definitions"]}
    assert "Product" in terms  # "Agreement" is a hub term in West and is not expanded
    for d in ctx["definitions"]:
        assert all({"doc_id", "section", "uid", "text"} <= x.keys() for x in d["defined_in"])


def test_get_clause_context_references_both_directions(driver):
    records, _, _ = driver.execute_query(
        "MATCH (a:Clause)-[:REFERENCES]->(b:Clause) RETURN a.doc_id AS doc, "
        "a.section AS src, b.section AS dst LIMIT 1"
    )
    doc, src, dst = records[0]["doc"], records[0]["src"], records[0]["dst"]
    out_ctx = queries.get_clause_context(driver, doc, src)
    assert ("outgoing", dst) in {(r["direction"], r["section"]) for r in out_ctx["references"]}
    in_ctx = queries.get_clause_context(driver, doc, dst)
    assert ("incoming", src) in {(r["direction"], r["section"]) for r in in_ctx["references"]}


def test_get_clause_context_unknown_section(driver):
    with pytest.raises(QueryError, match="not found"):
        queries.get_clause_context(driver, WEST, "99.9")


def test_get_clause_context_expands_subsections_of_referenced_clause(driver):
    # Regression: 8.2 caps liability at "the insurance coverage set out in Section 9.0";
    # 9.0 is only the heading, the amounts are in its subsection 9.1.
    ctx = queries.get_clause_context(driver, DYNAMEX, "8.2")
    ref = next(r for r in ctx["references"] if r["section"] == "9.0")
    subs = {s["section"]: s["text"] for s in ref["subsections"]}
    assert "9.1" in subs
    assert "$100,000.00" in subs["9.1"] and "$2,000,000.00" in subs["9.1"]
    assert list(subs) == sorted(subs, key=lambda x: tuple(map(int, x.split("."))))


def test_get_clause_context_lists_own_subsections(driver):
    ctx = queries.get_clause_context(driver, DYNAMEX, "9.0")
    assert [s["section"] for s in ctx["clause"]["subsections"]][:2] == ["9.1", "9.2"]
