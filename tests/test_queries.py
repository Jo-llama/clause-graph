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


# ---------------------------------------------------------------- semantic search (needs Ollama)


@pytest.fixture(scope="module")
def ollama():
    from clause_graph import embeddings

    try:
        embeddings.embed_query("ping")
    except embeddings.EmbeddingError as e:
        pytest.skip(f"Ollama not available: {e}")


def test_search_clauses_in_one_contract(driver, ollama):
    out = queries.search_clauses(driver, "limitation of liability", doc_id=DYNAMEX, k=5)
    sections = [c["section"] for c in out["clauses"]]
    assert "8.2" in sections[:3]  # the liability cap
    scores = [c["score"] for c in out["clauses"]]
    assert scores == sorted(scores, reverse=True) and all(-1 <= s <= 1 for s in scores)
    assert all({"doc_id", "section", "uid", "text"} <= c.keys() for c in out["clauses"])


def test_search_clauses_rejects_empty_query(driver):
    with pytest.raises(QueryError, match="empty"):
        queries.search_clauses(driver, "  ", doc_id=DYNAMEX)


def test_assemble_context_brings_the_liability_regime_together(driver, ollama):
    out = queries.assemble_context(driver, "how is liability limited?", DYNAMEX)
    by_section = {i["section"]: i for i in out["items"]}
    assert {"6.0", "8.1", "8.2", "9.1"} <= by_section.keys()
    assert any(w.startswith("vector match") for w in by_section["8.2"]["why"])
    assert "referenced by 8.2" in by_section["9.0"]["why"]
    assert any("'Freight'" in w for w in by_section["1.1"]["why"])
    assert all(i["why"] for i in out["items"])
    positions = [i["uid"] for i in out["items"]]
    assert len(positions) == len(set(positions))  # deduplicated


def test_assemble_context_is_bounded(driver, ollama):
    out = queries.assemble_context(driver, "how is liability limited?", DYNAMEX, max_items=6)
    assert out["returned"] == 6
    assert out["omitted"]["total"] > 0 and out["omitted"]["by_relation"]
    assert sum(i["role"] == "seed" for i in out["items"]) == 5  # seeds are kept first


# ------------------------------------------------------------- hop reserve (fixture graph, no Neo4j)

import fake_graph as fg  # noqa: E402

from clause_graph import regimes  # noqa: E402


def collect(graph, max_items, hop_reserve=0, kinds=regimes.CANDIDATE_KINDS, k=None):
    return queries.collect_context(graph, graph.seed_rows(k), max_items, kinds, hop_reserve)


def seeds_and_hops(kept):
    seeds = [i["uid"] for i in kept if i["tier"] == queries.TIER_SEED]
    return seeds, [i["uid"] for i in kept if i["tier"] != queries.TIER_SEED]


def test_assemble_context_unchanged_on_fixture_graph(monkeypatch):
    """Regression pin: output as produced before hop_reserve existed (default 0)."""
    g = fg.small_graph()
    monkeypatch.setattr(queries, "_embed_query", lambda q: [0.0])
    monkeypatch.setattr(queries, "resolve_doc_id", lambda d, x: fg.DOC)
    monkeypatch.setattr(queries, "_score_clauses_in_doc", lambda d, doc, v: g.seed_rows())
    out = queries.assemble_context(g, "q", "FIX", k=5, max_items=8)
    assert [(i["section"], i["role"], i["score"], i["why"]) for i in out["items"]] == [
        ("1", "seed", 0.9, ["vector match 0.9 (rank 1)"]),
        ("2", "seed", 0.88, ["vector match 0.88 (rank 2)"]),
        ("3", "seed", 0.86, ["vector match 0.86 (rank 3)"]),
        ("4", "seed", 0.84, ["vector match 0.84 (rank 4)"]),
        ("5", "seed", 0.82, ["vector match 0.82 (rank 5)"]),
        ("6", "context", 0.8, ["referenced by 1"]),
        ("20", "context", None, ["defines 'Losses' used by 1"]),
        ("21", "context", None, ["referenced by 2"]),
    ]
    assert out["returned"] == 8
    assert out["omitted"] == {"total": 3, "by_relation": {"parent": 1, "reference_in": 1,
                                                          "sibling": 1}}
    assert out["notes"][-1].startswith("3 more related clauses omitted (max_items=8)")


def test_hop_reserve_zero_pinned_on_fixture_graphs():
    """r = 0 is the plain ranked cut: seeds (a seed that is also a hop target ranks with
    its citing seed), then direct hops, structure, citing; pinned uid lists and omitted."""
    g = fg.small_graph()
    kept, omitted = collect(g, 10)
    assert [i["section"] for i in kept] == ["1", "6", "2", "3", "4", "5", "7", "8", "20", "21"]
    assert omitted == {"total": 4, "by_relation": {"reference_out": 1, "parent": 1,
                                                   "subsection": 1, "reference_in": 1}}
    kept, omitted = collect(g, 10, kinds=queries.ALL_KINDS)
    assert [i["section"] for i in kept] == ["1", "6", "2", "3", "4", "5", "7", "8", "20", "21"]
    assert omitted["total"] == 5 and omitted["by_relation"]["sibling"] == 1
    g = fg.wide_graph(12)
    kept, omitted = collect(g, 9, kinds=queries.ALL_KINDS, k=8)
    assert [i["section"] for i in kept] == [str(s) for s in range(1, 9)] + ["101"]
    assert omitted == {"total": 7, "by_relation": {"reference_out": 7}}


@pytest.mark.parametrize("max_items", [3, 6, 8, 10, 14, 20])
def test_hop_reserve_zero_is_the_plain_ranked_cut(max_items):
    g = fg.small_graph()
    full, _ = collect(g, 100)
    kept, omitted = collect(g, max_items)
    assert [i["uid"] for i in kept] == [i["uid"] for i in full[:max_items]]
    assert omitted["total"] == max(0, len(full) - max_items)
    assert collect(g, max_items, hop_reserve=0)[1] == omitted


def test_hop_reserve_keeps_reserved_hops_and_best_seeds():
    g = fg.small_graph()  # 8 seeds; 6 hop targets in CANDIDATE_KINDS (seed 6 is not one)
    full, _ = collect(g, 100)
    n, m = seeds_and_hops(full)
    assert (len(n), len(m)) == (8, 6)
    pos = {i["uid"]: p for p, i in enumerate(full)}
    for max_items in (4, 6, 10, 12):
        for r in (0, 1, 2, 3, 5, 6, 10, 30):
            kept, omitted = collect(g, max_items, hop_reserve=r)
            seeds, hops = seeds_and_hops(kept)
            assert len(kept) == max_items, (max_items, r)
            assert len(hops) == min(len(m), max(min(r, len(m), max_items),
                                                max_items - len(n))), (max_items, r)
            assert seeds == n[: len(seeds)]  # every kept seed outranks every dropped seed
            assert hops == m[: len(hops)]
            assert [pos[i["uid"]] for i in kept] == sorted(pos[i["uid"]] for i in kept)
            assert omitted["total"] == len(full) - max_items
            assert omitted["by_relation"].get("seed", 0) == len(n) - len(seeds)


def test_hop_reserve_has_no_effect_when_everything_fits():
    g = fg.small_graph()  # 14 items in CANDIDATE_KINDS: 14 is exactly the budget
    for max_items in (14, 100):
        base = collect(g, max_items)
        for r in (1, 5, 25, 10**6):
            assert collect(g, max_items, hop_reserve=r) == base, (max_items, r)


def test_omitted_counts_each_cut_item_once_per_kind():
    g = fg.wide_graph(30)
    for r in (0, 5, 10):
        kept, om = collect(g, 25, hop_reserve=r)
        assert len(kept) == 25 and om["total"] == 60 - 25
        assert om["by_relation"] == {"seed": 30 - (25 - r), "reference_out": 30 - r}


def test_seed_that_is_also_a_hop_target_uses_no_reserved_slot():
    """Seed 6 is also referenced by seed 1: it stays a seed (tier and, as before, the
    citing seed's rank), and the reserved slots all go to clauses that are not seeds."""
    g = fg.small_graph()
    kept, omitted = collect(g, 6, hop_reserve=3)
    seeds, hops = seeds_and_hops(kept)
    assert seeds == [fg.uid(s) for s in (1, 6, 2)]
    assert hops == [fg.uid(s) for s in (20, 21, 22)]  # direct tier before parent/subsection
    assert omitted["by_relation"] == {"seed": 5, "parent": 1, "reference_in": 1,
                                      "subsection": 1}
    item6 = next(i for i in collect(g, 100)[0] if i["uid"] == fg.uid(6))
    assert item6["tier"] == queries.TIER_SEED and "reference_out" in item6["kinds"]


def test_items_record_the_seeds_whose_hop_brought_them_in():
    kept, _ = collect(fg.small_graph(), 100)
    sources = {i["section"]: i["sources"] for i in kept}
    assert sources["1"] == [] and sources["2"] == []  # pure seeds
    assert sources["6"] == [fg.uid(1)]  # seed and hop target of seed 1
    assert sources["20"] == [fg.uid(1)] and sources["22"] == [fg.uid(8)]


# ---------------------------------------------------------------- reserve_hops (pure)


def mk(n_seeds, n_hops):
    return ([{"tier": queries.TIER_SEED, "n": f"s{i}"} for i in range(n_seeds)]
            + [{"tier": queries.TIER_DIRECT, "n": f"h{i}"} for i in range(n_hops)])


def names(items):
    return [i["n"] for i in items]


def test_reserve_hops_fills_free_slots_in_ranked_order():
    ranked = [{"tier": queries.TIER_SEED, "n": i} for i in range(3)] + [
        {"tier": queries.TIER_DIRECT, "n": 10 + i} for i in range(5)]
    kept, cut = queries.reserve_hops(ranked, 6, hop_reserve=2)
    assert [i["n"] for i in kept] == [0, 1, 2, 10, 11, 12]  # 3 seeds < 6 - 2: hop fills
    assert [i["n"] for i in cut] == [13, 14]
    kept, cut = queries.reserve_hops(ranked, 6, hop_reserve=9)
    assert [i["n"] for i in kept] == [0, 10, 11, 12, 13, 14]


@pytest.mark.parametrize("r", [0, 1, 5, 99])
@pytest.mark.parametrize("cap", [0, 1])
def test_reserve_hops_tiny_budgets(r, cap):
    ranked = mk(3, 3)
    kept, cut = queries.reserve_hops(ranked, cap, r)
    assert len(kept) == cap and len(kept) + len(cut) == 6
    if cap == 1:
        assert kept == [ranked[3] if r >= 1 else ranked[0]]  # best hop if reserved, else best seed


def test_reserve_hops_r_larger_than_max_items_means_hops_only():
    kept, _ = queries.reserve_hops(mk(10, 10), 4, 50)
    assert names(kept) == ["h0", "h1", "h2", "h3"]


def test_reserve_hops_fewer_hops_than_reserve_gives_the_rest_to_seeds():
    kept, cut = queries.reserve_hops(mk(10, 2), 6, 5)
    assert names(kept) == ["s0", "s1", "s2", "s3", "h0", "h1"]
    assert len(cut) == 6


def test_reserve_hops_negative_reserve_is_the_plain_cut_and_input_is_not_mutated():
    ranked = mk(5, 5)
    snap = list(ranked)
    assert queries.reserve_hops(ranked, 4, -3) == (ranked[:4], ranked[4:])
    queries.reserve_hops(ranked, 5, 2)
    assert ranked == snap
