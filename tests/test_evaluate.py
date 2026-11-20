"""Step 7a: gold span matching and candidate scoring (synthetic text; no database, no network)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from clause_graph import evaluate as ev
from clause_graph.regimes import Regime, load_regimes

ROOT = Path(__file__).resolve().parent.parent

# Synthetic contract: clause texts are slices of the source, as split_sections makes them.
SOURCE = (
    "THIS AGREEMENT is made between Alpha Ltd and Beta GmbH.\n\n"
    "1.  Liability\n\nNeither party shall be liable for\nindirect damages.   "
    "The total liability of Alpha shall not exceed the fees paid.\n\n"
    "2.  Assignment\n\nBeta may not assign this Agreement without consent.\n\n"
    "3.  Governing Law\n\nThis Agreement is governed by the laws of Ruritania."
)
CLAUSES = [
    ev.ClauseText("D#0", "0", "THIS AGREEMENT is made between Alpha Ltd and Beta GmbH."),
    ev.ClauseText("D#1", "1", "1.  Liability\n\nNeither party shall be liable for\nindirect "
                  "damages.   The total liability of Alpha shall not exceed the fees paid."),
    ev.ClauseText("D#2", "2", "2.  Assignment\n\nBeta may not assign this Agreement without "
                  "consent."),
    ev.ClauseText("D#3", "3", "3.  Governing Law\n\nThis Agreement is governed by the laws of "
                  "Ruritania."),
]


def labels(**spans):
    return {cat: {"present": bool(s), "evidence": list(s)} for cat, s in spans.items()}


def only(matches):
    assert len(matches) == 1
    return matches[0]


# ---------------------------------------------------------------- categories


def test_category_key_matches_gold_keys():
    assert ev.category_key("Cap On Liability") == "cap_on_liability"
    assert ev.category_key("Anti-Assignment") == "anti_assignment"
    assert ev.category_key("Change Of Control") == "change_of_control"


def test_every_regimes_json_category_maps_to_exactly_one_regime():
    regimes = load_regimes()
    mapping = ev.category_to_regime(regimes)
    listed = [ev.category_key(c) for r in regimes for c in r.cuad_categories]
    assert sorted(mapping) == sorted(set(listed)) and len(listed) == len(set(listed))
    assert mapping["governing_law"] == "Governing Law & Disputes"
    assert mapping["uncapped_liability"] == "Liability"


def test_category_in_two_regimes_is_an_error():
    regimes = [Regime("A", "", ("Exclusivity",)), Regime("B", "", ("Exclusivity",))]
    with pytest.raises(ValueError):
        ev.category_to_regime(regimes)


def test_unmapped_category_and_regime_without_gold_are_reported():
    regimes = [Regime("Law", "", ("Governing Law",)), Regime("Confidentiality", "", ())]
    gold = [{"doc_id": "D", "labels": labels(governing_law=["x"], most_favored_nation=[])}]
    report = ev.category_report(gold, ev.category_to_regime(regimes), regimes)
    assert report == {"unmapped": ["most_favored_nation"], "missing_in_gold": [],
                      "no_gold": ["Confidentiality"]}


def test_present_flag_disagreements_are_listed_without_text():
    gold = [{"doc_id": "D", "labels": {
        "a": {"present": False, "evidence": ["some span"]},
        "b": {"present": True, "evidence": []},
        "c": {"present": True, "evidence": ["ok"]},
        "d": {"present": False, "evidence": []},
    }}]
    out = ev.present_disagreements(gold)
    assert [(r["category"], r["present"], r["evidence_spans"]) for r in out] == [
        ("a", False, 1), ("b", True, 0)]
    assert all("some span" not in str(r) for r in out)


# ---------------------------------------------------------------- span matching


def test_span_inside_one_clause_maps_to_that_clause():
    m = only(ev.match_contract("D", labels(governing_law=["governed by the laws"]), CLAUSES, SOURCE))
    assert m.status == ev.CONTAINED
    assert [(g.uid, g.section) for g in m.clauses] == [("D#3", "3")]


def test_whitespace_differences_are_ignored():
    span = "Neither party shall be liable for indirect\n damages.  The total"
    m = only(ev.match_contract("D", labels(cap_on_liability=[span]), CLAUSES, SOURCE))
    assert m.status == ev.CONTAINED and [g.uid for g in m.clauses] == ["D#1"]


def test_matching_is_case_sensitive_and_not_fuzzy():
    spans = ["NEITHER PARTY shall be liable", "Neither party shall be held liable"]
    out = ev.match_contract("D", labels(cap_on_liability=spans), CLAUSES, SOURCE)
    assert [m.status for m in out] == [ev.NOT_IN_SOURCE, ev.NOT_IN_SOURCE]
    assert not any(m.matched for m in out)


def test_span_crossing_a_boundary_maps_to_every_clause_it_overlaps():
    span = "without consent.\n\n3.  Governing Law"
    m = only(ev.match_contract("D", labels(anti_assignment=[span]), CLAUSES, SOURCE))
    assert m.status == ev.OVERLAP
    assert [(g.uid, g.method, g.piece) for g in m.clauses] == [
        ("D#2", ev.OVERLAP, "without consent."), ("D#3", ev.OVERLAP, "3. Governing Law")]


def test_span_covering_three_clauses():
    span = "fees paid. 2. Assignment Beta may not assign this Agreement without consent. 3."
    m = only(ev.match_contract("D", labels(anti_assignment=[span]), CLAUSES, SOURCE))
    assert [g.uid for g in m.clauses] == ["D#1", "D#2", "D#3"]


def test_span_in_neither_clauses_nor_source_is_unmatched_not_dropped():
    out = ev.match_contract(
        "D", labels(exclusivity=["Alpha grants Beta exclusive rights"], non_compete=[" \n"]),
        CLAUSES, SOURCE)
    assert [(m.category, m.span_idx, m.status, m.matched) for m in out] == [
        ("exclusivity", 0, ev.NOT_IN_SOURCE, False), ("non_compete", 0, ev.EMPTY, False)]


def test_span_in_source_but_in_no_clause_is_unmatched():
    # The preamble clause is missing from the graph (e.g. empty): span lies in a gap.
    out = ev.match_contract("D", labels(governing_law=["made between Alpha"]), CLAUSES[1:], SOURCE)
    assert only(out).status == ev.NO_CLAUSE


def test_provenance_is_kept_per_span():
    gold = labels(governing_law=["no such text", "laws of Ruritania"])
    out = ev.match_contract("DOC-7", gold, CLAUSES, SOURCE)
    g = out[1].clauses[0]
    assert (g.doc_id, g.category, g.span_idx, g.uid, g.section) == (
        "DOC-7", "governing_law", 1, "D#3", "3")


def test_span_contained_in_two_clauses_maps_to_both():
    clauses = [ev.ClauseText("X#1", "1", "Notices go to Alpha."),
               ev.ClauseText("X#2", "2", "Invoices go to Alpha.")]
    m = only(ev.match_contract("X", labels(a=["go to Alpha"]), clauses,
                               "Notices go to Alpha. Invoices go to Alpha."))
    assert [g.uid for g in m.clauses] == ["X#1", "X#2"]


def test_locate_clauses_in_document_order_with_repeated_text():
    clauses = [ev.ClauseText("R#1", "1", "Same text."), ev.ClauseText("R#2", "2", "Same text.")]
    assert ev.locate_clauses("Same text.\n\nSame   text.", clauses) == [(0, 10), (11, 21)]


# ---------------------------------------------------------------- scoring


def gold(uid, piece, span_idx=0):
    return ev.GoldClause("D", "cap_on_liability", span_idx, uid, uid[-1], ev.CONTAINED, piece)


def test_score_seed_hop_miss_and_excerpt():
    g = [gold("D#1", "shall not exceed the fees"), gold("D#2", "may not assign"),
         gold("D#3", "laws of Ruritania"), gold("D#3", "governed by", 1)]
    sent = [
        ev.Sent("D#1", True, "The total liability of Alpha shall not\nexceed the fees paid."),
        ev.Sent("D#2", False, "2. Assignment [...] something else"),
        ev.Sent("D#0", True, "THIS AGREEMENT"),
    ]
    s = ev.score(g, sent)
    assert (s.gold_clauses, s.hits, s.hits_seed, s.hits_hop, s.hits_excerpt) == (3, 2, 1, 1, 1)
    assert s.outcome == {"D#1": ("seed", True), "D#2": ("hop", False), "D#3": ("miss", False)}
    assert s.recall == pytest.approx(2 / 3) and s.recall_excerpt == pytest.approx(1 / 3)


def test_score_without_gold_has_no_recall():
    s = ev.score([], [ev.Sent("D#1", True, "x")])
    assert s.gold_clauses == 0 and s.recall is None and s.recall_excerpt is None


def test_is_seed_reads_candidate_why():
    assert ev.is_seed(["vector match 0.61 (rank 2)", "references 9.1"])
    assert not ev.is_seed(["defines 'Losses' used in 9.1"])


def test_micro_macro_leave_out_pairs_without_gold():
    out = ev.micro_macro([(1, 1), (1, 3), (0, 0)])
    assert out == {"pairs_scored": 2, "pairs_no_gold": 1, "gold_clauses": 4, "hits": 2,
                   "micro": 0.5, "macro": round((1 + 1 / 3) / 2, 4)}


# ---------------------------------------------------------------- no LLM


def test_eval_script_imports_no_llm_module():
    """The eval must not be able to call the LLM: importing it must not load llm.py."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('ec', r'{ROOT / 'scripts' / 'eval_candidates.py'}')\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "assert 'clause_graph.llm' not in sys.modules, 'llm imported'\n"
        "assert 'anthropic' not in sys.modules, 'anthropic imported'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PATH": "", "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0, r.stderr
    assert importlib.util.find_spec("clause_graph.evaluate") is not None
