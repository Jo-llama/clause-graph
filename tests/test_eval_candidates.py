"""Step 7a: eval_candidates.py and get_contracts.py end to end with a fake Neo4j driver.

Offline: no database, no Ollama, no LLM. Synthetic contract text only.
NOT covered here (needs the live graph): real Cypher semantics of TOP_CHUNKS / expansions,
real bge-m3 vectors, the full 75-contract corpus numbers.
"""

import csv
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clause_graph import evaluate as ev
from clause_graph import queries, regimes

ROOT = Path(__file__).resolve().parent.parent
# Config name of a default run (no --k / --hop-reserve / --max-candidates).
DEFAULT_CFG = regimes.candidate_config_name(
    regimes.DEFAULT_K, regimes.HOP_RESERVE, regimes.MAX_CANDIDATES)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- fake graph

FILLER = "Lorem ipsum dolor sit amet. " * 200  # ~5600 chars, longer than CANDIDATE_CHARS
GOLD_TAIL = "Beta may not assign this Agreement without consent."


def clause_texts(doc):
    return [
        ("0", f"THIS AGREEMENT is made by Alpha and Beta ({doc})."),
        ("1", "1.  Liability\n\nThe total liability of Alpha shall not exceed the fees paid."),
        ("2", "2.  Assignment\n\n" + FILLER + GOLD_TAIL),  # gold span beyond the excerpt cut
        ("3", "3.  Governing Law\n\nThis Agreement is governed by the laws of Ruritania."),
    ]


def source(doc):
    return "\n\n".join(t for _, t in clause_texts(doc))


def gold_row(doc):
    return {"doc_id": doc, "labels": {
        "cap_on_liability": {"present": True, "evidence": ["shall not exceed the fees paid"]},
        "anti_assignment": {"present": True, "evidence": [GOLD_TAIL]},
        "governing_law": {"present": True, "evidence": ["laws of Ruritania"]},
        "most_favored_nation": {"present": False, "evidence": []},
        "exclusivity": {"present": True, "evidence": ["never appears anywhere"]},
    }}


class Rec:
    def __init__(self, d):
        self.d = d

    def data(self):
        return self.d


class FakeDriver:
    """Answers the queries the script and regimes.candidates() issue. Seeds: clauses 1 and 3;
    one graph hop: clause 3 references clause 2."""

    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute_query(self, query, params=None, routing_=None):
        self.queries.append(query)
        p = params or {}
        doc = p.get("doc", "")
        texts = {f"{doc}#{s}": (s, t) for s, t in clause_texts(doc)}
        if "k.doc_id AS d" in query:
            rows = [{"d": d} for d in self.docs]
        elif query == regimes.TOP_CHUNKS:
            hit = {"score": 0.9, "idx": 0, "start": 0}
            rows = [{"section": "1", "uid": f"{doc}#1", "best": hit, "hits": [hit]},
                    {"section": "3", "uid": f"{doc}#3", "best": hit, "hits": [hit]}]
        elif query == queries_clauses_cypher():
            rows = [{"uid": f"{doc}#{s}", "section": s, "text": t, "position": i}
                    for i, (s, t) in enumerate(clause_texts(doc))]
        elif "c.uid IN $uids" in query:
            rows = []
            for uid in p["uids"]:
                d = uid.split("#")[0]
                s, t = dict(clause_texts(d))[uid.split("#")[1]] and (
                    uid.split("#")[1], dict(clause_texts(d))[uid.split("#")[1]])
                rows.append({"uid": uid, "section": s, "text": t, "position": int(s)})
        elif "(:Clause {uid: seed})-[:REFERENCES]->(x:Clause)" in query:
            d = p["seeds"][0].split("#")[0]
            rows = [{"seed": f"{d}#3", "uid": f"{d}#2", "term": None, "via": None}
                    ] if f"{d}#3" in p["seeds"] else []
        else:
            rows = []
        return [Rec(r) for r in rows], None, None


_script = {}


def queries_clauses_cypher():
    return _script["ec"].CLAUSES


@pytest.fixture
def env(tmp_path, monkeypatch):
    ec = load_script("eval_candidates")
    _script["ec"] = ec
    docs = ["DOC-A", "DOC-B", "DOC-C"]
    (tmp_path / "runs").mkdir()
    contracts = tmp_path / "contracts.jsonl"
    contracts.write_text("".join(json.dumps({"doc_id": d, "text": source(d)}) + "\n" for d in docs))
    gold = tmp_path / "gold.jsonl"
    gold.write_text("".join(json.dumps(gold_row(d)) + "\n" for d in docs))
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("doc_id,sections,caps_sections\nDOC-A,0,0\nDOC-B,12,15\nDOC-C,12,2\n")
    driver = FakeDriver(docs)
    monkeypatch.setattr(ec, "CONTRACTS", contracts)
    monkeypatch.setattr(ec, "GOLD", gold)
    monkeypatch.setattr(ec, "EXTRACTION_METRICS", metrics)
    monkeypatch.setattr(ec, "OUT_CSV", tmp_path / "runs" / "e.csv")
    monkeypatch.setattr(ec, "OUT_SUMMARY", tmp_path / "runs" / "s.json")
    monkeypatch.setattr(ec, "OUT_GOLD", tmp_path / "runs" / "g.csv")
    monkeypatch.setattr(ec, "SWEEP_CSV", tmp_path / "runs" / "sweep_rows.csv")
    monkeypatch.setattr(ec, "SWEEP_GOLD", tmp_path / "runs" / "sweep_gold.csv")
    monkeypatch.setattr(ec, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(ec, "embed_query", lambda text: [0.0, 1.0])
    monkeypatch.setattr(ec.GraphDatabase, "driver", staticmethod(lambda *a, **k: driver))
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-be-removed")
    return SimpleNamespace(ec=ec, driver=driver, tmp=tmp_path, monkeypatch=monkeypatch)


def run(env, *args):
    env.monkeypatch.setattr(sys, "argv", ["eval_candidates.py", *args])
    return env.ec.main()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def outputs(env):
    t = env.tmp / "runs"
    return [(t / n).read_bytes() for n in ("e.csv", "s.json", "g.csv")]


# ---------------------------------------------------------------- criterion 8/9/10/11/12


def test_rows_record_counts_and_seed_hop_split(env):
    assert run(env) == 0
    rows = read_csv(env.tmp / "runs" / "e.csv")
    liab = next(r for r in rows if r["doc_id"] == "DOC-B" and r["regime"] == "Liability")
    assert (liab["gold_spans"], liab["gold_clauses"], liab["hits"]) == ("1", "1", "1")
    assert (liab["hits_seed"], liab["hits_hop"], liab["candidate_clauses"]) == ("1", "0", "3")
    assert liab["recall"] == "1.0"
    assign = next(r for r in rows if r["doc_id"] == "DOC-B" and r["regime"] == "Assignment & Control")
    assert (assign["hits"], assign["hits_hop"], assign["hits_seed"]) == ("1", "1", "0")


def test_clause_hit_but_excerpt_miss_when_gold_is_beyond_the_cut(env):
    run(env)
    rows = read_csv(env.tmp / "runs" / "e.csv")
    assign = next(r for r in rows if r["doc_id"] == "DOC-B" and r["regime"] == "Assignment & Control")
    assert len(clause_texts("DOC-B")[2][1]) > regimes.CANDIDATE_CHARS
    assert (assign["recall"], assign["hits_excerpt"], assign["recall_excerpt"]) == ("1.0", "0", "0.0")
    gov = next(r for r in rows if r["doc_id"] == "DOC-B" and r["regime"] == "Governing Law & Disputes")
    assert (gov["hits_excerpt"], gov["recall_excerpt"]) == ("1", "1.0")


def test_pairs_without_gold_are_left_out_of_recall_and_counted(env):
    run(env)
    rows = read_csv(env.tmp / "runs" / "e.csv")
    term = next(r for r in rows if r["doc_id"] == "DOC-B" and r["regime"] == "Termination")
    assert term["gold_clauses"] == "0" and term["recall"] == ""
    summary = json.loads((env.tmp / "runs" / "s.json").read_text())
    by_regime = summary["by_config"][DEFAULT_CFG]["by_regime"]
    assert by_regime["Termination"]["pairs_scored"] == 0
    assert by_regime["Termination"]["pairs_no_gold"] == 3
    assert by_regime["Termination"]["micro"] is None and by_regime["Termination"]["macro"] is None
    assert by_regime["Liability"]["micro"] == 1.0 and by_regime["Liability"]["pairs_scored"] == 3
    assert by_regime["Assignment & Control"]["micro_excerpt"] == 0.0
    assert by_regime["Assignment & Control"]["micro"] == 1.0


def test_confidentiality_has_no_gold_and_is_not_scored(env):
    run(env)
    rows = read_csv(env.tmp / "runs" / "e.csv")
    assert "Confidentiality" not in {r["regime"] for r in rows}
    summary = json.loads((env.tmp / "runs" / "s.json").read_text())
    assert "Confidentiality" in summary["categories"]["no_gold"]


def test_two_runs_are_byte_identical(env):
    run(env)
    first = outputs(env)
    for n in ("e.csv", "s.json", "g.csv"):
        (env.tmp / "runs" / n).unlink()
    run(env)
    assert outputs(env) == first


def test_k_sweep_gives_one_row_set_per_k(env):
    run(env, "--k", "15", "--k", "25", "--k", "40")
    rows = read_csv(env.tmp / "runs" / "e.csv")
    n_regimes = len({r["regime"] for r in rows})
    for k in ("15", "25", "40"):
        assert len([r for r in rows if r["k"] == k]) == 3 * n_regimes
    summary = json.loads((env.tmp / "runs" / "s.json").read_text())
    assert sorted(summary["by_config"]) == ["k15", "k25", "k40"]
    assert summary["config"]["k"] == [15, 25, 40]
    gold_rows = read_csv(env.tmp / "runs" / "g.csv")
    assert {"k15", "k25", "k40", "k15_excerpt"} <= set(gold_rows[0])


def test_no_write_queries_and_no_llm_key(env):
    run(env)
    bad = ("CREATE", "MERGE", " SET ", "DELETE", "REMOVE", "DROP")
    assert env.driver.queries
    assert not [q for q in env.driver.queries if any(w in q.upper() for w in bad)]


def test_script_removes_anthropic_key_from_env(env):
    import os
    run(env)
    assert "ANTHROPIC_API_KEY" not in os.environ


# ---------------------------------------------------------------- criterion 6/7


def test_gold_clauses_csv_keeps_provenance(env):
    run(env)
    g = read_csv(env.tmp / "runs" / "g.csv")
    row = next(r for r in g if r["doc_id"] == "DOC-B" and r["category"] == "anti_assignment")
    assert (row["uid"], row["section"], row["span_idx"], row["regime"]) == (
        "DOC-B#2", "2", "0", "Assignment & Control")
    assert row[DEFAULT_CFG] == "hop"
    assert row["match"] == ev.CONTAINED


def test_summary_lists_unmatched_without_text_and_match_rate(env, capsys):
    run(env)
    summary = json.loads((env.tmp / "runs" / "s.json").read_text())
    sp = summary["spans"]
    assert sp["total"] == 12 and sp["matched"] == 9 and sp["match_rate"] == 0.75
    assert sp["unmatched"][0] == {"doc_id": "DOC-A", "category": "exclusivity", "span_idx": 0,
                                  "reason": ev.NOT_IN_SOURCE}
    dumped = json.dumps(summary) + capsys.readouterr().out
    for secret in ("never appears anywhere", "laws of Ruritania", GOLD_TAIL):
        assert secret not in dumped
    # present=False with no evidence is consistent; no disagreements in the synthetic gold
    assert summary["present_disagreements"] == []


def test_unmapped_category_is_reported_not_ignored(env):
    run(env)
    cats = json.loads((env.tmp / "runs" / "s.json").read_text())["categories"]
    assert "most_favored_nation" in cats["unmapped"]


# ---------------------------------------------------------------- criterion 13: health


def test_health_groups(env):
    h = env.ec.health
    assert h(0, 50) == "sections_0"
    assert h(12, env.ec.HIGH_CAPS) == "high_caps"
    assert h(12, env.ec.HIGH_CAPS - 1) == "ok"
    assert h(12, None) == "ok"
    assert h(None, None) == "unknown"


def test_misses_grouped_by_health(env):
    run(env)
    s = json.loads((env.tmp / "runs" / "s.json").read_text())
    groups = s["by_config"][DEFAULT_CFG]["by_health"]
    assert set(groups) == {"sections_0", "high_caps", "ok"}
    assert all(g["gold_clauses"] > 0 for g in groups.values())
    rows = read_csv(env.tmp / "runs" / "e.csv")
    assert {r["doc_id"]: r["health"] for r in rows} == {
        "DOC-A": "sections_0", "DOC-B": "high_caps", "DOC-C": "ok"}


# ---------------------------------------------------------------- seed coupling


class SeedDriver:
    def __init__(self):
        self.d = FakeDriver(["Z"])

    def execute_query(self, *a, **k):
        return self.d.execute_query(*a, **k)


def test_is_seed_recognises_what_collect_context_really_writes(env):
    """Pins the coupling: seeds' Candidate.why must start with ev.SEED_PREFIX, graph hops must not."""
    cands, _ = regimes.candidates(SeedDriver(), "Z", [0.0], 15, 25)
    by = {c.section: c for c in cands}
    assert set(by) == {"1", "2", "3"}
    assert ev.is_seed(by["1"].why) and ev.is_seed(by["3"].why)
    assert not ev.is_seed(by["2"].why), by["2"].why


def test_seed_that_is_also_a_hop_target_counts_as_seed():
    assert ev.is_seed(["referenced by 3", "vector match 0.5 (rank 1)"])


# ---------------------------------------------------------------- score edge cases


def gc(uid, piece, idx=0):
    return ev.GoldClause("D", "c", idx, uid, uid[-1], ev.CONTAINED, piece)


def test_excerpt_hit_ignores_whitespace_but_not_case():
    sent = [ev.Sent("D#1", True, "must  pay\nthe fee"), ev.Sent("D#2", True, "MUST PAY the fee")]
    s = ev.score([gc("D#1", "must pay the fee"), gc("D#2", "must pay the fee")], sent)
    assert s.outcome == {"D#1": ("seed", True), "D#2": ("seed", False)}


def test_score_counts_a_clause_once_when_several_spans_hit_it():
    s = ev.score([gc("D#1", "a", 0), gc("D#1", "b", 1)], [ev.Sent("D#1", False, "a")])
    assert (s.gold_clauses, s.hits, s.hits_excerpt) == (1, 1, 1)


def test_micro_vs_macro_differ_for_unequal_pairs():
    out = ev.micro_macro([(1, 1), (0, 9)])
    assert out["micro"] == 0.1 and out["macro"] == 0.5


def test_micro_macro_all_pairs_empty():
    out = ev.micro_macro([(0, 0)])
    assert out["micro"] is None and out["macro"] is None and out["pairs_no_gold"] == 1


def test_span_occurring_twice_in_source_maps_to_both_clauses_it_spans():
    clauses = [ev.ClauseText("X#1", "1", "Pay on time."), ev.ClauseText("X#2", "2", "Pay on time.")]
    # Spans "time. Pay" straddles the boundary once.
    m = ev.match_contract("X", {"a": {"evidence": ["time. Pay"]}}, clauses, "Pay on time.\n\nPay on time.")
    assert [g.uid for g in m[0].clauses] == ["X#1", "X#2"]


def test_labels_without_evidence_key_or_none_do_not_crash():
    out = ev.match_contract("D", {"a": {"present": False}, "b": {"evidence": None}}, [], "")
    assert out == []


def test_empty_source_and_no_clauses():
    out = ev.match_contract("D", {"a": {"evidence": ["x"]}}, [], "")
    assert [m.status for m in out] == [ev.NOT_IN_SOURCE]


@pytest.mark.parametrize("ws", ["\t", "\xa0", "\t\xa0", " \xa0\n "])
def test_nbsp_and_tab_whitespace_collapse(ws):
    text = f"Alpha{ws}shall pay."
    clauses = [ev.ClauseText("X#1", "1", text)]
    m = ev.match_contract("X", {"a": {"evidence": ["Alpha shall pay."]}}, clauses, text)
    assert m[0].status == ev.CONTAINED
    # and the other way round: the whitespace variant in the gold span
    m = ev.match_contract("X", {"a": {"evidence": [text]}},
                          [ev.ClauseText("X#1", "1", "Alpha shall pay.")], "Alpha shall pay.")
    assert m[0].status == ev.CONTAINED


# ---------------------------------------------------------------- M1: per-contract macro


def summary_row(doc, regime, hits, gold, health="ok"):
    return {"doc_id": doc, "regime": regime, "health": health, "hits": hits,
            "gold_clauses": gold, "hits_seed": hits, "hits_hop": 0, "hits_excerpt": hits,
            "spans_matched": gold, "span_hits": hits, "candidate_clauses": 1,
            "candidate_chars": 1, "contract_chars": 2, "hops_kept": 0, "hops_from_cut_seeds": 0}


def test_per_contract_sums_each_doc_first():
    assert ev.per_contract([("A", 1, 1), ("A", 0, 9), ("B", 1, 1), ("C", 0, 0)]) == [
        (1, 10), (1, 1), (0, 0)]


def test_all_and_health_macro_is_a_mean_over_contracts_not_pairs(env):
    rows = [summary_row("A", "R1", 1, 1), summary_row("A", "R2", 0, 9),
            summary_row("B", "R1", 1, 1), summary_row("B", "R2", 0, 0)]
    out = env.ec.summarise([{**r, "all": "all"} for r in rows], "all", per_contract=True)["all"]
    assert out["micro"] == round(2 / 11, 4)
    assert out["macro"] == round((1 / 10 + 1 / 1) / 2, 4)  # per contract: A 0.1, B 1.0
    assert out["macro_excerpt"] == out["macro"]
    assert (out["contracts_scored"], out["pairs_scored"], out["pairs_no_gold"]) == (2, 3, 1)
    assert out["macro"] != round(2 / 3, 4)  # the old mean over (contract, regime) pairs
    health = env.ec.summarise(rows, "health", per_contract=True)["ok"]
    assert health["macro"] == out["macro"]


def test_by_regime_macro_is_unchanged_per_contract(env):
    rows = [summary_row("A", "R1", 1, 1), summary_row("B", "R1", 0, 9)]
    out = env.ec.summarise(rows, "regime")["R1"]
    assert (out["micro"], out["macro"]) == (0.1, 0.5)
    assert "contracts_scored" not in out


def test_summary_all_macro_matches_per_contract_definition(env):
    run(env)
    s = json.loads((env.tmp / "runs" / "s.json").read_text())
    rows = [r for r in read_csv(env.tmp / "runs" / "e.csv") if r["config"] == DEFAULT_CFG]
    per_doc = {}
    for r in rows:
        acc = per_doc.setdefault(r["doc_id"], [0, 0])
        acc[0] += int(r["hits"])
        acc[1] += int(r["gold_clauses"])
    expect = round(sum(h / g for h, g in per_doc.values() if g) / len(per_doc), 4)
    assert s["by_config"][DEFAULT_CFG]["all"]["macro"] == expect


# ---------------------------------------------------------------- M2: spans in several clauses


def test_span_in_two_clauses_is_two_gold_clauses_but_one_span():
    clauses = [ev.ClauseText("X#1", "1", "Pay on time."), ev.ClauseText("X#9", "9", "Pay on time.")]
    m = ev.match_contract("X", {"a": {"evidence": ["Pay on time."]}}, clauses,
                          "Pay on time.\n\nPay on time.")
    assert m[0].status == ev.CONTAINED and len(m[0].clauses) == 2
    s = ev.score(list(m[0].clauses), [ev.Sent("X#1", True, "Pay on time.")])
    assert (s.gold_clauses, s.hits) == (2, 1)  # clause level: half
    assert ev.span_hits(m, {"X#1"}) == (1, 1)  # span level: hit
    assert ev.span_hits(m, set()) == (1, 0)


def test_span_hits_ignore_unmatched_spans():
    m = ev.match_contract("X", {"a": {"evidence": ["nowhere"]}}, [], "")
    assert ev.span_hits(m, {"X#1"}) == (0, 0)


def test_summary_has_span_level_recall_and_multi_clause(env):
    run(env)
    s = json.loads((env.tmp / "runs" / "s.json").read_text())
    assert s["spans"]["multi_clause"] == 0
    k = s["by_config"][DEFAULT_CFG]
    for part in (k["all"], *k["by_regime"].values(), *k["by_health"].values(),
                 *k["by_category"].values()):
        assert {"spans_matched", "span_hits", "span_recall"} <= set(part)
    # 9 matched spans, all in a candidate clause (assignment via the graph hop)
    assert (k["all"]["spans_matched"], k["all"]["span_hits"], k["all"]["span_recall"]) == (9, 9, 1.0)
    rows = read_csv(env.tmp / "runs" / "e.csv")
    assign = next(r for r in rows if r["doc_id"] == "DOC-B" and r["regime"] == "Assignment & Control")
    assert (assign["spans_matched"], assign["span_hits"], assign["span_recall"]) == ("1", "1", "1.0")


def test_multi_clause_counted_in_summary(env, monkeypatch):
    original = clause_texts

    def with_copy(doc):
        return original(doc) + [("4", "4.  Assignment again\n\n" + GOLD_TAIL)]
    monkeypatch.setitem(globals(), "clause_texts", with_copy)
    env.ec.CONTRACTS.write_text("".join(
        json.dumps({"doc_id": d, "text": "\n\n".join(t for _, t in with_copy(d))}) + "\n"
        for d in env.driver.docs))
    run(env)
    s = json.loads((env.tmp / "runs" / "s.json").read_text())
    assert s["spans"]["multi_clause"] == 3  # the anti_assignment span of each contract
    a = s["by_config"][DEFAULT_CFG]["by_regime"]["Assignment & Control"]
    assert (a["gold_clauses"], a["spans_matched"], a["span_hits"]) == (6, 3, 3)
    assert a["hits"] == 3  # clause 4 is neither seed nor hop: clause level 0.5, span level 1.0


# ---------------------------------------------------------------- M3: coverage of the gold


def test_gold_contracts_not_evaluated_are_listed(env):
    env.driver.docs = ["DOC-A", "DOC-B"]  # DOC-C has gold but is not loaded
    run(env)
    s = json.loads((env.tmp / "runs" / "s.json").read_text())
    assert s["gold_contracts_not_evaluated"] == ["DOC-C"]
    assert s["config"]["contracts"] == 2


def test_all_gold_evaluated_gives_empty_list(env):
    run(env)
    s = json.loads((env.tmp / "runs" / "s.json").read_text())
    assert s["gold_contracts_not_evaluated"] == []


def test_loaded_doc_missing_from_contracts_jsonl_gets_no_source(env):
    env.ec.CONTRACTS.write_text("".join(
        json.dumps({"doc_id": d, "text": source(d)}) + "\n" for d in ("DOC-A", "DOC-B")))
    run(env)
    s = json.loads((env.tmp / "runs" / "s.json").read_text())
    assert s["config"]["contracts"] == 3  # DOC-C is loaded, so it is still evaluated
    by = s["spans"]["by_status"]
    assert by[ev.NO_SOURCE] == 1 and by[ev.NOT_IN_SOURCE] == 2 and by[ev.CONTAINED] == 9
    assert {"doc_id": "DOC-C", "category": "exclusivity", "span_idx": 0,
            "reason": ev.NO_SOURCE} in s["spans"]["unmatched"]


def test_match_contract_without_source():
    clauses = [ev.ClauseText("X#1", "1", "Alpha shall pay.")]
    m = ev.match_contract("X", {"a": {"evidence": ["shall pay", "elsewhere"]}}, clauses, None)
    assert [x.status for x in m] == [ev.CONTAINED, ev.NO_SOURCE]


# ---------------------------------------------------------------- get_contracts.py (criterion 1, 2)


CUAD_CATS = ["cap_on_liability", "governing_law", "anti_assignment"]


def fake_golden(n):
    lines = []
    for i in range(n):
        labels = {c: {"present": i % 2 == 0, "evidence": [f"synthetic span {i}"] if i % 2 == 0 else []}
                  for c in CUAD_CATS}
        lines.append(json.dumps({"doc_id": f"d{i}", "source_text": f"text {i}", "labels": labels}))
    lines.append(lines[0])  # duplicate doc_id must be dropped
    lines.insert(1, "")
    return ("\n".join(lines) + "\n").encode()


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def gc_mod(tmp_path, monkeypatch):
    m = load_script("get_contracts")
    monkeypatch.setattr(m, "DATA", tmp_path)
    monkeypatch.setattr(m, "OUT", tmp_path / "contracts.jsonl")
    monkeypatch.setattr(m, "GOLD", tmp_path / "gold.jsonl")
    monkeypatch.setattr(m.urllib.request, "urlopen", lambda *a, **k: Resp(fake_golden(75)))
    return m


def test_get_contracts_writes_75_gold_rows_with_same_categories(gc_mod):
    assert gc_mod.main() == 0
    gold = [json.loads(line) for line in gc_mod.GOLD.read_text().splitlines()]
    assert len(gold) == 75
    assert all(set(r) == {"doc_id", "labels"} for r in gold)
    assert all(sorted(r["labels"]) == sorted(CUAD_CATS) for r in gold)
    src = [json.loads(line) for line in gc_mod.OUT.read_text().splitlines()]
    assert [r["doc_id"] for r in src] == [r["doc_id"] for r in gold]
    assert set(src[0]) == {"doc_id", "text"}


def test_get_contracts_leaves_contracts_byte_identical_and_untouched(gc_mod):
    gc_mod.main()
    before = gc_mod.OUT.read_bytes()
    mtime = gc_mod.OUT.stat().st_mtime_ns
    gc_mod.main()
    assert gc_mod.OUT.read_bytes() == before
    assert gc_mod.OUT.stat().st_mtime_ns == mtime


def test_get_contracts_fails_and_writes_nothing_when_labels_are_missing(gc_mod, monkeypatch):
    lines = fake_golden(3).decode().splitlines()
    rec = json.loads(lines[2])
    del rec["labels"]
    lines[2] = json.dumps(rec)
    payload = ("\n".join(lines) + "\n").encode()
    monkeypatch.setattr(gc_mod.urllib.request, "urlopen", lambda *a, **k: Resp(payload))
    with pytest.raises(SystemExit) as exc:
        gc_mod.main()
    msg = str(exc.value.code)
    assert exc.value.code not in (0, None) and "labels" in msg and rec["doc_id"] in msg
    assert not gc_mod.OUT.exists() and not gc_mod.GOLD.exists()


def test_gold_jsonl_is_git_ignored_and_eval_outputs_are_not():
    def ignored(p):
        # `check-ignore -v` also reports negated ("!") patterns, which mean NOT ignored.
        r = subprocess.run(["git", "check-ignore", "-v", p], cwd=ROOT, capture_output=True, text=True)
        return r.returncode == 0 and not r.stdout.split(":", 2)[2].lstrip("0123456789:").startswith("!")
    assert ignored("data/gold.jsonl")
    assert not ignored("runs/eval_candidates.csv")
    assert not ignored("runs/eval_candidates_summary.json")
    assert ignored("runs/eval_gold_clauses.csv")


# ---------------------------------------------------------------- candidate-config sweep (ADR 0006)


def test_baseline_is_the_7a_selection_and_its_name(env):
    assert env.ec.BASELINE == (15, 0, 25)
    assert (regimes.DEFAULT_K, regimes.HOP_RESERVE, regimes.MAX_CANDIDATES) == env.ec.BASELINE
    assert regimes.candidate_config_name(*env.ec.BASELINE) == "k15" == DEFAULT_CFG


def test_hop_kind_joins_non_seed_kinds(env):
    assert env.ec.hop_kind(("seed",)) == "" and env.ec.hop_kind(()) == ""
    assert env.ec.hop_kind(("definition", "reference_in", "seed")) == "definition+reference_in"


def test_grid_gives_one_row_set_per_config_with_split_and_kept_counts(env):
    args = ("--k", "15", "--k", "25", "--hop-reserve", "0", "--hop-reserve", "5")
    run(env, *args)
    first = outputs(env)
    rows = read_csv(env.tmp / "runs" / "e.csv")
    names = ["k15", "k15_r5", "k25", "k25_r5"]
    n_regimes = len({r["regime"] for r in rows})
    for n in names:
        assert len([r for r in rows if r["config"] == n]) == 3 * n_regimes
    for r in rows:
        assert r["split"] == ev.split_of(r["doc_id"])
        assert int(r["seeds_kept"]) + int(r["hops_kept"]) == int(r["candidate_clauses"])
        assert {"omitted_seeds", "omitted_hops", "hop_reserve", "max_candidates",
                "hops_from_cut_seeds"} <= set(r)
    s = json.loads((env.tmp / "runs" / "s.json").read_text())
    assert list(s["by_config"]) == names
    assert s["config"]["hop_reserve"] == [0, 5]
    assert s["by_config"]["k25_r5"]["hop_reserve"] == 5
    assert set(s["by_config"]["k15"]["by_split"]) == {"dev"}  # DOC-A/B/C hash to dev
    g = read_csv(env.tmp / "runs" / "g.csv")
    assert {f"{n}{suffix}" for n in names for suffix in ("", "_excerpt", "_hop_kind")} <= set(g[0])
    for n in ("e.csv", "s.json", "g.csv"):
        (env.tmp / "runs" / n).unlink()
    run(env, *args)
    assert outputs(env) == first


def test_hop_kind_is_logged_per_hit_and_summarised(env):
    run(env)
    g = read_csv(env.tmp / "runs" / "g.csv")
    row = next(r for r in g if r["doc_id"] == "DOC-B" and r["category"] == "anti_assignment")
    assert (row[DEFAULT_CFG], row[f"{DEFAULT_CFG}_hop_kind"]) == ("hop", "reference_out")
    seed = next(r for r in g if r["doc_id"] == "DOC-B" and r["category"] == "governing_law")
    assert (seed[DEFAULT_CFG], seed[f"{DEFAULT_CFG}_hop_kind"]) == ("seed", "")
    s = json.loads((env.tmp / "runs" / "s.json").read_text())["by_config"][DEFAULT_CFG]
    assert s["hits_by_hop_kind"]["all"] == {"reference_out": s["all"]["hits_hop"]}
    assert s["hits_by_hop_kind"]["by_regime"] == {"Assignment & Control": {"reference_out": 3}}


def test_hops_from_cut_seeds_counted_per_row_and_summarised(env):
    """Seeds are clauses 1 and 3; clause 2 is a hop of seed 3. With one slot reserved for
    a hop and a cap of 1, clause 2 is kept and its only source seed is cut."""
    run(env, "--hop-reserve", "0", "--hop-reserve", "1", "--max-candidates", "1",
        "--max-candidates", "2")
    rows = read_csv(env.tmp / "runs" / "e.csv")
    by = {}
    for r in rows:
        by.setdefault(r["config"], set()).add((r["hops_kept"], r["hops_from_cut_seeds"]))
    assert by == {"k15_c1": {("0", "0")}, "k15_r1_c1": {("1", "1")},
                  "k15_c2": {("0", "0")}, "k15_r1_c2": {("1", "1")}}
    s = json.loads((env.tmp / "runs" / "s.json").read_text())["by_config"]
    a = s["k15_r1_c1"]["all"]
    assert (a["hops_from_cut_seeds"], a["share_hops_from_cut_seeds"]) == (a["hops_kept"], 1.0)
    assert s["k15_c2"]["all"]["share_hops_from_cut_seeds"] is None  # no hops kept


def test_other_caps_are_report_only_and_never_chosen(env):
    out = env.tmp / "runs" / "sweep.json"
    run(env, "--k", "15", "--hop-reserve", "0", "--hop-reserve", "5",
        "--max-candidates", "25", "--max-candidates", "30", "--sweep-out", str(out))
    sw = json.loads(out.read_text())
    assert sw["baseline"] == "k15"
    assert sw["report_only"] == ["k15_c30", "k15_r5_c30"]
    assert set(sw["grid"]) == {"k15", "k15_r5", "k15_c30", "k15_r5_c30"}
    table = {t["config"]: t for t in sw["decision"]["table"]}
    assert table["k15_c30"]["fails"][0] == "report_only"
    assert sw["decision"]["chosen"] == "k15" and not sw["adopted"]
    assert "hops_from_cut_seeds" in sw["grid"]["k15_r5"]["all"]
    assert "Lorem" not in out.read_text() and GOLD_TAIL not in out.read_text()


def test_sweep_does_not_overwrite_the_7a_outputs(env):
    run(env, "--k", "15", "--k", "25")
    before = outputs(env)
    out = env.tmp / "runs" / "sweep.json"
    run(env, "--k", "15", "--k", "25", "--hop-reserve", "0", "--hop-reserve", "5",
        "--sweep-out", str(out))
    assert outputs(env) == before
    rows = read_csv(env.tmp / "runs" / "sweep_rows.csv")
    assert {r["config"] for r in rows} == {"k15", "k15_r5", "k25", "k25_r5"}
    gold = read_csv(env.tmp / "runs" / "sweep_gold.csv")
    assert {"k25_r5", "k25_r5_hop_kind"} <= set(gold[0])
    assert set(json.loads(out.read_text())["grid"]) == {"k15", "k15_r5", "k25", "k25_r5"}


def test_sweep_needs_the_baseline_in_the_grid(env):
    with pytest.raises(SystemExit):
        run(env, "--k", "25", "--sweep-out", str(env.tmp / "runs" / "sweep.json"))


def sweep_summary(names, dev_hits):
    """by_config summary with just what sweep() reads."""
    by = {}
    for name, (hits, ro) in zip(names, dev_hits):
        d = {"hits": hits, "gold_clauses": 100, "micro": hits / 100, "candidate_chars": 10,
             "chars_sent_ratio": 0.3, "mean_candidates": 15.0}
        by[name] = {"k": 15, "hop_reserve": 0, "max_candidates": 25, "report_only": ro,
                    "all": {"hits": hits, "micro_excerpt": 0.5, "span_recall": 0.5,
                            "hits_hop": 1, "chars_sent_ratio": 0.3, "mean_candidates": 15.0},
                    "by_regime": {}, "by_health": {}, "by_split": {"dev": d, "test": {"hits": 5}},
                    "by_split_regime": {"dev": {"A": {"hits": hits}}}, "hits_by_hop_kind": {}}
    return {"config": {k: None for k in ("k", "hop_reserve", "max_candidates", "candidate_chars",
                                         "contracts", "embed_model", "split")},
            "by_config": by}


def test_sweep_never_adopts_or_checks_a_report_only_winner(env):
    s = sweep_summary(["k15", "k25_r5", "k40_r5_c30"], [(50, False), (50, False), (90, True)])
    out = env.ec.sweep(s, "k15")
    assert out["decision"]["best"] == "k15" and out["adoption_check"] is None
    assert not out["adopted"] and out["report_only"] == ["k40_r5_c30"]


def test_sweep_adopted_needs_dev_gain_and_adoption_check(env):
    out = env.ec.sweep(sweep_summary(["k15", "k25_r5"], [(50, False), (60, False)]), "k15")
    assert out["decision"]["adopt"] and out["adoption_check"]["config"] == "k25_r5"
    assert out["adopted"] == out["adoption_check"]["pass"]  # by_regime etc. are empty here


def test_sweep_json_is_committed_like_the_other_eval_outputs():
    r = subprocess.run(["git", "check-ignore", "-v", "runs/eval_candidates_sweep.json"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 or r.stdout.split(":", 2)[2].lstrip("0123456789:").startswith("!")
    for path in ("runs/eval_candidates_sweep_rows.csv", "runs/eval_gold_clauses_sweep.csv"):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0


# ---------------------------------------------------------------- committed sweep artefact

SWEEP = ROOT / "runs" / "eval_candidates_sweep.json"


@pytest.fixture(scope="module")
def sweep_json():
    if not SWEEP.exists():
        pytest.skip("runs/eval_candidates_sweep.json missing")
    return json.loads(SWEEP.read_text())


def test_sweep_grid_uses_canonical_names(sweep_json):
    want = {regimes.candidate_config_name(k, r, c)
            for k in (15, 25, 40) for r in (0, 5, 10) for c in (25, 30)}
    assert set(sweep_json["grid"]) == want and len(want) == 18
    for name, c in sweep_json["grid"].items():
        assert regimes.candidate_config_name(c["k"], c["hop_reserve"], c["max_candidates"]) == name


def test_sweep_decision_reproduces_from_its_own_grid(sweep_json):
    """Re-run decide() on the stored per-config dev numbers: same table, same verdict."""
    g = sweep_json["grid"]
    cfgs = {n: {"report_only": c["report_only"],
                "dev": {**c["by_split"]["dev"], "by_regime": c["by_split_regime"]["dev"]}}
            for n, c in g.items()}
    d = ev.decide(cfgs, sweep_json["baseline"])
    assert d == sweep_json["decision"]
    assert all(t["config"].endswith("_c30") == t["report_only"] for t in d["table"])
    assert not any(t["eligible"] for t in d["table"] if t["report_only"])
    assert d["chosen"] == "k15" and sweep_json["adopted"] is False


def test_sweep_json_has_no_text_fields(sweep_json):
    blob = SWEEP.read_text()
    assert '"text"' not in blob and '"excerpt"' not in blob
