"""Step 6: deterministic validation of LLM regime assignments (fake responses, no API)."""

import json

import pytest

from clause_graph import llm, regimes
from clause_graph.queries import QueryError
from clause_graph.regimes import validate

TEXTS = {
    "0": "THIS AGREEMENT is made between Dynamex and the Carrier.",
    "1.1": '"Freight" means all goods tendered by Dynamex to Carrier.',
    "8.1": "Carrier shall indemnify Dynamex against all claims arising hereunder.",
    "8.2": "In no event shall Carrier's liability exceed the insurance limits in Section 9.0.",
    "9.1": "Carrier shall maintain cargo insurance of not less than [***] per occurrence.",
    "12": "This Agreement is governed by the laws of Texas.",
}
CANDIDATES = {"1.1", "8.1", "8.2", "9.1"}


def member(section, role="rule", evidence=None, confidence=0.9):
    return {"section": section, "role": role,
            "evidence": TEXTS[section] if evidence is None else evidence,
            "confidence": confidence}


def answer(members=(), relies_on=()):
    return {"members": list(members), "relies_on": list(relies_on)}


def test_verbatim_member_is_accepted_with_whitespace_normalised():
    ev = "In no event  shall Carrier's\nliability exceed"
    v = validate(answer([member("8.2", "cap", ev)]), CANDIDATES, TEXTS)
    assert v.error is None
    assert [(m.data["section"], m.status, m.checks) for m in v.members] == [("8.2", "accepted", [])]


def test_empty_member_list_is_a_valid_answer():
    v = validate(answer(), CANDIDATES, TEXTS)
    assert v.error is None and v.members == [] and v.counts()["members"] == 0


@pytest.mark.parametrize(
    "evidence, check",
    [
        ("Carrier shall indemnify Dynamex against all losses", "evidence_not_verbatim"),
        ("Carrier shall indemnify ... arising hereunder", "evidence_elided"),
        ("Carrier shall indemnify … arising hereunder", "evidence_elided"),
        ("", "evidence_empty"),
    ],
)
def test_bad_evidence_is_flagged_not_dropped(evidence, check):
    v = validate(answer([member("8.1", evidence=evidence)]), CANDIDATES, TEXTS)
    assert len(v.members) == 1
    assert v.members[0].status == "needs_review"
    assert check in v.members[0].checks


def test_redacted_evidence_is_flagged_even_if_verbatim():
    ev = "cargo insurance of not less than [***] per occurrence"
    v = validate(answer([member("9.1", "cap", ev)]), CANDIDATES, TEXTS)
    assert v.members[0].checks == ["evidence_redacted"]


def test_low_or_invalid_confidence_needs_review():
    v = validate(answer([member("8.1", confidence=0.4), member("8.2", "cap", confidence=1.7)]),
                 CANDIDATES, TEXTS)
    checks = {m.data["section"]: m.checks for m in v.members}
    assert checks == {"8.1": ["low_confidence"], "8.2": ["confidence_out_of_range"]}


def test_section_outside_candidates_is_flagged_and_invented_section_rejected():
    v = validate(answer([member("12"), member("99.9", evidence="anything")]), CANDIDATES, TEXTS)
    assert [(m.data["section"], m.checks) for m in v.members] == [("12", ["section_not_candidate"])]
    assert [r["reason"] for r in v.rejected] == ["section_not_in_contract"]


def test_section_spelling_is_normalised():
    v = validate(answer([member("8.2", "cap") | {"section": "Section 8.2."}]), CANDIDATES, TEXTS)
    assert v.members[0].data["section"] == "8.2" and v.members[0].status == "accepted"


def test_one_clause_may_play_two_roles_but_duplicates_collapse():
    ms = [member("8.2", "cap", confidence=0.7), member("8.2", "exclusion"),
          member("8.2", "cap", confidence=0.95)]
    v = validate(answer(ms), CANDIDATES, TEXTS)
    assert sorted((m.data["role"], m.data["confidence"]) for m in v.members) == [
        ("cap", 0.95), ("exclusion", 0.9)]
    assert v.duplicates == 1


def test_relies_on_checks():
    rel = [
        {"from_section": "8.1", "to_section": "8.2", "evidence": "arising hereunder"},
        {"from_section": "8.1", "to_section": "8.1", "evidence": "hereunder"},
        {"from_section": "8.2", "to_section": "1.1", "evidence": "the insurance required above"},
        {"from_section": "12", "to_section": "8.1", "evidence": "governed by the laws"},
    ]
    v = validate(answer(relies_on=rel), CANDIDATES, TEXTS)
    got = {(r.data["from_section"], r.data["to_section"]): r.checks for r in v.relies_on}
    assert got == {
        ("8.1", "8.2"): [],
        ("8.2", "1.1"): ["evidence_not_verbatim"],
        ("12", "8.1"): ["from_section_not_candidate"],
    }
    assert [r["reason"] for r in v.rejected] == ["self_reference"]


def test_schema_violation_and_call_errors_escalate_whole_call():
    assert validate({"members": [{"section": "8.2"}], "relies_on": []},
                    CANDIDATES, TEXTS).error.startswith("schema_error")
    assert validate(answer([member("8.2", "obligation")]),  # role not in enum
                    CANDIDATES, TEXTS).error.startswith("schema_error")
    assert validate(None, CANDIDATES, TEXTS, error="refusal").error == "refusal"
    assert validate(None, CANDIDATES, TEXTS, error="max_tokens").error == "max_tokens"


def test_regimes_config_and_matching():
    names = [r.name for r in regimes.load_regimes()]
    assert names == ["Liability", "Termination", "Assignment & Control",
                     "Restrictive Covenants", "Governing Law & Disputes", "Confidentiality"]
    assert regimes.match_regime("liab").name == "Liability"
    assert regimes.match_regime("governing law & disputes").name == "Governing Law & Disputes"
    with pytest.raises(QueryError, match="Valid regimes"):
        regimes.match_regime("Payment")


def test_prompt_and_schema_agree():
    prompt = regimes.load_prompt()
    for role in regimes.ROLES:
        assert f"- {role}:" in prompt
    schema_roles = regimes.ASSIGNMENT_SCHEMA["properties"]["members"]["items"]["properties"]
    assert schema_roles["role"]["enum"] == list(regimes.ROLES)
    reg = regimes.match_regime("Liability")
    cands = [regimes.Candidate("8.2", "D#8.2", TEXTS["8.2"], TEXTS["8.2"], ["vector"], False)]
    system, user = regimes.build_messages(reg, cands, prompt)
    assert "{{" not in system and reg.description in system
    assert '<clause section="8.2">' in user


def test_cache_key_changes_with_input():
    a = regimes.input_hash("s", "u", "m", "medium")
    assert a == regimes.input_hash("s", "u", "m", "medium")
    assert a != regimes.input_hash("s", "u2", "m", "medium")
    assert a != regimes.input_hash("s", "u", "m2", "medium")
    p = regimes.cache_path("LECLANCHÉ S.A. / X", "Assignment & Control", "m", a)
    assert "/" not in p.name and p.name.endswith(f"__{a}.json")


def test_cost():
    assert llm.cost_usd("claude-sonnet-5-5", 1_000_000, 100_000) == pytest.approx(3.0)
    assert llm.cost_usd("unknown-model", 1, 1) is None


@pytest.mark.skipif(not llm.has_api_key(), reason="ANTHROPIC_API_KEY not set")
def test_api_returns_schema_valid_json():
    schema = {
        "type": "object",
        "properties": {"members": {"type": "array", "items": {"type": "string"}}},
        "required": ["members"],
        "additionalProperties": False,
    }
    c = llm.complete_json("Return the list of fruits mentioned.", "Apples and pears.", schema)
    assert c.error is None, c.raw_text
    assert set(json.loads(c.raw_text)) == {"members"}
    assert c.input_tokens > 0 and c.output_tokens > 0


# ---------------------------------------------------------------- candidate budget (fixture graph)

import csv  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import fake_graph as fg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def cand_split(cands):
    seeds = [c.section for c in cands if "seed" in c.kinds]
    return seeds, [c.section for c in cands if "seed" not in c.kinds]


def test_candidates_at_k15_cap25_ignore_a_reserve_up_to_10():
    """At most 15 seeds fit next to any reserve <= 10, so r changes nothing at k=15."""
    g = fg.wide_graph(30)
    runs = [regimes.candidates(g, fg.DOC, [0.0], 15, 25, r) for r in (0, 5, 10)]
    assert len(runs[0][0]) == 25
    for cands, omitted in runs[1:]:
        assert [(c.uid, c.why, c.kinds) for c in cands] == [
            (c.uid, c.why, c.kinds) for c in runs[0][0]]
        assert omitted == runs[0][1]


def test_candidates_reserve_hop_slots_when_seeds_fill_the_cap():
    g = fg.wide_graph(30)
    cands, omitted = regimes.candidates(g, fg.DOC, [0.0], 25, 25, 0)
    assert cand_split(cands) == ([str(s) for s in range(1, 26)], [])
    cands, omitted = regimes.candidates(g, fg.DOC, [0.0], 25, 25, 5)
    seeds, hops = cand_split(cands)
    assert seeds == [str(s) for s in range(1, 21)]  # best 20 seeds
    assert hops == [str(100 + s) for s in range(1, 6)]  # hops of the best seeds
    assert all(k == ("reference_out",) for k in (c.kinds for c in cands if c.section in hops))
    assert omitted["total"] == 25 and omitted["by_relation"]["seed"] == 5
    assert regimes.candidates(g, fg.DOC, [0.0], 25, 25)[0] == regimes.candidates(
        g, fg.DOC, [0.0], 25, 25, regimes.HOP_RESERVE)[0]


def test_candidates_record_hop_sources_including_cut_seeds():
    """cap 4, r 3: one seed kept; hops 102 and 103 come from seeds 2 and 3, which are cut."""
    cands, _ = regimes.candidates(fg.wide_graph(10), fg.DOC, [0.0], 10, 4, 3)
    got = {c.section: c.sources for c in cands}
    assert got == {"1": (), "101": (fg.uid(1),), "102": (fg.uid(2),), "103": (fg.uid(3),)}


def test_candidate_config_name_is_canonical():
    n = regimes.candidate_config_name
    assert n(15, 0, 25) == "k15"  # the 7a selection: r = 0 and cap 25 are elided
    assert n(25, 5, 25) == "k25_r5"
    assert n(15, 0, 30) == "k15_c30"
    assert n(40, 10, 30) == "k40_r10_c30"
    assert n(regimes.DEFAULT_K, regimes.HOP_RESERVE, regimes.MAX_CANDIDATES) == "k15"


def test_candidate_kinds_mark_seeds_and_hops():
    cands, _ = regimes.candidates(fg.small_graph(), fg.DOC, [0.0], 8, 25)
    by = {c.section: c.kinds for c in cands}
    assert by["1"] == ("seed",) and by["20"] == ("definition",)
    assert by["6"] == ("reference_out", "seed")  # seed and hop target
    assert "25" not in by  # siblings are not a candidate kind


class ExtractGraph(fg.FakeGraph):
    """FakeGraph plus the queries extract_regimes.py issues itself."""

    def execute_query(self, query, params=None, routing_=None, **kw):
        if "k.doc_id AS d" in query:
            return fg.result([{"d": fg.DOC}])
        if "RETURN c.section AS section, c.text AS text" in query:
            return fg.result([{"section": str(s), "text": fg.text(s)} for s in range(1, 9)])
        if "e.prompt_version = $pv" in query:
            return fg.result([])
        return super().execute_query(query, params, routing_)

    def session(self):
        graph = self

        class S:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute_write(self, *a, **k):
                graph.writes = getattr(graph, "writes", 0) + 1
        return S()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def extract(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("extract_regimes", ROOT / "scripts" / "extract_regimes.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    contracts = tmp_path / "contracts.jsonl"
    contracts.write_text(json.dumps({"doc_id": fg.DOC, "text": "synthetic"}) + "\n")
    monkeypatch.setattr(ex, "CONTRACTS", contracts)
    monkeypatch.setattr(ex, "ROOT", tmp_path)
    monkeypatch.setattr(regimes, "CACHE", tmp_path / "llm_cache")
    monkeypatch.setattr(ex, "METRICS_CSV", tmp_path / "metrics.csv")
    monkeypatch.setattr(ex, "CALLS_LOG", tmp_path / "calls.jsonl")
    monkeypatch.setattr(ex, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(ex, "embed_query", lambda text: [0.0])
    monkeypatch.setattr(ex.llm, "config", lambda: llm.Config("anthropic", "m", "m2", "low"))
    monkeypatch.setattr(ex.llm, "has_api_key", lambda: False)

    def no_api(*a, **k):
        raise AssertionError("LLM called")
    monkeypatch.setattr(ex.llm, "complete_json", no_api)
    graph = fg.wide_graph(30)
    graph.__class__ = ExtractGraph
    monkeypatch.setattr(ex.GraphDatabase, "driver", staticmethod(lambda *a, **k: graph))
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    return ex, tmp_path, monkeypatch


def test_dry_run_prints_candidate_config_without_api(extract, capsys):
    ex, tmp, mp = extract
    mp.setattr(sys, "argv", ["extract_regimes.py", "--dry-run", "--regime", "Liability",
                             "--k", "25", "--hop-reserve", "5"])
    assert ex.main() == 0
    out = capsys.readouterr().out
    assert "k=25 max_candidates=25 hop_reserve=5 DRY RUN" in out
    assert "cand=25" in out
    assert not (tmp / "metrics.csv").exists()


def fake_call(extract):
    ex, tmp, mp = extract
    record = {"error": None, "raw_text": json.dumps({"members": [], "relies_on": []}),
              "input_tokens": 10, "output_tokens": 5}
    calls = []
    mp.setattr(ex, "call_llm", lambda *a, **k: calls.append(1) or (record, True))
    mp.setattr(ex.llm, "has_api_key", lambda: True)  # call_llm is faked: no API either way
    return calls


def test_call_log_and_metrics_record_candidate_config(extract):
    ex, tmp, mp = extract
    fake_call(extract)
    mp.setattr(sys, "argv", ["extract_regimes.py", "--regime", "Liability", "--k", "25",
                             "--max-candidates", "24"])
    assert ex.main() == 0
    call = json.loads((tmp / "calls.jsonl").read_text().splitlines()[0])
    assert (call["k"], call["max_candidates"], call["hop_reserve"]) == (25, 24, 0)
    assert call["candidate_config"] == "k25_c24"
    assert len(call["candidates"]) == 24
    with (tmp / "metrics.csv").open() as f:
        assert next(csv.DictReader(f))["candidate_config"] == "k25_c24"


def test_hop_reserve_without_dry_run_is_refused_before_anything_runs(extract, capsys):
    ex, tmp, mp = extract
    calls = fake_call(extract)
    graph = ex.GraphDatabase.driver()
    for r in ("5", "-1"):
        mp.setattr(sys, "argv", ["extract_regimes.py", "--regime", "Liability",
                                 "--hop-reserve", r])
        with pytest.raises(SystemExit) as e:
            ex.main()
        assert "--dry-run" in str(e.value)
    assert calls == [] and getattr(graph, "writes", 0) == 0
    assert not (tmp / "metrics.csv").exists() and not (tmp / "calls.jsonl").exists()
    mp.setattr(sys, "argv", ["extract_regimes.py", "--regime", "Liability", "--hop-reserve",
                             str(regimes.HOP_RESERVE)])
    assert ex.main() == 0 and calls == [1]  # the adopted default still writes


def read_metrics(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_write_metrics_keeps_rows_of_other_candidate_configs(extract):
    ex, tmp, _ = extract
    path = tmp / "metrics.csv"
    path.write_text("doc_id,model,prompt_version,regimes\nOLD,m,p1,3\n")

    def row(doc, cfg, n):
        return {"doc_id": doc, "model": "m", "prompt_version": "p1",
                "candidate_config": cfg, "regimes": n}
    ex.write_metrics([row("OLD", "k15", 9), row("D", "k15", 1)])
    ex.write_metrics([row("D", "k25_c30", 2)])
    ex.write_metrics([row("D", "k15", 4)])
    got = [(r["doc_id"], r["candidate_config"], r["regimes"]) for r in read_metrics(path)]
    # The old row (written before candidate_config existed) is blank, kept, not shifted.
    assert sorted(got) == [("D", "k15", "4"), ("D", "k25_c30", "2"), ("OLD", "", "3"),
                           ("OLD", "k15", "9")]
