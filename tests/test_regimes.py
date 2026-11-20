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
