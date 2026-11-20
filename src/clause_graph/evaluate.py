"""Step 7a: evaluation against the CUAD gold labels (ADR 0005). Pure: no model, no database.

Gold (data/gold.jsonl, written by scripts/get_contracts.py) has one row per contract:
{"doc_id", "labels": {<category>: {"present": bool, "evidence": [span, ...]}}}.
Gold is defined by the evidence spans, not by the "present" flag.

1. category_to_regime() - CUAD category -> regime, inverted from regimes.json.
2. match_contract()     - each gold span -> the clauses it lies in. Exact text after
                          whitespace collapse; no fuzzy matching, no case folding.
   a. "contained": every clause whose collapsed text contains the span.
   b. "overlap":   otherwise the span is placed in the collapsed source text and mapped
                   to every clause it overlaps. Clause positions in the source come from
                   finding each clause's collapsed text in document order (Clause.position).
   c. unmatched:   "not_in_source", "no_clause" or "no_source" (the contract has no source
                   text to place the span in) - counted and listed, never dropped.
3. score()              - which gold clauses are candidates (seed or graph hop), and whether
                          the excerpt actually sent to the LLM contains the gold text.
   span_hits()          - secondary, span level: a matched span is hit when ANY of its
                          clauses is a candidate (a span in N clauses is N gold clauses but
                          one gold span).
4. split_of(), decide(), adoption_check() - candidate-budget retune: a hashed dev/test split,
                          the selection rule fixed before the sweep, and the adoption targets.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from clause_graph.chunking import collapse
from clause_graph.regimes import Regime, load_regimes

CONTAINED = "contained"
OVERLAP = "overlap"
NOT_IN_SOURCE = "not_in_source"  # span text does not occur in the contract source
NO_CLAUSE = "no_clause"  # span is in the source but overlaps no located clause
NO_SOURCE = "no_source"  # not contained in a clause and the contract has no source text
EMPTY = "empty_span"
SEED_PREFIX = "vector match"  # Candidate.why reason of a vector seed (queries.collect_context)


# ---------------------------------------------------------------- 1. categories


def category_key(name: str) -> str:
    """regimes.json names ("Cap On Liability", "Anti-Assignment") -> gold keys
    ("cap_on_liability", "anti_assignment")."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def category_to_regime(regimes: list[Regime] | None = None) -> dict[str, str]:
    """Gold category key -> regime name. A category listed under two regimes is an error."""
    out: dict[str, str] = {}
    for r in regimes if regimes is not None else load_regimes():
        for cat in r.cuad_categories:
            key = category_key(cat)
            if key in out and out[key] != r.name:
                raise ValueError(f"CUAD category {cat!r} is mapped to {out[key]!r} and {r.name!r}")
            out[key] = r.name
    return out


def category_report(
    gold: list[dict], mapping: dict[str, str], regimes: list[Regime] | None = None
) -> dict:
    """Gold categories without a regime ("unmapped"), mapped categories absent from the
    gold ("missing_in_gold") and regimes without any CUAD category ("no_gold")."""
    regimes = regimes if regimes is not None else load_regimes()
    seen = sorted({c for row in gold for c in row["labels"]})
    return {
        "unmapped": [c for c in seen if c not in mapping],
        "missing_in_gold": sorted(c for c in mapping if c not in seen),
        "no_gold": [r.name for r in regimes if not r.cuad_categories],
    }


def load_gold(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def present_disagreements(gold: list[dict]) -> list[dict]:
    """(doc_id, category) where the present flag and having evidence disagree. No text."""
    out = []
    for row in gold:
        for cat in sorted(row["labels"]):
            label = row["labels"][cat]
            spans = len(label.get("evidence") or [])
            if bool(label.get("present")) != (spans > 0):
                out.append({"doc_id": row["doc_id"], "category": cat,
                            "present": bool(label.get("present")), "evidence_spans": spans})
    return out


# ---------------------------------------------------------------- 2. span matching


@dataclass(frozen=True)
class ClauseText:
    uid: str
    section: str
    text: str


@dataclass(frozen=True)
class GoldClause:
    """One (gold span, clause) pair, with provenance. piece = the collapsed part of the
    span that lies in this clause (the whole span when contained)."""

    doc_id: str
    category: str
    span_idx: int
    uid: str
    section: str
    method: str  # CONTAINED | OVERLAP
    piece: str


@dataclass(frozen=True)
class SpanMatch:
    doc_id: str
    category: str
    span_idx: int
    status: str  # CONTAINED | OVERLAP | NOT_IN_SOURCE | NO_CLAUSE | NO_SOURCE | EMPTY
    clauses: tuple[GoldClause, ...]

    @property
    def matched(self) -> bool:
        return bool(self.clauses)


def _occurrences(haystack: str, needle: str) -> list[int]:
    out, i = [], haystack.find(needle)
    while i >= 0:
        out.append(i)
        i = haystack.find(needle, i + 1)
    return out


def locate_clauses(source: str, clauses: list[ClauseText]) -> list[tuple[int, int] | None]:
    """(start, end) of each clause's collapsed text in the collapsed source, searched in
    document order from the end of the previous clause (first occurrence anywhere as a
    fallback). None when the clause text is not in the source."""
    src = collapse(source)
    out: list[tuple[int, int] | None] = []
    cursor = 0
    for c in clauses:
        text = collapse(c.text)
        if not text:
            out.append(None)
            continue
        i = src.find(text, cursor)
        if i < 0:
            i = src.find(text)
        if i < 0:
            out.append(None)
            continue
        out.append((i, i + len(text)))
        cursor = i + len(text)
    return out


def match_span(
    doc_id: str,
    category: str,
    span_idx: int,
    span: str,
    clauses: list[ClauseText],
    collapsed: list[str],
    source: str | None,
    located: list[tuple[int, int] | None],
) -> SpanMatch:
    """source is the collapsed contract text, None when the contract has no source row."""
    s = collapse(span)
    if not s:
        return SpanMatch(doc_id, category, span_idx, EMPTY, ())
    hits = tuple(
        GoldClause(doc_id, category, span_idx, c.uid, c.section, CONTAINED, s)
        for c, text in zip(clauses, collapsed)
        if s in text
    )
    if hits:
        return SpanMatch(doc_id, category, span_idx, CONTAINED, hits)
    if source is None:
        return SpanMatch(doc_id, category, span_idx, NO_SOURCE, ())
    starts = _occurrences(source, s)
    if not starts:
        return SpanMatch(doc_id, category, span_idx, NOT_IN_SOURCE, ())
    found: dict[str, GoldClause] = {}
    for a in starts:
        b = a + len(s)
        for c, loc in zip(clauses, located):
            if loc is None or loc[1] <= a or loc[0] >= b or c.uid in found:
                continue
            piece = source[max(a, loc[0]) : min(b, loc[1])].strip()
            if piece:
                found[c.uid] = GoldClause(doc_id, category, span_idx, c.uid, c.section,
                                          OVERLAP, piece)
    if not found:
        return SpanMatch(doc_id, category, span_idx, NO_CLAUSE, ())
    return SpanMatch(doc_id, category, span_idx, OVERLAP, tuple(found.values()))


def match_contract(
    doc_id: str, labels: dict, clauses: list[ClauseText], source: str | None
) -> list[SpanMatch]:
    """Every gold span of one contract, in (category, span index) order.
    clauses must be in document order (Clause.position). source=None (no source row for
    the contract): only "contained" matching is possible, other spans get NO_SOURCE."""
    collapsed = [collapse(c.text) for c in clauses]
    src = None if source is None else collapse(source)
    located = [None] * len(clauses) if source is None else locate_clauses(source, clauses)
    out = []
    for cat in sorted(labels):
        for idx, span in enumerate(labels[cat].get("evidence") or []):
            out.append(match_span(doc_id, cat, idx, span, clauses, collapsed, src, located))
    return out


# ---------------------------------------------------------------- 3. scoring


@dataclass(frozen=True)
class Sent:
    """What candidate selection produced for one clause: was it a vector seed, and the
    excerpt sent to the LLM."""

    uid: str
    seed: bool
    excerpt: str


def is_seed(why: list[str]) -> bool:
    return any(w.startswith(SEED_PREFIX) for w in why)


@dataclass
class Score:
    gold_clauses: int
    hits: int
    hits_seed: int
    hits_hop: int
    hits_excerpt: int
    outcome: dict[str, tuple[str, bool]]  # gold uid -> ("seed"|"hop"|"miss", in excerpt)

    @property
    def recall(self) -> float | None:
        return self.hits / self.gold_clauses if self.gold_clauses else None

    @property
    def recall_excerpt(self) -> float | None:
        return self.hits_excerpt / self.gold_clauses if self.gold_clauses else None


def score(gold: list[GoldClause], sent: list[Sent]) -> Score:
    """Clause-level recall of the candidates against the gold clauses. A gold clause is an
    excerpt hit when at least one of its gold pieces is verbatim (whitespace-collapsed) in
    the excerpt sent for it, i.e. the LLM saw some gold evidence for that clause."""
    pieces: dict[str, list[str]] = {}
    for g in gold:
        pieces.setdefault(g.uid, []).append(g.piece)
    by_uid = {s.uid: s for s in sent}
    outcome: dict[str, tuple[str, bool]] = {}
    for uid in sorted(pieces):
        s = by_uid.get(uid)
        if s is None:
            outcome[uid] = ("miss", False)
            continue
        excerpt = collapse(s.excerpt)
        outcome[uid] = ("seed" if s.seed else "hop", any(p in excerpt for p in pieces[uid]))
    hits = [o for o in outcome.values() if o[0] != "miss"]
    return Score(
        gold_clauses=len(outcome),
        hits=len(hits),
        hits_seed=sum(o[0] == "seed" for o in hits),
        hits_hop=sum(o[0] == "hop" for o in hits),
        hits_excerpt=sum(o[1] for o in hits),
        outcome=outcome,
    )


def span_hits(matches: list[SpanMatch], candidate_uids: set[str]) -> tuple[int, int]:
    """(matched spans, matched spans with at least one clause among the candidates)."""
    matched = [m for m in matches if m.matched]
    return len(matched), sum(any(g.uid in candidate_uids for g in m.clauses) for m in matched)


def micro_macro(pairs: list[tuple[int, int]]) -> dict:
    """pairs of (hits, gold_clauses), one per scored unit. micro = sum(hits) / sum(gold);
    macro = mean of hits / gold over the units. Units without gold clauses are left out of
    recall and counted separately. For a per-contract macro over several regimes, sum each
    contract's pairs first (per_contract())."""
    scored = [(h, g) for h, g in pairs if g]
    hits = sum(h for h, _ in scored)
    gold = sum(g for _, g in scored)
    return {
        "pairs_scored": len(scored),
        "pairs_no_gold": len(pairs) - len(scored),
        "gold_clauses": gold,
        "hits": hits,
        "micro": round(hits / gold, 4) if gold else None,
        "macro": round(sum(h / g for h, g in scored) / len(scored), 4) if scored else None,
    }


def per_contract(rows: list[tuple[str, int, int]]) -> list[tuple[int, int]]:
    """(doc_id, hits, gold) rows -> one (hits, gold) pair per doc_id, summed, in first-seen
    order. micro_macro() of the result has the same micro and a true per-contract macro."""
    out: dict[str, list[int]] = {}
    for doc, h, g in rows:
        acc = out.setdefault(doc, [0, 0])
        acc[0] += h
        acc[1] += g
    return [(h, g) for h, g in out.values()]


# ---------------------------------------------------------------- 4. retune: split + rule


def split_of(doc_id: str) -> str:
    """Deterministic dev/test split, no database: about 1 in 4 contracts is "test"."""
    h = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16)
    return "test" if h % 4 == 0 else "dev"


# Fixed before the sweep was run (retune spec, criterion 11); do not tune on results.
DECISION_RULE = {
    "max_chars_sent_ratio": 0.39,  # dev
    "max_mean_candidates": 20.0,  # dev
    "max_regime_loss": 1,  # gold clauses lost per regime vs the baseline, dev
    "min_gain": 2,  # dev gold clauses over the baseline needed to change the defaults
}


def decide(configs: dict[str, dict], baseline: str, rule: dict = DECISION_RULE) -> dict:
    """Pick a candidate config on the dev split.

    configs: name -> {"report_only": bool, "dev": {hits, gold_clauses, micro,
    candidate_chars, chars_sent_ratio, mean_candidates, by_regime: {regime: {hits}}}},
    in grid order. Highest dev micro clause recall among eligible configs (not report-only,
    within the cost limits, no regime losing more than max_regime_loss gold clauses);
    ties go to fewer chars sent, then grid order. The defaults change only if the best
    config beats the baseline by at least min_gain dev gold clauses."""
    base = configs[baseline]["dev"]
    table = []
    for idx, (name, c) in enumerate(configs.items()):
        d = c["dev"]
        fails = []
        if c["report_only"]:
            fails.append("report_only")
        if d["chars_sent_ratio"] is not None and d["chars_sent_ratio"] > rule["max_chars_sent_ratio"]:
            fails.append("chars_sent_ratio")
        if d["mean_candidates"] > rule["max_mean_candidates"]:
            fails.append("mean_candidates")
        losses = {
            regime: b["hits"] - d["by_regime"].get(regime, {}).get("hits", 0)
            for regime, b in base["by_regime"].items()
        }
        too_many = sorted(r for r, n in losses.items() if n > rule["max_regime_loss"])
        if too_many:
            fails.append("regime_loss: " + ", ".join(too_many))
        table.append({
            "config": name, "order": idx, "report_only": c["report_only"],
            "hits": d["hits"], "gold_clauses": d["gold_clauses"], "micro": d["micro"],
            "gain": d["hits"] - base["hits"],
            "max_regime_loss": max(losses.values(), default=0),
            "chars_sent_ratio": d["chars_sent_ratio"], "candidate_chars": d["candidate_chars"],
            "mean_candidates": d["mean_candidates"], "eligible": not fails, "fails": fails,
        })
    eligible = sorted((t for t in table if t["eligible"]),
                      key=lambda t: (-t["hits"], t["candidate_chars"], t["order"]))
    best = eligible[0] if eligible else None
    adopt = bool(best and best["config"] != baseline and best["gain"] >= rule["min_gain"])
    if best is None:
        reason = "no eligible config"
    elif adopt:
        reason = f"{best['config']} beats {baseline} by {best['gain']} dev gold clauses"
    else:
        reason = (f"best eligible config {best['config']} gains {best['gain']} dev gold "
                  f"clauses (< {rule['min_gain']}): defaults unchanged")
    for t in table:
        del t["order"]
    return {"rule": rule, "baseline": baseline, "best": best["config"] if best else None,
            "adopt": adopt, "chosen": best["config"] if adopt else baseline,
            "reason": reason, "table": table}


ADOPTION_TARGETS = {
    "min_extra_gold_clauses": 4,  # all contracts, vs the baseline (282/330 -> >= 0.865)
    "max_chars_sent_ratio": 0.39,
    "max_mean_candidates": 20.0,
    "max_high_caps_chars_sent_ratio": 0.25,
    "max_test_loss": 1,  # gold clauses on the test split vs the baseline
}


def adoption_check(chosen: dict, baseline: dict, targets: dict = ADOPTION_TARGETS) -> dict:
    """Retune criteria 12-16 on all contracts. chosen / baseline: per-config summaries with
    all, by_regime, by_health and by_split (scripts/eval_candidates.py). Every target is
    listed with its value, threshold and result; adopt only if all pass."""
    a, b = chosen["all"], baseline["all"]
    out = []

    def check(name, value, op, threshold):
        ok = value is not None and threshold is not None and (
            value >= threshold if op == ">=" else value <= threshold if op == "<=" else value > threshold)
        out.append({"target": name, "value": value, "op": op, "threshold": threshold,
                    "pass": bool(ok)})

    check("12 clause hits (all)", a["hits"], ">=", b["hits"] + targets["min_extra_gold_clauses"])
    check("12 micro excerpt recall", a["micro_excerpt"], ">=", b["micro_excerpt"])
    check("12 span recall", a["span_recall"], ">=", b["span_recall"])
    for regime, r in baseline["by_regime"].items():
        if r["micro"] is not None:
            check(f"13 micro clause recall {regime}",
                  chosen["by_regime"].get(regime, {}).get("micro"), ">=", r["micro"])
    check("14 hits_hop", a["hits_hop"], ">", 0)
    check("15 chars_sent_ratio", a["chars_sent_ratio"], "<=", targets["max_chars_sent_ratio"])
    check("15 mean_candidates", a["mean_candidates"], "<=", targets["max_mean_candidates"])
    hc = chosen["by_health"].get("high_caps")
    if hc:
        check("15 high_caps chars_sent_ratio", hc["chars_sent_ratio"], "<=",
              targets["max_high_caps_chars_sent_ratio"])
    bt, ct = baseline["by_split"].get("test"), chosen["by_split"].get("test")
    if bt and ct:
        check("16 test clause hits", ct["hits"], ">=", bt["hits"] - targets["max_test_loss"])
    return {"targets": targets, "checks": out, "pass": all(c["pass"] for c in out)}
