"""Step 7a: candidate recall against the CUAD gold labels (ADR 0005). No LLM.

Usage: uv run scripts/eval_candidates.py [--limit N] [--doc PREFIX ...]
                                         [--k 15 --k 25 --k 40] [--max-candidates 25]

For each contract and each regime with CUAD categories: embed the regime description once
(locally), build the step-6 candidates with regimes.candidates() (vector top-k + one graph
hop) and compare them with the clauses holding the gold evidence spans. This is the upper
limit on LLM recall: a clause that is not a candidate is never shown to the LLM.
Clause level: is a gold clause a candidate (seed or hop)? Excerpt level: does the excerpt
actually sent (cut at regimes.CANDIDATE_CHARS) contain gold evidence for that clause?
Span level (secondary): is at least one clause holding a matched gold span a candidate?
micro = pooled hits / pooled gold; macro = mean per-contract recall (for by_regime and
by_category one row is one contract; for all and by_health each contract's regimes are
summed first).

Reads Neo4j (read-only), data/contracts.jsonl, data/gold.jsonl, runs/extraction_metrics.csv.
Writes runs/eval_candidates.csv (one row per doc_id, regime, k),
       runs/eval_candidates_summary.json (micro/macro recall per regime and category, per k),
       runs/eval_gold_clauses.csv (every gold span -> clause pair with its outcome per k).
No contract or gold text is written or printed.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from clause_graph import evaluate as ev
from clause_graph import queries, regimes
from clause_graph.embeddings import EmbeddingError, embed_query

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = ROOT / "data" / "contracts.jsonl"
GOLD = ROOT / "data" / "gold.jsonl"
EXTRACTION_METRICS = ROOT / "runs" / "extraction_metrics.csv"
OUT_CSV = ROOT / "runs" / "eval_candidates.csv"
OUT_SUMMARY = ROOT / "runs" / "eval_candidates_summary.json"
OUT_GOLD = ROOT / "runs" / "eval_gold_clauses.csv"
HIGH_CAPS = 10  # caps_sections at or above this: ALL-CAPS headings carry much of the structure

CLAUSES = """
MATCH (:Contract {doc_id: $doc})-[:HAS_CLAUSE]->(c:Clause)
RETURN c.uid AS uid, c.section AS section, c.text AS text, c.position AS position
ORDER BY c.position, c.uid
"""


def select_docs(driver, prefixes: list[str], limit: int | None) -> list[str]:
    """Loaded contracts in contracts.jsonl order (as extract_regimes.py), then loaded
    contracts missing from contracts.jsonl, sorted: they are still evaluated, and their
    spans that no clause contains get status no_source instead of disappearing."""
    loaded = {r["d"] for r in queries._read(driver, "MATCH (k:Contract) RETURN k.doc_id AS d")}
    order = [json.loads(line)["doc_id"] for line in CONTRACTS.open(encoding="utf-8") if line.strip()]
    order = [d for d in order if d in loaded]
    order += sorted(loaded - set(order))
    if prefixes:
        return [d for d in order if any(d.startswith(p) for p in prefixes)]
    return order[:limit] if limit else order


def health(sections: int | None, caps: int | None) -> str:
    if sections is None:
        return "unknown"
    if sections == 0:
        return "sections_0"
    if (caps or 0) >= HIGH_CAPS:
        return "high_caps"
    return "ok"


def ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def summarise(rows: list[dict], key: str, per_contract: bool = False) -> dict:
    """micro/macro (clause and excerpt level), span-level micro and candidate cost for rows
    grouped by key. per_contract=True (groups spanning several regimes: all, by_health) sums
    each contract's rows before the macro, so macro is a mean over contracts, not over
    (contract, regime) pairs; pairs_scored / pairs_no_gold still count pairs."""

    def mm(rs: list[dict], hits: str) -> dict:
        pairs = ev.micro_macro([(r[hits], r["gold_clauses"]) for r in rs])
        if per_contract:
            docs = ev.micro_macro(ev.per_contract(
                [(r["doc_id"], r[hits], r["gold_clauses"]) for r in rs]))
            pairs["macro"] = docs["macro"]
            pairs["contracts_scored"] = docs["pairs_scored"]
        return pairs

    out: dict = {}
    for group in dict.fromkeys(r[key] for r in rows):
        rs = [r for r in rows if r[key] == group]
        clause = mm(rs, "hits")
        excerpt = mm(rs, "hits_excerpt")
        spans = sum(r["spans_matched"] for r in rs)
        span_hits = sum(r["span_hits"] for r in rs)
        out[group] = {
            **clause,
            "hits_seed": sum(r["hits_seed"] for r in rs),
            "hits_hop": sum(r["hits_hop"] for r in rs),
            "misses": clause["gold_clauses"] - clause["hits"],
            "hits_excerpt": excerpt["hits"],
            "micro_excerpt": excerpt["micro"],
            "macro_excerpt": excerpt["macro"],
            "spans_matched": spans,
            "span_hits": span_hits,
            "span_recall": ratio(span_hits, spans),
            "mean_candidates": round(sum(r["candidate_clauses"] for r in rs) / len(rs), 2),
            "chars_sent_ratio": ratio(sum(r["candidate_chars"] for r in rs),
                                      sum(r["contract_chars"] for r in rs)),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc", action="append", default=[], help="doc_id prefix (repeatable)")
    parser.add_argument("--k", type=int, action="append", default=[],
                        help=f"top-k chunks (repeatable for a sweep; default {regimes.DEFAULT_K})")
    parser.add_argument("--max-candidates", type=int, default=regimes.MAX_CANDIDATES)
    args = parser.parse_args()
    ks = sorted(set(args.k)) or [regimes.DEFAULT_K]

    load_dotenv(ROOT / ".env")
    # This eval never calls the LLM (llm.py is not imported); make an accidental call impossible.
    os.environ.pop("ANTHROPIC_API_KEY", None)
    if not GOLD.exists():
        sys.exit(f"{GOLD.relative_to(ROOT)} is missing. Run: uv run scripts/get_contracts.py")

    all_regimes = regimes.load_regimes()
    mapping = ev.category_to_regime(all_regimes)
    scorable = [r for r in all_regimes if r.cuad_categories]
    gold_rows = ev.load_gold(GOLD)
    gold = {row["doc_id"]: row["labels"] for row in gold_rows}
    sources = {
        r["doc_id"]: r["text"]
        for r in (json.loads(line) for line in CONTRACTS.open(encoding="utf-8") if line.strip())
    }
    metrics = {}
    if EXTRACTION_METRICS.exists():
        with EXTRACTION_METRICS.open(encoding="utf-8") as f:
            metrics = {r["doc_id"]: r for r in csv.DictReader(f)}
    categories = ev.category_report(gold_rows, mapping, all_regimes)
    if categories["unmapped"]:
        print(f"WARNING unmapped gold categories (not scored): {categories['unmapped']}")

    try:
        vectors = {r.name: embed_query(r.description) for r in scorable}
    except EmbeddingError as e:
        sys.exit(str(e))

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    driver = GraphDatabase.driver(uri, auth=("neo4j", os.environ["NEO4J_PASSWORD"]))
    rows: list[dict] = []
    cat_rows: list[dict] = []
    gold_out: list[dict] = []
    span_matches: list[ev.SpanMatch] = []
    with driver:
        docs = select_docs(driver, args.doc, args.limit)
        print(f"contracts={len(docs)} regimes={len(scorable)} k={ks} "
              f"max_candidates={args.max_candidates}")
        for doc in docs:
            if doc not in gold:
                print(f"{doc[:40]}: no gold row, skipped")
                continue
            clauses = [ev.ClauseText(r["uid"], r["section"], r["text"] or "")
                       for r in queries._read(driver, CLAUSES, doc=doc)]
            total_chars = sum(len(c.text) for c in clauses)
            # None (doc loaded but not in contracts.jsonl) -> NO_SOURCE, not NOT_IN_SOURCE
            matches = ev.match_contract(doc, gold[doc], clauses, sources.get(doc))
            span_matches += matches
            m = metrics.get(doc, {})
            sections = int(m["sections"]) if m.get("sections") else None
            caps = int(m["caps_sections"]) if m.get("caps_sections") else None
            group = health(sections, caps)
            outcomes: dict[tuple[int, str], dict] = {}

            for regime in scorable:
                cats = [ev.category_key(c) for c in regime.cuad_categories]
                r_matches = [x for x in matches if x.category in cats]
                r_gold = [g for x in r_matches for g in x.clauses]
                for k in ks:
                    cands, _ = regimes.candidates(
                        driver, doc, vectors[regime.name], k, args.max_candidates)
                    sent = [ev.Sent(c.uid, ev.is_seed(c.why), c.excerpt) for c in cands]
                    cand_uids = {c.uid for c in cands}
                    s = ev.score(r_gold, sent)
                    n_spans, n_span_hits = ev.span_hits(r_matches, cand_uids)
                    outcomes[(k, regime.name)] = s.outcome
                    base = {"doc_id": doc, "regime": regime.name, "k": k}
                    cand_chars = sum(len(c.excerpt) for c in cands)
                    rows.append({
                        **base,
                        "gold_spans": len(r_matches),
                        "unmatched_spans": sum(not x.matched for x in r_matches),
                        "gold_clauses": s.gold_clauses,
                        "candidate_clauses": len(cands),
                        "hits": s.hits, "hits_seed": s.hits_seed, "hits_hop": s.hits_hop,
                        "hits_excerpt": s.hits_excerpt,
                        "recall": None if s.recall is None else round(s.recall, 4),
                        "recall_excerpt": (None if s.recall_excerpt is None
                                           else round(s.recall_excerpt, 4)),
                        "spans_matched": n_spans, "span_hits": n_span_hits,
                        "span_recall": ratio(n_span_hits, n_spans),
                        "candidate_chars": cand_chars, "contract_chars": total_chars,
                        "sections": sections, "caps_sections": caps, "health": group,
                    })
                    for cat in cats:
                        c_gold = [g for g in r_gold if g.category == cat]
                        cs = ev.score(c_gold, sent)
                        c_spans, c_span_hits = ev.span_hits(
                            [x for x in r_matches if x.category == cat], cand_uids)
                        cat_rows.append({
                            **base, "category": cat, "gold_clauses": cs.gold_clauses,
                            "hits": cs.hits, "hits_seed": cs.hits_seed,
                            "hits_hop": cs.hits_hop, "hits_excerpt": cs.hits_excerpt,
                            "spans_matched": c_spans, "span_hits": c_span_hits,
                            "candidate_clauses": len(cands), "candidate_chars": cand_chars,
                            "contract_chars": total_chars, "health": group,
                        })
            for x in matches:
                for g in x.clauses:
                    out = {"doc_id": doc, "regime": mapping.get(g.category, ""),
                           "category": g.category, "span_idx": g.span_idx,
                           "section": g.section, "uid": g.uid, "match": g.method,
                           "health": group}
                    for k in ks:
                        o = outcomes.get((k, out["regime"]), {}).get(g.uid, ("", False))
                        out[f"k{k}"] = o[0]
                        out[f"k{k}_excerpt"] = o[1]
                    gold_out.append(out)
            first_k = [r for r in rows if r["doc_id"] == doc and r["k"] == ks[0]]
            print(f"{doc[:40]:<40} spans={len(matches):>3} "
                  f"unmatched={sum(not x.matched for x in matches)} "
                  f"gold={sum(r['gold_clauses'] for r in first_k):>3} "
                  f"hits@k{ks[0]}={sum(r['hits'] for r in first_k):>3}", flush=True)

    if not rows:
        print("nothing evaluated")
        return 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    for path, data in ((OUT_CSV, rows), (OUT_GOLD, gold_out)):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]) if data else ["doc_id"])
            writer.writeheader()
            writer.writerows(data)

    matched = [x for x in span_matches if x.matched]
    status_counts = {s: sum(x.status == s for x in span_matches)
                     for s in (ev.CONTAINED, ev.OVERLAP, ev.NOT_IN_SOURCE, ev.NO_CLAUSE,
                               ev.NO_SOURCE, ev.EMPTY)}
    evaluated = {r["doc_id"] for r in rows}
    summary = {
        "config": {
            "k": ks, "max_candidates": args.max_candidates,
            "candidate_chars": regimes.CANDIDATE_CHARS, "contracts": len({r["doc_id"] for r in rows}),
            "embed_model": os.environ.get("EMBED_MODEL"), "high_caps_threshold": HIGH_CAPS,
        },
        "categories": {**categories, "mapping": mapping},
        "spans": {
            "total": len(span_matches),
            "matched": len(matched),
            "match_rate": ratio(len(matched), len(span_matches)),
            "by_status": status_counts,
            "multi_clause": sum(len(x.clauses) > 1 for x in span_matches),
            "unmatched": [{"doc_id": x.doc_id, "category": x.category, "span_idx": x.span_idx,
                           "reason": x.status} for x in span_matches if not x.matched],
        },
        "gold_contracts_not_evaluated": [g["doc_id"] for g in gold_rows
                                         if g["doc_id"] not in evaluated],
        "present_disagreements": ev.present_disagreements(
            [g for g in gold_rows if g["doc_id"] in evaluated]),
        "by_k": {},
    }
    for k in ks:
        rk = [r for r in rows if r["k"] == k]
        ck = [r for r in cat_rows if r["k"] == k]
        summary["by_k"][str(k)] = {
            "all": summarise([{**r, "all": "all"} for r in rk], "all", per_contract=True)["all"],
            "by_regime": summarise(rk, "regime"),
            "by_category": summarise(ck, "category"),
            "by_health": summarise(rk, "health", per_contract=True),
        }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    print(f"\nspans matched: {len(matched)}/{len(span_matches)} {status_counts}")
    for k in ks:
        print(f"k={k}")
        for name, s in summary["by_k"][str(k)]["by_regime"].items():
            print(f"  {name:<26} micro={s['micro']} macro={s['macro']} "
                  f"excerpt micro={s['micro_excerpt']} macro={s['macro_excerpt']} "
                  f"span={s['span_recall']} "
                  f"cand={s['mean_candidates']} sent={s['chars_sent_ratio']}")
    print(f"-> {OUT_CSV}, {OUT_SUMMARY}, {OUT_GOLD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
