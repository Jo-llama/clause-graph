"""Step 6: assign clauses to contractual regimes with an LLM (ADR 0004).

Usage: uv run scripts/extract_regimes.py [--limit N] [--doc PREFIX ...] [--regime NAME ...]
                                         [--model ID] [--k 15] [--max-candidates 25]
                                         [--hop-reserve 0] [--dry-run]

Contracts: those matching --doc prefixes, else the first --limit in data/contracts.jsonl.
Per (contract, regime): candidates (vector + graph, no LLM) -> one structured-output LLM
call -> deterministic validation -> PART_OF / RELIES_ON edges. Every raw response is
cached in runs/llm_cache/<prompt_version>/<model>/, keyed by a hash of the exact input,
so re-runs never pay twice. --dry-run builds candidates only (no API, no writes).
Per-contract metrics -> runs/regime_metrics.csv; per-call log -> runs/regime_calls.jsonl.
Both record the candidate config the call was built with (k, max_candidates, hop_reserve and
its name, regimes.candidate_config_name). A --hop-reserve other than regimes.HOP_RESERVE
was not adopted (ADR 0006) and is allowed only with --dry-run, so no edges come from it.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from clause_graph import llm, regimes
from clause_graph.chunking import is_near_empty
from clause_graph.embeddings import EmbeddingError, embed_query
from clause_graph.llm import LLMError

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = ROOT / "data" / "contracts.jsonl"
METRICS_CSV = ROOT / "runs" / "regime_metrics.csv"
CALLS_LOG = ROOT / "runs" / "regime_calls.jsonl"
CORPUS_SIZE = 75


def select_docs(driver, prefixes: list[str], limit: int | None) -> list[str]:
    loaded = {r["d"] for r in driver.execute_query(
        "MATCH (k:Contract) RETURN k.doc_id AS d").records}
    order = [json.loads(line)["doc_id"] for line in CONTRACTS.open(encoding="utf-8") if line.strip()]
    order = [d for d in order if d in loaded]
    if prefixes:
        return [d for d in order if any(d.startswith(p) for p in prefixes)]
    return order[:limit] if limit else order


def clause_texts(driver, doc: str) -> dict[str, str]:
    return {
        r["section"]: r["text"] or ""
        for r in driver.execute_query(
            "MATCH (:Contract {doc_id: $doc})-[:HAS_CLAUSE]->(c:Clause) "
            "RETURN c.section AS section, c.text AS text", doc=doc
        ).records
    }


def call_llm(system: str, user: str, model: str, cfg: llm.Config, path: Path) -> tuple[dict, bool]:
    """Cached LLM call. Returns the cache record and whether it came from the cache."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    started = time.perf_counter()
    c = llm.complete_json(system, user, regimes.ASSIGNMENT_SCHEMA, model=model, cfg=cfg)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": cfg.provider,
        "model": model,
        "response_model": c.model,
        "effort": cfg.effort,
        "thinking": llm.THINKING.get(model, "adaptive (default)"),
        "stop_reason": c.stop_reason,
        "error": c.error,
        "input_tokens": c.input_tokens,
        "output_tokens": c.output_tokens,
        "latency_s": round(time.perf_counter() - started, 1),
        "raw_text": c.raw_text,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record, False


def contract_metrics(driver, doc: str, model: str, pv: str, texts: dict[str, str]) -> dict:
    """Counts from the graph for one contract, for this model + prompt version."""
    rows = driver.execute_query(
        """
        MATCH (c:Clause {doc_id: $doc})-[e:PART_OF]->(r:Regime)
        WHERE e.model = $model AND e.prompt_version = $pv
        RETURN r.name AS regime, c.section AS section, e.status AS status
        """, doc=doc, model=model, pv=pv,
    ).records
    per_regime = {r.name: set() for r in regimes.load_regimes()}
    in_regime: set[str] = set()
    review = 0
    for r in rows:
        per_regime[r["regime"]].add(r["section"])
        in_regime.add(r["section"])
        review += r["status"] == "needs_review"
    eligible = {s for s, t in texts.items() if s != "0" and not is_near_empty(t)}
    without = eligible - in_regime
    out = {f"members_{name.lower().replace(' & ', '_').replace(' ', '_')}": len(secs)
           for name, secs in per_regime.items()}
    out.update(
        needs_review=review,
        clauses_eligible=len(eligible),
        clauses_without_regime=len(without),
        share_without_regime=round(len(without) / len(eligible), 3) if eligible else None,
    )
    return out


def metrics_key(row: dict) -> tuple[str, str, str, str]:
    """(doc_id, model, prompt_version, candidate_config). Rows written before the column
    existed have candidate_config "" (unknown), so a new run never replaces them."""
    return (row["doc_id"], row["model"], row["prompt_version"],
            row.get("candidate_config") or "")


def write_metrics(rows: list[dict]) -> None:
    """Replace rows with the same metrics_key; keep the others."""
    existing = []
    if METRICS_CSV.exists():
        with METRICS_CSV.open(encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    keys = {metrics_key(r) for r in rows}
    merged = [r for r in existing if metrics_key(r) not in keys]
    merged += rows
    fields = list(rows[0])
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc", action="append", default=[], help="doc_id prefix (repeatable)")
    parser.add_argument("--regime", action="append", default=[], help="regime name (repeatable)")
    parser.add_argument("--model", default=None, help="default: LLM_MODEL from .env")
    parser.add_argument("--k", type=int, default=regimes.DEFAULT_K)
    parser.add_argument("--max-candidates", type=int, default=regimes.MAX_CANDIDATES)
    parser.add_argument("--hop-reserve", type=int, default=regimes.HOP_RESERVE,
                        help="candidate slots reserved for graph-hop clauses")
    parser.add_argument("--dry-run", action="store_true", help="candidates only: no API, no writes")
    args = parser.parse_args()
    if args.hop_reserve != regimes.HOP_RESERVE and not args.dry_run:
        sys.exit(f"--hop-reserve {args.hop_reserve} is not the adopted default "
                 f"({regimes.HOP_RESERVE}, ADR 0006): allowed only with --dry-run, "
                 "so no edges are written from it.")

    load_dotenv(ROOT / ".env")
    cfg = llm.config()
    model = args.model or cfg.model
    pv = regimes.PROMPT_VERSION
    prompt = regimes.load_prompt(pv)
    all_regimes = regimes.load_regimes()
    selected = [regimes.match_regime(n, all_regimes) for n in args.regime] or all_regimes
    if not args.dry_run and cfg.provider == "anthropic" and not llm.has_api_key():
        sys.exit("ANTHROPIC_API_KEY is not set in .env (or use --dry-run).")

    # Regime descriptions are embedded once, locally.
    try:
        vectors = {r.name: embed_query(r.description) for r in selected}
    except EmbeddingError as e:
        sys.exit(str(e))

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    driver = GraphDatabase.driver(uri, auth=("neo4j", os.environ["NEO4J_PASSWORD"]))
    docs = select_docs(driver, args.doc, args.limit)
    cand_cfg = {"k": args.k, "max_candidates": args.max_candidates,
                "hop_reserve": args.hop_reserve}
    cand_name = regimes.candidate_config_name(args.k, args.hop_reserve, args.max_candidates)
    print(f"model={model} prompt={pv} provider={cfg.provider} effort={cfg.effort} "
          f"contracts={len(docs)} regimes={len(selected)} k={args.k} "
          f"max_candidates={args.max_candidates} hop_reserve={args.hop_reserve}"
          f"{' DRY RUN' if args.dry_run else ''}")

    totals = {"calls": 0, "cached": 0, "errors": 0, "in": 0, "out": 0, "paid_in": 0, "paid_out": 0}
    metric_rows = []
    CALLS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with driver, CALLS_LOG.open("a", encoding="utf-8") as log:
        for doc in docs:
            texts = clause_texts(driver, doc)
            total_chars = sum(len(t) for t in texts.values())
            doc_stats = {"chars_sent": 0, "call_shares": [], "in": 0, "out": 0, "errors": 0,
                         "rejected": 0, "relies_on": 0}
            for regime in selected:
                cands, omitted = regimes.candidates(
                    driver, doc, vectors[regime.name], args.k, args.max_candidates,
                    args.hop_reserve)
                system, user = regimes.build_messages(regime, cands, prompt)
                chars_sent = sum(len(c.excerpt) for c in cands)
                share = chars_sent / total_chars if total_chars else 0
                doc_stats["chars_sent"] += chars_sent
                doc_stats["call_shares"].append(share)
                line = (f"{doc[:34]:<34} {regime.name[:22]:<22} cand={len(cands):>2} "
                        f"sent={chars_sent:>6}/{total_chars:<6} ({share:.0%})")
                if args.dry_run:
                    print(line)
                    continue

                digest = regimes.input_hash(system, user, model, cfg.effort)
                path = regimes.cache_path(doc, regime.name, model, digest, pv)
                try:
                    record, cached = call_llm(system, user, model, cfg, path)
                except LLMError as e:
                    print(f"{line}  LLM ERROR: {e}")
                    return 1
                data = None
                if not record["error"]:
                    data = json.loads(record["raw_text"])
                v = regimes.validate(
                    data, {c.section for c in cands}, texts, error=record["error"])
                if v.error:
                    doc_stats["errors"] += 1
                    totals["errors"] += 1
                else:
                    with driver.session() as s:
                        s.execute_write(regimes.write_regime, doc, regime.name, v, model, pv)

                n = v.counts()
                doc_stats["rejected"] += n["rejected"]
                doc_stats["relies_on"] += n["relies_on"]
                tin, tout = record["input_tokens"], record["output_tokens"]
                doc_stats["in"] += tin
                doc_stats["out"] += tout
                totals["calls"] += 1
                totals["in"] += tin
                totals["out"] += tout
                if cached:
                    totals["cached"] += 1
                else:
                    totals["paid_in"] += tin
                    totals["paid_out"] += tout
                cost = llm.cost_usd(model, tin, tout)
                print(
                    f"{line}  members={n['members']:>2} review={n['needs_review']:>2} "
                    f"rel={n['relies_on']} rej={n['rejected']} tok={tin}/{tout} "
                    f"${cost or 0:.3f}{' cached' if cached else ''}"
                    f"{'  ERROR ' + v.error if v.error else ''}",
                    flush=True,
                )
                log.write(json.dumps({
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "doc_id": doc, "regime": regime.name, "model": model, "prompt_version": pv,
                    **cand_cfg, "candidate_config": cand_name,
                    "cache": str(path.relative_to(ROOT)), "cached": cached,
                    "candidates": [c.section for c in cands], "omitted": omitted,
                    "chars_sent": chars_sent, "contract_chars": total_chars,
                    "input_tokens": tin, "output_tokens": tout, "cost_usd": cost,
                    "validation": regimes.validation_dict(v),
                }, ensure_ascii=False) + "\n")

            if args.dry_run:
                continue
            m = contract_metrics(driver, doc, model, pv, texts)
            shares = doc_stats["call_shares"]
            metric_rows.append({
                "doc_id": doc, "model": model, "prompt_version": pv,
                "candidate_config": cand_name, **m,
                "relies_on": doc_stats["relies_on"], "rejected": doc_stats["rejected"],
                "call_errors": doc_stats["errors"],
                "contract_chars": total_chars, "chars_sent": doc_stats["chars_sent"],
                "share_sent_per_call": round(sum(shares) / len(shares), 3) if shares else None,
                "input_tokens": doc_stats["in"], "output_tokens": doc_stats["out"],
                "cost_usd": round(llm.cost_usd(model, doc_stats["in"], doc_stats["out"]) or 0, 4),
            })

    if args.dry_run or not metric_rows:
        return 0
    write_metrics(metric_rows)

    cost_all = llm.cost_usd(model, totals["in"], totals["out"])
    cost_paid = llm.cost_usd(model, totals["paid_in"], totals["paid_out"])
    print()
    print(f"calls={totals['calls']} cached={totals['cached']} call_errors={totals['errors']}")
    print(f"tokens in={totals['in']:,} out={totals['out']:,}")
    if cost_all is not None:
        per_contract = cost_all / len(docs)
        print(f"cost for these {len(docs)} contracts: ${cost_all:.2f} "
              f"(paid this run: ${cost_paid:.2f}); prices {model} as of {llm.PRICES_AS_OF}")
        print(f"extrapolated to {CORPUS_SIZE} contracts: ${per_contract * CORPUS_SIZE:.2f} "
              f"(${per_contract:.3f}/contract)")
    print(f"metrics -> {METRICS_CSV.relative_to(ROOT)}; calls -> {CALLS_LOG.relative_to(ROOT)}")
    return 1 if totals["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
