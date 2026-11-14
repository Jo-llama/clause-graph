"""Extract contracts deterministically and load them into Neo4j.

Usage: uv run scripts/load_graph.py [--limit N] [--reset] [--metrics-out PATH]

Schema (see CLAUDE.md):
  (:Contract {doc_id, <metrics>})-[:HAS_CLAUSE]->(:Clause {uid, doc_id, section, text})
  (:Clause)-[:DEFINES]->(:DefinedTerm {uid, doc_id, name})
  (:Clause)-[:USES_TERM]->(:DefinedTerm)
  (:Clause)-[:REFERENCES {number}]->(:Clause)
  (:Clause)-[:SUBSECTION_OF]->(:Clause)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from clause_graph.extract import ContractGraph, extract, term_uid

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = ROOT / "data" / "contracts.jsonl"
METRICS_CSV = ROOT / "runs" / "extraction_metrics.csv"
METRIC_FIELDS = [
    "sections",
    "caps_sections",
    "preamble_chars",
    "max_clause_chars",
    "definitions",
    "term_uses",
    "hub_terms",
    "references",
    "subsection_links",
    "unresolved_refs",
    "unresolved_share",
]

CONSTRAINTS = [
    "CREATE CONSTRAINT contract_doc_id IF NOT EXISTS FOR (n:Contract) REQUIRE n.doc_id IS UNIQUE",
    "CREATE CONSTRAINT clause_uid IF NOT EXISTS FOR (n:Clause) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT term_uid IF NOT EXISTS FOR (n:DefinedTerm) REQUIRE n.uid IS UNIQUE",
]

# Re-loading a contract replaces its subgraph, so runs without --reset stay idempotent.
DELETE_CONTRACT = """
MATCH (n) WHERE (n:Clause OR n:DefinedTerm) AND n.doc_id = $doc_id
DETACH DELETE n
"""
MERGE_CONTRACT = """
MERGE (k:Contract {doc_id: $doc_id})
SET k += $metrics
"""
MERGE_CLAUSES = """
MATCH (k:Contract {doc_id: $doc_id})
UNWIND $rows AS row
MERGE (c:Clause {uid: row.uid})
SET c.doc_id = row.doc_id, c.section = row.section, c.text = row.text,
    c.position = row.position, c.unresolved_refs = row.unresolved_refs
MERGE (k)-[:HAS_CLAUSE]->(c)
"""
MERGE_TERMS = """
UNWIND $rows AS row
MERGE (t:DefinedTerm {uid: row.uid})
SET t.doc_id = row.doc_id, t.name = row.name, t.hub = row.hub
"""
MERGE_DEFINES = """
UNWIND $rows AS row
MATCH (c:Clause {uid: row.clause}), (t:DefinedTerm {uid: row.term})
MERGE (c)-[:DEFINES]->(t)
"""
MERGE_USES = """
UNWIND $rows AS row
MATCH (c:Clause {uid: row.clause}), (t:DefinedTerm {uid: row.term})
MERGE (c)-[:USES_TERM]->(t)
"""
MERGE_REFERENCES = """
UNWIND $rows AS row
MATCH (a:Clause {uid: row.src}), (b:Clause {uid: row.dst})
MERGE (a)-[r:REFERENCES]->(b)
SET r.number = row.number
"""
MERGE_SUBSECTIONS = """
UNWIND $rows AS row
MATCH (child:Clause {uid: row.child}), (parent:Clause {uid: row.parent})
MERGE (child)-[:SUBSECTION_OF]->(parent)
"""


def read_contracts(limit: int | None) -> list[dict]:
    if not CONTRACTS.exists():
        sys.exit(f"{CONTRACTS} not found - run: uv run scripts/get_contracts.py")
    rows = []
    with CONTRACTS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_contract(tx, g: ContractGraph) -> None:
    doc_id = g.doc_id
    unresolved_by_clause: dict[str, list[str]] = {}
    for uid, number in g.unresolved:
        unresolved_by_clause.setdefault(uid, []).append(number)

    tx.run(DELETE_CONTRACT, doc_id=doc_id)
    tx.run(MERGE_CONTRACT, doc_id=doc_id, metrics=g.metrics)
    tx.run(
        MERGE_CLAUSES,
        doc_id=doc_id,
        rows=[
            {
                "uid": c.uid,
                "doc_id": c.doc_id,
                "section": c.section,
                "text": c.text,
                "position": c.position,
                "unresolved_refs": unresolved_by_clause.get(c.uid, []),
            }
            for c in g.clauses
        ],
    )
    tx.run(
        MERGE_TERMS,
        rows=[
            {
                "uid": term_uid(doc_id, name),
                "doc_id": doc_id,
                "name": name,
                "hub": name in g.hub_terms,
            }
            for name in g.terms
        ],
    )
    tx.run(
        MERGE_DEFINES,
        rows=[{"clause": c, "term": term_uid(doc_id, t)} for c, t in g.defines],
    )
    tx.run(
        MERGE_USES,
        rows=[{"clause": c, "term": term_uid(doc_id, t)} for c, t in g.uses],
    )
    tx.run(
        MERGE_REFERENCES,
        rows=[{"src": a, "dst": b, "number": n} for a, b, n in g.references],
    )
    tx.run(
        MERGE_SUBSECTIONS,
        rows=[{"child": c, "parent": p} for c, p in g.subsections],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="load only the first N contracts")
    parser.add_argument("--reset", action="store_true", help="delete the whole graph first")
    parser.add_argument(
        "--metrics-out", type=Path, default=METRICS_CSV, help="where to write per-contract metrics"
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        sys.exit("NEO4J_PASSWORD is not set (expected in .env)")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")

    contracts = read_contracts(args.limit)
    graphs = [extract(row["doc_id"], row["text"]) for row in contracts]

    with GraphDatabase.driver(uri, auth=("neo4j", password)) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            if args.reset:
                session.run("MATCH (n) DETACH DELETE n").consume()
                session.run("DROP CONSTRAINT term_unique IF EXISTS").consume()
            for stmt in CONSTRAINTS:
                session.run(stmt).consume()
            for g in graphs:
                session.execute_write(write_contract, g)
                m = g.metrics
                print(
                    f"{g.doc_id[:60]:<60} sections={m['sections']:>3} defs={m['definitions']:>3} "
                    f"refs={m['references']:>3} unresolved={m['unresolved_refs']:>3}"
                )

    metrics_out = args.metrics_out.resolve()
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", *METRIC_FIELDS])
        writer.writeheader()
        for g in graphs:
            writer.writerow({"doc_id": g.doc_id, **g.metrics})

    print(f"loaded {len(graphs)} contracts into {uri}; metrics -> {metrics_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
