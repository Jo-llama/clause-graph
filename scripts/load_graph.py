"""Extract contracts deterministically and load them into Neo4j.

Usage: uv run scripts/load_graph.py [--limit N] [--reset] [--rechunk DOC_PREFIX]
                                   [--metrics-out PATH]

Schema (see CLAUDE.md):
  (:Contract {doc_id, <metrics>})-[:HAS_CLAUSE]->(:Clause {uid, doc_id, section, text})
  (:Clause)-[:DEFINES]->(:DefinedTerm {uid, doc_id, name})
  (:Clause)-[:USES_TERM]->(:DefinedTerm)
  (:Clause)-[:REFERENCES {number}]->(:Clause)
  (:Clause)-[:SUBSECTION_OF]->(:Clause)
  (:Chunk {uid, doc_id, section, idx, text, text_hash, chunker_version, start, end,
            tokens_est, tokens})-[:CHUNK_OF]->(:Clause)
Chunk embeddings are added afterwards by scripts/embed_chunks.py. Reloading a
contract keeps the embedding of every chunk whose uid and text_hash are unchanged;
embed_chunks.py re-embeds any chunk whose embed_hash differs from its text_hash, so
a reload can never leave a stale embedding (--reset deletes everything, embeddings too).

Chunker version (ADR 0003): a contract that already has chunks is re-chunked with the
version it has (missing = v1), so a reload keeps its embeddings; new contracts, --reset
and --rechunk DOC_PREFIX use chunking.CURRENT_CHUNKER (v2: real bge-m3 tokens).
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from clause_graph.chunking import (
    CHUNKER_V1,
    CHUNKER_V2,
    CURRENT_CHUNKER,
    chunk_clause,
    chunker_version,
    estimate_tokens,
    text_hash,
)
from clause_graph.embeddings import EmbeddingError, config, count_tokens
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
    "chunks",
    "chunks_skipped",
    "chunker_version",
    "unresolved_refs",
    "unresolved_share",
]

CONSTRAINTS = [
    "CREATE CONSTRAINT contract_doc_id IF NOT EXISTS FOR (n:Contract) REQUIRE n.doc_id IS UNIQUE",
    "CREATE CONSTRAINT clause_uid IF NOT EXISTS FOR (n:Clause) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT term_uid IF NOT EXISTS FOR (n:DefinedTerm) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT chunk_uid IF NOT EXISTS FOR (n:Chunk) REQUIRE n.uid IS UNIQUE",
]

# Re-loading a contract replaces its subgraph, so runs without --reset stay idempotent.
DELETE_CONTRACT = """
MATCH (n) WHERE (n:Clause OR n:DefinedTerm OR n:Chunk) AND n.doc_id = $doc_id
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
MERGE_CHUNKS = """
UNWIND $rows AS row
MATCH (c:Clause {uid: row.clause})
MERGE (k:Chunk {uid: row.uid})
SET k.doc_id = row.doc_id, k.section = row.section, k.idx = row.idx, k.text = row.text,
    k.text_hash = row.text_hash, k.chunker_version = row.chunker_version,
    k.start = row.start, k.end = row.end, k.tokens_est = row.tokens_est, k.tokens = row.tokens
MERGE (k)-[:CHUNK_OF]->(c)
"""
CHUNKER_VERSIONS = """
MATCH (k:Chunk)
RETURN k.doc_id AS doc_id, max(coalesce(k.chunker_version, 1)) AS version
"""
SAVED_EMBEDDINGS = """
MATCH (k:Chunk {doc_id: $doc_id}) WHERE k.embedding IS NOT NULL
RETURN k.uid AS uid, k.text AS text, k.embedding AS embedding, k.embed_model AS model
"""
RESTORE_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (k:Chunk {uid: row.uid})
CALL db.create.setNodeVectorProperty(k, 'embedding', row.embedding)
SET k.embed_model = row.model, k.embed_hash = k.text_hash  // text verified unchanged
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


def chunk_rows(g: ContractGraph, version: int = CHUNKER_V1) -> tuple[list[dict], int]:
    """Chunk rows for every clause, and how many near-empty chunks were skipped."""
    if version == CHUNKER_V2:
        cfg = config()
        count = lambda text: count_tokens(text, cfg)  # noqa: E731
    else:
        count = estimate_tokens
    rows, skipped = [], 0
    for c in g.clauses:
        chunks, n_skipped = chunk_clause(c.text, count=count)
        skipped += n_skipped
        rows += [
            {
                "uid": f"{c.uid}#{k.idx}",
                "clause": c.uid,
                "doc_id": c.doc_id,
                "section": c.section,
                "idx": k.idx,
                "text": k.text,
                "text_hash": text_hash(k.text),
                "chunker_version": version,
                "start": k.start,
                "end": k.end,
                "tokens_est": estimate_tokens(k.text),
                "tokens": count(k.text) if version == CHUNKER_V2 else None,
            }
            for k in chunks
        ]
    return rows, skipped


def write_contract(tx, g: ContractGraph, chunks: list[dict]) -> None:
    doc_id = g.doc_id
    unresolved_by_clause: dict[str, list[str]] = {}
    for uid, number in g.unresolved:
        unresolved_by_clause.setdefault(uid, []).append(number)

    saved = {r["uid"]: r for r in tx.run(SAVED_EMBEDDINGS, doc_id=doc_id).data()}
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
    tx.run(MERGE_CHUNKS, rows=chunks)
    unchanged = [
        saved[c["uid"]]
        for c in chunks
        if c["uid"] in saved and saved[c["uid"]]["text"] == c["text"]
    ]
    tx.run(RESTORE_EMBEDDINGS, rows=unchanged)
    tx.run(
        MERGE_SUBSECTIONS,
        rows=[{"child": c, "parent": p} for c, p in g.subsections],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="load only the first N contracts")
    parser.add_argument("--reset", action="store_true", help="delete the whole graph first")
    parser.add_argument(
        "--rechunk",
        metavar="DOC_PREFIX",
        help=f"re-chunk matching contracts with chunker v{CURRENT_CHUNKER} (needs re-embedding)",
    )
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
            existing = {r["doc_id"]: r["version"] for r in session.run(CHUNKER_VERSIONS)}
            chunks_of: dict[str, list[dict]] = {}
            try:
                for g in graphs:
                    version = chunker_version(g.doc_id, existing, args.rechunk)
                    chunks_of[g.doc_id], skipped = chunk_rows(g, version)
                    g.metrics["chunks"] = len(chunks_of[g.doc_id])
                    g.metrics["chunks_skipped"] = skipped
                    g.metrics["chunker_version"] = version
            except EmbeddingError as e:
                sys.exit(f"chunker v{CHUNKER_V2} needs the model tokenizer: {e}")
            for stmt in CONSTRAINTS:
                session.run(stmt).consume()
            for g in graphs:
                session.execute_write(write_contract, g, chunks_of[g.doc_id])
                m = g.metrics
                print(
                    f"{g.doc_id[:60]:<60} sections={m['sections']:>3} defs={m['definitions']:>3} "
                    f"refs={m['references']:>3} unresolved={m['unresolved_refs']:>3} chunks={m['chunks']:>3} "
                    f"v{m['chunker_version']}"
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
