"""Embed (:Chunk) nodes locally via Ollama and build the Neo4j vector index.

Usage: uv run scripts/embed_chunks.py [--batch-size N] [--limit N] [--doc PREFIX]

Resumable and stale-proof: a chunk is embedded when it has no embedding, when its
embedding came from another EMBED_MODEL, or when its embed_hash (hash of the text
that was embedded) differs from its current text_hash. An interrupted run continues
where it stopped, a model swap re-embeds everything, and a reload that changed a
chunk's text forces a re-embed. A chunk the model rejects (e.g. too long) is
retried alone, recorded as failed and left for the next run; a connection error
stops the run. Each run is appended to runs/embedding_metrics.json.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from clause_graph import chunking, embeddings
from clause_graph.embeddings import EmbeddingError, EmbeddingInputError

ROOT = Path(__file__).resolve().parent.parent
METRICS_JSON = ROOT / "runs" / "embedding_metrics.json"
INDEX = "chunk_embedding"

TODO = """
MATCH (k:Chunk)
WHERE (k.embedding IS NULL OR k.embed_model IS NULL OR k.embed_model <> $model
       OR k.embed_hash IS NULL OR k.embed_hash <> k.text_hash)
  AND k.doc_id STARTS WITH $doc
RETURN k.uid AS uid, k.text AS text, k.text_hash AS text_hash, k.tokens_est AS tokens_est
ORDER BY k.doc_id, k.uid
"""
WRITE = """
UNWIND $rows AS row
MATCH (k:Chunk {uid: row.uid})
WHERE k.text_hash = row.text_hash  // unchanged since read (a reload may run meanwhile)
CALL db.create.setNodeVectorProperty(k, 'embedding', row.vector)
SET k.embed_model = $model, k.embed_hash = row.text_hash
"""
INDEX_DIMS = """
SHOW VECTOR INDEXES YIELD name, options WHERE name = $name
RETURN options.indexConfig['vector.dimensions'] AS dims
"""
CREATE_INDEX = """
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (k:Chunk) ON k.embedding
OPTIONS {indexConfig: {`vector.dimensions`: $dims, `vector.similarity_function`: 'cosine'}}
"""


def ensure_index(session, dims: int) -> None:
    record = session.run(INDEX_DIMS, name=INDEX).single()
    if record is not None and record["dims"] != dims:
        sys.exit(
            f"Vector index {INDEX} has {record['dims']} dimensions but the model returns {dims}. "
            f"Drop it first: DROP INDEX {INDEX}"
        )
    session.run(CREATE_INDEX, dims=dims).consume()
    session.run("CALL db.awaitIndexes(300)").consume()


def append_metrics(run: dict) -> None:
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    data = {"runs": []}
    if METRICS_JSON.exists():
        data = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    data["runs"].append(run)
    METRICS_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="embed at most N chunks")
    parser.add_argument("--doc", default="", help="only contracts whose doc_id starts with this")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        sys.exit("NEO4J_PASSWORD is not set (expected in .env)")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    cfg = embeddings.config()

    try:
        dims = len(embeddings.embed_query("dimension probe", cfg))
    except EmbeddingError as e:
        sys.exit(str(e))

    with GraphDatabase.driver(uri, auth=("neo4j", password)) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            ensure_index(session, dims)
            todo = session.run(TODO, model=cfg.model, doc=args.doc).data()
            total_chunks = session.run("MATCH (k:Chunk) RETURN count(k) AS n").single()["n"]
            if args.limit is not None:
                todo = todo[: args.limit]
            print(
                f"model={cfg.model} dims={dims} at {cfg.url}: {len(todo)} of {total_chunks} "
                "chunks to embed"
            )

            started = time.perf_counter()
            done = tokens = tokens_est = 0
            failed: list[dict] = []
            stop_error = None

            def write(rows: list[dict], vectors: list[list[float]]) -> None:
                session.run(
                    WRITE,
                    model=cfg.model,
                    rows=[
                        {"uid": r["uid"], "text_hash": r["text_hash"], "vector": v}
                        for r, v in zip(rows, vectors)
                    ],
                ).consume()

            for i in range(0, len(todo), args.batch_size):
                batch = todo[i : i + args.batch_size]
                try:
                    try:
                        result = embeddings.embed([r["text"] for r in batch], cfg)
                        write(batch, result.vectors)
                        done += len(batch)
                        tokens += result.tokens
                        tokens_est += sum(r["tokens_est"] for r in batch)
                    except EmbeddingInputError:
                        for r in batch:  # isolate the chunk(s) the model rejects
                            try:
                                one = embeddings.embed([r["text"]], cfg)
                            except EmbeddingInputError as e:
                                failed.append({"uid": r["uid"], "error": str(e)[:200]})
                                print(f"\nskipped {r['uid']}: {e}", file=sys.stderr)
                                continue
                            write([r], one.vectors)
                            done += 1
                            tokens += one.tokens
                            tokens_est += r["tokens_est"]
                except EmbeddingError as e:
                    stop_error = str(e)
                    print(f"\nstopped after {done} chunks: {e}", file=sys.stderr)
                    break
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0
                eta = (len(todo) - done - len(failed)) / rate if rate else 0
                print(
                    f"{done}/{len(todo)} chunks  {rate:.2f} chunks/s  "
                    f"{tokens / elapsed:.0f} tokens/s  eta {eta / 60:.0f} min  "
                    f"failed {len(failed)}",
                    flush=True,
                )
            elapsed = time.perf_counter() - started
            print()
            embedded_total = session.run(
                "MATCH (k:Chunk) WHERE k.embed_model = $model AND k.embed_hash = k.text_hash "
                "RETURN count(k) AS n",
                model=cfg.model,
            ).single()["n"]
            versions = {
                f"v{r['v']}": r["n"]
                for r in session.run(
                    "MATCH (k:Chunk) RETURN coalesce(k.chunker_version, 1) AS v, count(k) AS n "
                    "ORDER BY v"
                )
            }

    run = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": cfg.model,
        "dims": dims,
        "chunk_max_tokens": chunking.MAX_TOKENS,
        "chunk_overlap_tokens": chunking.OVERLAP_TOKENS,
        "chunker_versions": versions,  # chunks in the graph per chunker version (ADR 0003)
        "batch_size": args.batch_size,
        "doc_filter": args.doc or None,
        "chunks_total": total_chunks,
        "chunks_planned": len(todo),
        "chunks_embedded": done,
        "chunks_failed": failed,
        "chunks_with_embedding": embedded_total,
        "stopped_by": stop_error,
        "total_time_s": round(elapsed, 1),
        "chunks_per_hour": round(done / elapsed * 3600) if elapsed else None,
        "tokens": tokens,
        "tokens_per_s": round(tokens / elapsed, 1) if elapsed else None,
        "tokens_est": tokens_est,
        # >1 means the chars/4 estimate in chunking.py undercounts real model tokens.
        "tokens_per_estimated_token": round(tokens / tokens_est, 3) if tokens_est else None,
    }
    append_metrics(run)
    print(json.dumps(run, indent=2))
    return 0 if done == len(todo) and stop_error is None else 1


if __name__ == "__main__":
    sys.exit(main())
