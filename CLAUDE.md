# clause-graph

GraphRAG for contracts. A Neo4j knowledge graph links clauses, defined terms and
cross-references so an agent can assemble a full contractual regime (e.g. liability:
cap clause + definition of "Losses" + carve-outs) instead of isolated clauses.
Portfolio project; companion to github.com/Jo-llama/clause-harness.

## Principles
- Deterministic code where the surface form is predictable (definitions, cross-references,
  section numbers). LLM only where understanding is needed (later: obligations, regimes).
- Every extraction step logs per-contract metrics, so silent failures (format drift) are visible.
- Provenance: every node/edge must be traceable to doc_id + section + source text.
- Embeddings run locally (Ollama, bge-m3). LLM API calls (harness, step 6) are a documented exception
  allowed only for public CUAD data; production would need an EU-hosted or local LLM.

## Stack
- Python >= 3.11, uv (uv sync / uv run), pytest
- Neo4j 5.26 community in Docker (docker-compose.yml), credentials in .env (NEO4J_PASSWORD)
- Ollama (docker-compose service, CPU, localhost only) for embeddings; .env: EMBED_MODEL=bge-m3,
  OLLAMA_URL=http://localhost:11434. Only src/clause_graph/embeddings.py talks to the model.
- LLM: only src/clause_graph/llm.py talks to it. .env: LLM_PROVIDER=anthropic|ollama, LLM_MODEL,
  COMPARE_MODEL, LLM_EFFORT, ANTHROPIC_API_KEY. Prompts are versioned files in prompts/;
  raw responses are cached in runs/llm_cache/ (never pay twice).
- Data: 75 CUAD contracts from clause-harness data/golden.jsonl (fields: doc_id, source_text)

## Graph schema
(:Contract {doc_id, <metrics>})-[:HAS_CLAUSE]->(:Clause {uid, doc_id, section, text})
(:Clause)-[:DEFINES]->(:DefinedTerm {uid, doc_id, name})
(:Clause)-[:USES_TERM]->(:DefinedTerm)
(:Clause)-[:REFERENCES]->(:Clause)
(:Clause)-[:SUBSECTION_OF]->(:Clause)   e.g. 9.1 -> 9 (or 9.0 if the article is only a heading)
uid = "<doc_id>#<section>" for clauses, "<doc_id>#<term name>" for terms (terms are unique per contract).
section is the number as written, "0" for the preamble, or H1, H2, ... for unnumbered ALL-CAPS
headings; Clause.position gives document order (use it to sort, not the section id).
(:Chunk {uid, doc_id, section, idx, text, text_hash, chunker_version, start, end, tokens_est, tokens,
          embedding, embed_model, embed_hash})-[:CHUNK_OF]->(:Clause)
  chunk uid = "<clause uid>#<idx>"; start/end are char offsets into the clause text;
  vector index chunk_embedding (cosine). An embedding is valid only if embed_hash = text_hash.
  chunker_version: 1 = estimated tokens (chars/4), 2 = real bge-m3 tokens (`tokens`); a reload keeps
  a contract's version, new contracts get v2 (ADR 0003).
  Pipeline: load_graph.py (keeps unchanged embeddings; --reset wipes all) -> embed_chunks.py
  -> extract_regimes.py.
(:Regime {uid: "<doc_id>#<name>", doc_id, name, runs})   names in src/clause_graph/regimes.json
(:Clause)-[:PART_OF {role, evidence, confidence, status, checks, method, model, prompt_version}]->(:Regime)
(:Clause)-[:RELIES_ON {evidence, status, checks, regime, method, model, prompt_version}]->(:Clause)
  LLM edges (step 6, ADR 0004): role in rule|cap|exclusion|definition|procedure|remedy|other;
  status accepted | needs_review (a deterministic check failed or confidence < 0.6).

## Roadmap (do one step at a time, do not jump ahead)
3 deterministic extraction + metrics | 4 MCP server | 5 semantic search + GraphRAG loop |
6 LLM extraction (obligations, roles, Regime) | 7 red-team + completeness eval |
8 A2A (harness as agent) | 9 n8n pipeline + HITL | 10 Azure deploy | 11 ADRs | 13 README

## Agents (.claude/agents/)
- pm (read-only spec + acceptance criteria) -> engineer (implements, writes ADR, no commits)
  -> tester (edits tests/ only) -> reviewer (read-only, ranked findings).
- /ship-step <task> runs that pipeline from the main session; the user approves the spec and commits.

## Working style
- Small, reviewable changes; explain design decisions briefly in the PR/commit message.
- Every step that introduces a design decision adds an ADR in docs/adr/ in the same commit
  (Context, Decision, Alternatives considered, Consequences, Status + date). Never edit an
  accepted ADR; supersede it with a new one.
- Never commit .env or data/contracts.jsonl.
