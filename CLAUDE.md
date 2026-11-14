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

## Stack
- Python >= 3.11, uv (uv sync / uv run), pytest
- Neo4j 5.26 community in Docker (docker-compose.yml), credentials in .env (NEO4J_PASSWORD)
- Data: 75 CUAD contracts from clause-harness data/golden.jsonl (fields: doc_id, source_text)

## Graph schema
(:Contract {doc_id, <metrics>})-[:HAS_CLAUSE]->(:Clause {uid, doc_id, section, text})
(:Clause)-[:DEFINES]->(:DefinedTerm {uid, doc_id, name})
(:Clause)-[:USES_TERM]->(:DefinedTerm)
(:Clause)-[:REFERENCES]->(:Clause)
uid = "<doc_id>#<section>" for clauses, "<doc_id>#<term name>" for terms (terms are unique per contract).

## Roadmap (do one step at a time, do not jump ahead)
3 deterministic extraction + metrics | 4 MCP server | 5 semantic search + GraphRAG loop |
6 LLM extraction (obligations, roles, Regime) | 7 red-team + completeness eval |
8 A2A (harness as agent) | 9 n8n pipeline + HITL | 10 Azure deploy | 11 ADRs | 13 README

## Working style
- Small, reviewable changes; explain design decisions briefly in the PR/commit message.
- Never commit .env or data/contracts.jsonl.
