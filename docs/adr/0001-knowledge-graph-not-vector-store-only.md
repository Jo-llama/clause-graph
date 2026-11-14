# 0001 Knowledge graph (Neo4j) instead of a vector store only

Status: Accepted, 2026-11-14 (recorded with step 5; decided in 5c49dd5 / f7288d8)

## Context
Contract questions are about regimes, not single clauses. "How is liability limited?" in
the DYNAMEX contract needs the cap (8.2), the indemnity it caps (8.1), the insurance
amounts it points to ("Section 9.0", whose content sits in 9.1), the freight liability
rule (6.0) and the definition of "Freight" (1.1). Several of these share little wording
with the question: 9.1 is about insurance, 1.1 is a definitions list. A similarity search
returns isolated chunks and silently misses the parts that are only connected by a
cross-reference or a defined term.

## Decision
Model each contract as a graph in Neo4j: Contract, Clause, DefinedTerm and (step 5) Chunk
nodes with HAS_CLAUSE, DEFINES, USES_TERM, REFERENCES, SUBSECTION_OF and CHUNK_OF edges.
Vector search (step 5) only finds entry points; the graph then assembles the regime
(`assemble_context`: seeds + one hop, each item with the reason it is included).

## Alternatives considered
- Vector store only (chunks + embeddings): simplest, but cannot follow "Section 9.0" or
  resolve a defined term; recall depends on wording overlap.
- Vector store + LLM to follow references at query time: works, but each answer costs LLM
  calls, is not reproducible and has no provenance for why a clause was pulled in.
- Relational tables for clauses/terms/references: possible, but multi-hop expansion
  (reference -> subsections -> definitions) becomes recursive SQL; Neo4j also has a
  native vector index, so one store serves both.

## Consequences
- Answers are explainable: every item carries provenance (doc_id, section, uid) and a
  "why" (e.g. "referenced by 8.2", "defines 'Freight' used by 6.0").
- Quality depends on extraction (ADR 0002): a missed section number or reference is a
  missing edge. Gaps are tracked in docs/known-issues.md.
- One more service to run (Neo4j in Docker) and a graph schema to keep in sync with
  CLAUDE.md.
