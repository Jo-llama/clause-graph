# Architecture Decision Records

One short file per design decision: Context, Decision, Alternatives considered,
Consequences, Status + date.

Rules:
- An accepted ADR is never edited. To change a decision, write a new ADR that
  supersedes it and set the old one's status to "Superseded by NNNN" (the only
  edit allowed on an accepted ADR).
- Every roadmap step that introduces a design decision adds an ADR in the same commit.
- Numbers are measured, not guessed, and say where they come from (runs/*.csv, runs/*.json).

| ADR | Title | Status |
|---|---|---|
| [0001](0001-knowledge-graph-not-vector-store-only.md) | Knowledge graph (Neo4j) instead of a vector store only | Accepted |
| [0002](0002-deterministic-extraction-before-llm.md) | Deterministic extraction (regex) before any LLM, with per-contract metrics | Accepted |
| [0003](0003-local-embeddings.md) | Local embeddings (Ollama + bge-m3); LLM APIs only for public CUAD data | Accepted |
| [0004](0004-llm-regime-extraction.md) | LLM regime extraction: structured output, deterministic validation, minimised candidates | Accepted |
