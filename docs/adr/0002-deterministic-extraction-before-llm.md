# 0002 Deterministic extraction (regex) before any LLM, with per-contract metrics

Status: Accepted, 2026-11-14 (recorded with step 5; decided in f7288d8, refined in cb99e9c)

## Context
Section numbers, quoted definitions ("Freight" means ...) and cross-references
("Section 9.0") have a predictable surface form. They make up the backbone of the graph
(ADR 0001), so they must be reproducible and traceable to source text. CUAD contracts are
EDGAR filings converted from PDF/HTML, so the format drifts between documents (spaced
numbering "5. 1.", roman ARTICLE headings, numbers separated from their text).

## Decision
Extract sections, defined terms, term uses, references and section hierarchy with pure
regex-based functions (src/clause_graph/extract.py), no LLM. Every load writes
per-contract metrics to runs/extraction_metrics.csv (sections, caps_sections,
preamble_chars, max_clause_chars, definitions, term_uses, hub_terms, references,
subsection_links, unresolved_refs, unresolved_share, chunks, chunks_skipped). Extractor
changes are measured before/after (scripts/compare_metrics.py; baseline
runs/extraction_metrics_v1.csv) and must not special-case individual contracts. LLMs
are reserved for what needs understanding (step 6: obligations, roles, regimes).

## Alternatives considered
- LLM extraction of everything: handles odd formats, but costs per document, is not
  deterministic, can hallucinate section numbers, and sends contract text to an API
  (see ADR 0003).
- A legal NLP/layout library: heavier dependency, still needs CUAD-specific tuning, and
  its failures are less visible than a regex plus a metric.

## Consequences
- Fast, free, reproducible; every node and edge points to doc_id + section + text.
- Format drift is visible instead of silent: e.g. sections=0 for 7 contracts, unresolved
  share 0.107, preamble/max-clause sizes flag under-splitting (step 3b numbers in
  cb99e9c).
- Some forms stay unhandled until a rule is added (docs/known-issues.md); two contracts
  (LECLANCHÉ, MJBIOTECH) cannot be fixed with surface regexes at all.
