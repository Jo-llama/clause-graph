# 0005 Evaluation against CUAD gold: clause-level span matching, candidate recall as upper limit

Status: Accepted, 2026-11-20 (step 7a). Refines the "step-7 eval" named in ADR 0004; ADR 0004
itself is unchanged.

## Context
ADR 0004 sends the LLM only candidate clauses (vector top-k chunks + one graph hop, at most
25, each excerpt cut at 4,000 chars). A regime clause that is not a candidate can never be
found, so candidate selection bounds LLM recall. That bound can be measured for free: no
LLM is involved. The CUAD gold labels from clause-harness (`golden.jsonl`) give, per
contract, 8 categories with verbatim evidence spans (489 spans, no char offsets) and a
`present` flag. Our graph has clauses, not offsets: `Clause.text` is a slice of the source
and `Clause.position` gives document order. The regimes in `regimes.json` already name their
CUAD categories (Confidentiality has none).

## Decision
- **Gold file**: `scripts/get_contracts.py` also writes `data/gold.jsonl` (`doc_id`, `labels`
  unchanged) from the same download. Like `contracts.jsonl` it is git-ignored, and
  `contracts.jsonl` is only rewritten when its content would change.
- **Category -> regime** is inverted from `regimes.json` (`evaluate.category_to_regime`).
  Names are normalised to the gold keys (`"Cap On Liability"` -> `cap_on_liability`). A
  category in two regimes is an error; a gold category without a regime is reported as
  `unmapped`; a regime without categories (Confidentiality) is reported as `no_gold`.
- **Gold is defined by the evidence spans, not by `present`** (user decision C). Rows where
  `present` and having evidence disagree are listed (doc_id + category, no text) in the
  summary JSON and in `docs/known-issues.md`.
- **Span -> clause matching is exact after whitespace collapse** (`chunking.collapse`): no
  fuzzy matching, no case folding.
  1. A span maps to every clause whose collapsed text contains it (`contained`).
  2. Otherwise it is placed in the collapsed source and mapped to every clause it overlaps
     (`overlap`). Clause positions come from finding each collapsed clause text in the
     collapsed source, in document order.
  3. Otherwise it is unmatched: `not_in_source` (not in the contract text), `no_clause` (in
     the text but in no located clause) or `no_source` (the contract is loaded but has no
     row in `contracts.jsonl`; such contracts are still evaluated). Unmatched spans are
     counted and listed by doc_id, category and span index, never dropped. Gold contracts
     that are not evaluated (not loaded, or filtered out) are listed in
     `gold_contracts_not_evaluated`.

  Every (span, clause) pair keeps its provenance: doc_id, section, clause uid, category,
  span index (`runs/eval_gold_clauses.csv`).
- **Recall is measured at clause level per regime**: gold clauses of a (contract, regime) are
  the union over its CUAD categories. Candidate recall = gold clauses that are candidates /
  gold clauses. Two aggregates:
  - **micro** = sum of hits / sum of gold clauses over all scored (contract, regime) pairs
    of the group (pooled; contracts with many gold clauses weigh more).
  - **macro** = mean over contracts of each contract's recall. For `by_regime` and
    `by_category` a contract has one row per group, so this is the mean over rows. For
    `all` and `by_health` each contract's regimes are summed first (hits and gold clauses),
    then the per-contract recalls are averaged (`contracts_scored` contracts), so a
    contract counts once, not once per regime.
  Pairs (and contracts) without gold clauses are left out of recall and counted separately
  (`pairs_no_gold`). Each hit records whether the
  clause was a vector seed or a graph-hop candidate (from `Candidate.why`).
- **Excerpt-level recall as a secondary measure** (user decision B): a gold clause is an
  excerpt hit when it is a candidate and at least one of its gold pieces (the part of a
  span lying in that clause) is verbatim, whitespace-collapsed, in the excerpt actually
  sent. "At least one" because one visible piece of evidence is enough for the LLM to
  assign the clause. Same micro/macro definitions.
- **Span-level recall as a second secondary measure**: a matched gold span is a hit when
  ANY of the clauses holding it is a candidate (`span_hits` / `spans_matched` =
  `span_recall`, micro only, per row, regime, category, health group and overall). It
  differs from clause level where a span lies in several clauses (counted
  in `spans.multi_clause`) and where several spans share one clause.
- **Candidate recall comes first** (step 7a): it is the free upper limit on LLM recall, and
  7b (scoring PART_OF edges) and 7c (local model) reuse the same matching and are compared
  against this baseline. `scripts/eval_candidates.py` makes no LLM call (it does not import
  `llm.py`, which a test enforces, and removes `ANTHROPIC_API_KEY` from its environment) and
  only reads Neo4j. `--k` is repeatable for a sweep; the sweep is reported, not adopted.
- **Contract health**: every row carries `sections`, `caps_sections` and a group
  (`sections_0`, `high_caps` = `caps_sections >= 10`, `ok`) from
  `runs/extraction_metrics.csv`, so misses caused by known extraction issues are visible.
- **Results are committed** (user decision A): `runs/eval_candidates.csv` and
  `runs/eval_candidates_summary.json` are un-ignored. They contain counts and ids only, no
  contract or gold text, and no timestamp, so a re-run on the same graph gives no diff.

## Alternatives considered
- **Char-offset matching**: needs offsets on Clause nodes (a reload) and on the gold spans
  (CUAD's original answer_start, not in golden.jsonl). Text containment gives the same
  answer here: on the loaded graph all 489 of 489 spans of the 75 contracts are contained
  in a clause (`runs/eval_candidates_summary.json`, `spans.by_status`; 0 overlap,
  0 unmatched, 0 gold contracts not evaluated).
- **Fuzzy matching (difflib, token overlap)**: would hide format drift between the gold and
  the graph, which is exactly what the eval must expose. Exact matching with an explicit
  unmatched list makes such drift a number.
- **Span-level or token-level recall as the main measure**: closer to CUAD's own metric,
  but the LLM assigns clauses, so clause level is what PART_OF edges are scored on in 7b.
  Span level is kept as a secondary number; token level is not measured.
- **Macro over (contract, regime) pairs** for the all-contracts and health aggregates:
  weighs a contract by how many regimes have gold in it, which is not a property of the
  contract's extraction quality.
- **Using `present` as gold**: `present` without evidence cannot be mapped to any clause,
  and evidence with `present = false` is still a verbatim span a lawyer marked.

## Consequences
- The step-6 recall limit is a measured number per regime and per CUAD category, before any
  money is spent on LLM calls. First run (`runs/eval_candidates_summary.json`, `by_k.15`,
  75 contracts, max 25 candidates): micro clause / excerpt / span recall overall
  0.85 / 0.77 / 0.87 (per-contract macro clause 0.92); Liability 0.81 / 0.77 / 0.84,
  Termination 0.87 / 0.74 / 0.87, Assignment & Control 0.92 / 0.84 / 0.92, Restrictive
  Covenants 0.69 / 0.56 / 0.78, Governing Law & Disputes 0.96 / 0.89 / 0.96. Details,
  the k sweep and the health split are in `docs/known-issues.md` section 8.
- A span contained in N clauses adds N gold clauses (each must be a candidate for full
  clause recall); span-level recall counts it once. On the current graph this affects
  1 span (`spans.multi_clause`).
- Clause-level recall flatters contracts with few, long clauses (sections=0, exhibits
  swallowed by one section): the gold clause is easy to hit because it is most of the
  contract. Excerpt recall and the health grouping show this.
- A span overlapping a boundary makes both clauses gold, even if only a heading lies in the
  second one. On the current extraction this does not occur (no `overlap` matches), but a
  change in section splitting can change gold clause counts; compare span match status
  across runs.
- Gold covers 5 of 6 regimes and only the CUAD categories named in `regimes.json`; a regime
  clause outside those categories (e.g. an indemnity procedure) is neither gold nor
  penalised. Recall here is about the categories, precision is not measured (7b/7d).
