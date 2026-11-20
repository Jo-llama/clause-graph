# Known issues

Status after step 3b (section splitting), measured on all 75 CUAD contracts.
Figures come from `runs/extraction_metrics.csv` (baseline: `runs/extraction_metrics_v1.csv`);
compare with `uv run scripts/compare_metrics.py`.

## 1. Missed section numbering

`HEADING_RE` only recognises numbers followed by whitespace and a capitalised word.
Three surface forms are still missed:

| Form | Example | Contracts affected |
|---|---|---|
| No space after the number | `5.Dolphin will provide`, `Article 1.Amendment` (HUBEI, VEONEER) | 2 |
| Roman article numbers | `ARTICLE I  CERTAIN DEFINITIONS`, `ARTICLE IV ...` (SFG, Garrett, Verizon) | 11 |
| Quoted title after the number | `1.1 "AFFILIATE" means ...` (SFG) | 12 |

Contracts affected = contracts where the pattern occurs at least once; not all of them lose sections.

Side effects:
- Roman `ARTICLE` lines are currently caught only as ALL-CAPS headings (synthetic `H<n>`),
  so their numbered subsections have no parent: 207 dotted sections (e.g. Garrett `1.01`)
  have no `SUBSECTION_OF` link because `1` was never extracted. A further 76 parentless
  dotted sections are `X.0` article headings, which are top level by design.
- References like "Article IV" are not extracted at all (`REFERENCE_RE` is digits only).

## 2. Contracts with sections=0

7 contracts have no numbered section (`sections` counts numbered sections only, so this
stays visible even when ALL-CAPS headings split the text):

| Contract | H-sections | Longest clause (chars) | Cause |
|---|---|---|---|
| LECLANCHÉ S.A. Joint Development and Marketing | 0 | 20,297 | PDF conversion put the numbers on their own lines, away from the text (`1. 1.1.\n\n1.2.\n\n1.3.\n\nThis AGREEMENT ...`) |
| MJBIOTECH Joint Venture | 0 | 23,582 | Same: `1.\n\n\n\n\n\nThe firm name ...` |
| SFGFINANCIALCORP Software License and Maintenance | 13 | 25,668 | Roman articles + quoted titles (see 1) |
| HUBEIMINKANG Outsourcing | 0 | 11,322 | No space after number (`5.Dolphin`) |
| VEONEER Joint Venture | 1 | 8,211 | No space after number (`Article 1.Amendment`) |
| Scansource Distributor Agreement (10-K EX-10.38) | 0 | 13,193 | Unnumbered paragraphs, mixed-case headings |
| NELNET Joint Filing Agreement | 1 | 988 | One-page agreement without sections (correct) |

LECLANCHÉ and MJBIOTECH cannot be fixed by surface regexes; they need layout-aware
re-extraction from the source PDFs or an LLM pass. Do not special-case them.

## 3. Reference fallback to the first subsection

`resolve_reference` maps a cited number that is not a clause to its first subsection
("Section 12" -> `12.1`) when `12` itself was not extracted. 18 of 1,472 resolved references
land on a different section than the number as written (including `9` -> `9.0`).
`get_clause_context` now expands subsections of a referenced clause, but a reference that fell
back to `12.1` shows only `12.1` and its children, not the siblings `12.2`, `12.3`, ...
Better: resolve to a virtual parent (all `12.*` clauses), or link to the H-clause holding a
roman `ARTICLE XII` heading once roman numerals are supported.

## 4. Very long clauses (handled in step 5 by sub-chunking)

59 of 4,067 clauses exceed 8,000 chars and 17 exceed 20,000. The longest:

| Contract | Section | Chars |
|---|---|---|
| JOINTCORP Franchise Agreement | 20 | 148,612 |
| Verizon ABS Service Agreement | 10 | 107,401 |
| Buffalo Wild Wings Franchise Agreement | 7 | 52,065 |
| Blue Hills Bancorp Agency Agreement | 4 | 51,287 |
| Buffalo Wild Wings Franchise Agreement | 14 | 39,317 |

Most are a final section that swallowed exhibits, schedules or a second document. These exceed
embedding-model input limits and would produce one diluted vector each. Step 5 needs
sub-chunking (e.g. by paragraph, ~1-2k chars with overlap) where every chunk keeps the
provenance of its clause (doc_id, section, uid, char offsets). The graph stays at clause level.

Step 5: done. Clauses are split into `(:Chunk)` nodes of at most 1,500 tokens at sentence
boundaries (src/clause_graph/chunking.py, ADR 0003). The clauses themselves are still
oversized (exhibits inside a section), which matters for `get_clause_context` output size.

## 5. Minor: ALL-CAPS heading false positives

In a sample of 30 detected ALL-CAPS headings about 25 were real boundaries. The rest are mostly
signature-block company names (`FEDERATED INVESTMENT MANAGEMENT COMPANY` followed by `By:`)
and table rows. They create small extra clauses at the end of a contract and rarely cut a
clause mid-sentence; repeated page headers (>2 occurrences) are already filtered.

## 6. Scale: embedding on CPU

Measured on this machine (i7-1165G7, 4 cores, Ollama CPU, bge-m3), `runs/embedding_metrics.json`:

| Run | Settings | Chunks | Time | Chunks/h | Tokens/s |
|---|---|---|---|---|---|
| 2026-11-14 17:12 | default num_batch | 552 | 45 min | ~740 | ~50-60 |
| 2026-11-14 19:46 | num_batch = num_ctx = 4096 | 3,066 | 2 h 22 min | 1,296 | 87 |

The whole corpus (75 contracts, 3,755 chunks, 956k real tokens) takes ~3 h on CPU.
Chunks/h depends on chunk length; tokens/s is the stable unit.

Extrapolated: 10k contracts ≈ 500k chunks ≈ 127M tokens ≈ **400-660 h on CPU** (660 h at
~750 chunks/h, ~400 h at the tuned 87 tokens/s). Production therefore needs a GPU (EU-hosted
or on-prem, since contract text must not leave a controlled environment, ADR 0003) or
overnight/incremental batching: embed_chunks.py is resumable and only embeds chunks whose
`embed_hash` differs from `text_hash`, so a daily run only pays for new or changed contracts.

## 7. Embedding: failed chunks and oversized v1 chunks

- Failed chunks: **0** (final run, 3,755 of 3,755 chunks embedded; `chunks_failed: []`).
- 23 chunks made by chunker v1 (estimated tokens) exceed the 1,500-token target in real
  bge-m3 tokens; max 2,068 (BONTON Agency Agreement H6#2, a dense table). All are far below
  bge-m3's 8,192-token limit and were embedded whole (`truncate: false`, num_ctx 4096), so
  they are kept; re-embedding them would cost hours for no measured retrieval gain.
  Contracts loaded from now on use chunker v2 (real tokens). See ADR 0003.

## 8. Candidate recall (step 7a, ADR 0005)

How many CUAD gold evidence clauses the step-6 candidate selection (top-k chunks + one graph
hop, no LLM) reaches: the upper limit on LLM recall. Measure with
`uv run scripts/eval_candidates.py --k 15 --k 25 --k 40`; results in
`runs/eval_candidates.csv` and `runs/eval_candidates_summary.json`
(`by_k.<k>.by_regime`, `.by_category`, `.by_health`).

Numbers below: first live run, 2026-11-20, 75 contracts, max 25 candidates, 4,000-char
excerpts, bge-m3 (`runs/eval_candidates_summary.json`; two runs are byte-identical).
Definitions (ADR 0005): **micro** = pooled hits / pooled gold clauses; **macro** = mean
per-contract recall (for `all` and `by_health` each contract's regimes are summed first);
**clause** = gold clause is a candidate; **excerpt** = the excerpt sent contains gold
evidence; **span** = at least one clause holding the span is a candidate.

- **Span matching**: 489 of 489 gold spans are contained in a clause after whitespace
  collapse (0 overlap, 0 unmatched, 0 no_source); `spans.multi_clause` = 1 (one span lies
  in two clauses and so adds two gold clauses). `gold_contracts_not_evaluated` is empty.
- **Candidate recall at k = 15** (micro, 330 gold clauses / 489 spans):

  | Regime | clause | excerpt | span | gold clauses | hop hits |
  |---|---|---|---|---|---|
  | Liability | 0.81 | 0.77 | 0.84 | 77 | 7 |
  | Termination | 0.87 | 0.74 | 0.87 | 31 | 2 |
  | Assignment & Control | 0.92 | 0.84 | 0.92 | 89 | 1 |
  | Restrictive Covenants | 0.69 | 0.56 | 0.78 | 62 | 5 |
  | Governing Law & Disputes | 0.96 | 0.89 | 0.96 | 71 | 0 |
  | **All** | **0.85** | **0.77** | **0.87** | 330 | 15 |

  All, per-contract macro (71 contracts with gold): clause 0.92, excerpt 0.85. Mean
  candidates per (contract, regime) 18.6; 35 % of the contract characters are sent.
  Weakest categories: exclusivity 0.69 (excerpt 0.57), non_compete 0.75 (excerpt 0.54).
  Span recall is above clause recall for Restrictive Covenants (0.78 vs 0.69) and Liability
  (0.84 vs 0.81): the missed gold clauses hold fewer gold spans on average than the hit ones.
- **Effect of k** (micro clause recall at k = 15 / 25 / 40): Liability 0.81 / 0.77 / 0.75,
  Termination 0.87 / 0.90 / 0.94, Assignment & Control 0.92 / 0.93 / 0.94, Restrictive
  Covenants 0.69 / 0.71 / 0.73, Governing Law & Disputes 0.96 / 0.97 / 0.97; all
  0.85 / 0.86 / 0.86, chars sent 35 % / 39 % / 41 %. Liability falls with k because the
  25-candidate cap lets the extra vector seeds push out graph-hop candidates: graph-hop hits
  are 15 at k = 15 (Liability 7) and 0 for every regime at k = 25 and 40. Changing
  `MAX_CANDIDATES` or the default k is NOT done in 7a (out of scope); it is the obvious lever
  for later (e.g. reserve candidate slots for graph hops), to be judged with this eval.
- **By contract health** (k = 15, micro clause / excerpt, per-contract macro clause):

  | Group | contracts with gold | clause | excerpt | span | macro | misses |
  |---|---|---|---|---|---|---|
  | ok | 55 | 0.91 | 0.84 | 0.93 | 0.95 | 19 |
  | high_caps (caps_sections >= 10) | 10 | 0.65 | 0.63 | 0.70 | 0.74 | 29 |
  | sections_0 | 6 | 1.00 | 0.63 | 1.00 | 1.00 | 0 |

  high_caps contracts (section 1) cause 29 of the 48 misses with 83 of 330 gold clauses: many
  small sections fill the 25-candidate cap (mean 24.3 candidates vs 19.5 for ok), consistent
  with the expectation that parentless subsections defeat the graph hop (not yet verified
  per miss; `runs/eval_gold_clauses.csv` has the misses). sections_0 contracts (section 2)
  confirm the expectation: their few huge clauses are always candidates (clause recall 1.00), but the 4,000-char excerpt
  misses the span in 9 of 24 gold clauses.
- **`present` flag vs evidence** (gold is the evidence, not the flag; user decision C):
  5 (contract, category) rows disagree (`present_disagreements` in the summary JSON).

  | Contract | Category | present | spans |
  |---|---|---|---|
  | TELKOMSALTD_01_30_2003-EX-10-LICENCE AND MAINTENANCE AGREEMENT | non_compete | false | 1 |
  | LECLANCHÉ S.A. - JOINT DEVELOPMENT AND MARKETING AGREEMENT | change_of_control | false | 2 |
  | AULAMERICANUNITTRUST_04_24_2020-EX-99.8.77-SERVICING AGREEMENT | anti_assignment | true | 0 |
  | JOINTCORP_09_19_2014-EX-10.15-FRANCHISE AGREEMENT | exclusivity | false | 1 |
  | JOINTCORP_09_19_2014-EX-10.15-FRANCHISE AGREEMENT | termination_for_convenience | false | 1 |

  The four spans with `present = false` are scored as gold; AULAMERICAN anti_assignment has
  nothing to match and does not count.
- Confidentiality has no CUAD category ("no gold") and is not scored.
