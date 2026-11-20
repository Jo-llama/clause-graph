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

### 8b. Candidate budget retune: reserved graph-hop slots (not adopted)

Follow-up to 8 (spec approved 2026-11-20, user decisions A-E). Question: within the unchanged
cap of 25 candidates, does reserving `r` slots for graph-hop candidates (and raising k) recover
the hop hits that extra vector seeds push out? Everything at $0 (no LLM call). Defaults are
**unchanged** (k = 15, `HOP_RESERVE = 0`). The evaluation method (fixed split, decision rule
fixed in advance, report-only caps, config names) is ADR 0006; no CLAUDE.md change.

Reproduce the grid (18 configs, ~3 min):
`uv run scripts/eval_candidates.py --k 15 --k 25 --k 40 --hop-reserve 0 --hop-reserve 5
--hop-reserve 10 --max-candidates 25 --max-candidates 30 --sweep-out runs/eval_candidates_sweep.json`.
It writes only `runs/eval_candidates_sweep.json` (committed: every config with all, by_regime,
by_health, by_split, by_split_regime, hits_by_hop_kind; the dev decision table and the rule) and
the git-ignored `runs/eval_candidates_sweep_rows.csv` / `runs/eval_gold_clauses_sweep.csv`
(per row / per gold clause, same columns as the 7a CSVs). It does not touch the 7a outputs.
Reproduce those with the 7a command `uv run scripts/eval_candidates.py --k 15 --k 25 --k 40`:
`runs/eval_candidates.csv` / `_summary.json` (committed) and `runs/eval_gold_clauses.csv`
(r = 0, cap 25) with the new columns; their numbers are identical to 7a (checked field by
field; micro clause 282/330 at k = 15). Config names are canonical (ADR 0006): `k{k}`, plus
`_r{r}` unless r = 0, plus `_c{cap}` unless cap = 25 (`k15`, `k25_r5`, `k15_c30`, ...); the
same config has the same name in the 7a summary, the sweep file and `regime_metrics.csv`.
Note: the summary key `by_k.<k>` named in section 8 is now `by_config.k<k>` (`by_k.15` ->
`by_config.k15`). Two runs of each command are byte-identical.

**Correction to the 7a crowding count.** From `runs/eval_gold_clauses.csv`: 10 gold clauses
are hop hits at k = 15 and misses at k = 25 (Liability 5, Restrictive Covenants 2, Termination
2, Assignment & Control 1); 3 of them come back as seeds at k = 40 (RC 1, Termination 2), so 7
are misses at both k = 25 and k = 40 (Liability 5, RC 1, A&C 1). The "7" was right for "both";
the breakdown summing to 10 is the k = 25 count. Of the 15 hop hits at k = 15, the other 8
become seed hits at k = 25 or 40 (5 already at k = 25). 11 gold clauses are gained as seeds at
k = 25. The 25 cap is reached in 156 / 217 / 225 of 375 pairs at k = 15 / 25 / 40.

**Mechanism** (`queries.reserve_hops`, `collect_context(..., hop_reserve)`): after the existing
(tier, seed rank, document order) ranking, keep seeds[:25 - r] and hops[:r] with
r = min(hop_reserve, hops), fill free slots in ranked order. A clause that is both a seed and a
hop target is a seed. r = 0 is the old cut (assemble_context / MCP unchanged, regression test).
At k = 15 a reserve <= 10 changes nothing (at most 15 seeds): k15 = k15_r5 = k15_r10 on the
live graph.

**Split** (`evaluate.split_of`): test if `int(sha256(doc_id)[:8], 16) % 4 == 0`: 59 dev / 16
test contracts; with gold: dev 57 contracts, 262 gold clauses; test 14 contracts, 68 gold
clauses.

**Dev decision** (rule fixed before the sweep, `evaluate.decide`): highest dev micro clause
recall with dev chars_sent_ratio <= 0.39, dev mean_candidates <= 20.0, no regime losing more
than 1 dev gold clause vs k15; ties to fewer chars; adopt only with >= 2 dev gold clauses
gain. Cap-30 rows are report-only and never eligible.

| config | dev hits | dev micro | gain | worst regime loss | dev chars sent | dev mean cand | eligible / fails |
|---|---|---|---|---|---|---|---|
| k15 (7a) | 222/262 | 0.8473 | +0 | 0 | 0.3467 | 18.57 | yes |
| k15_r5 | 222/262 | 0.8473 | +0 | 0 | 0.3467 | 18.57 | yes |
| k15_r10 | 222/262 | 0.8473 | +0 | 0 | 0.3467 | 18.57 | yes |
| k25 | 224/262 | 0.855 | +2 | 3 | 0.3956 | 19.84 | chars; regime loss Liability |
| k25_r5 | 225/262 | 0.8588 | +3 | 1 | 0.3956 | 19.84 | chars |
| k25_r10 | 219/262 | 0.8359 | -3 | 2 | 0.3879 | 19.84 | regime loss RC |
| k40 | 225/262 | 0.8588 | +3 | 3 | 0.4119 | 19.9 | chars; regime loss Liability |
| k40_r5 | 222/262 | 0.8473 | +0 | 1 | 0.4054 | 19.9 | chars |
| k40_r10 | 215/262 | 0.8206 | -7 | 3 | 0.3959 | 19.9 | chars; regime loss A&C, Liability, RC |

Result: only the three (identical) k = 15 configs are eligible; the best gains 0 dev gold
clauses, so **the defaults stay k = 15, r = 0, cap 25**. The closest contender, k25_r5, misses
the dev chars limit by 0.0056 and would also have failed the adoption targets on all 75
contracts (clause hits 284 < 286; Liability 0.7792 < 0.8052; chars sent 0.3923 > 0.39).

**Before / after** (all 75 contracts; "after" = chosen config = k15, so identical to 7a;
micro; `by_config.k15` in the summary):

| Regime | clause | excerpt | span | hop hits | mean cand | chars sent | dev hits | test hits |
|---|---|---|---|---|---|---|---|---|
| Liability | 0.8052 | 0.7662 | 0.8417 | 7 | 18.64 | 0.3602 | 50/62 | 12/15 |
| Termination | 0.871 | 0.7419 | 0.8684 | 2 | 18.56 | 0.3456 | 23/25 | 4/6 |
| Assignment & Control | 0.9213 | 0.8427 | 0.9205 | 1 | 18.57 | 0.3333 | 66/71 | 16/18 |
| Restrictive Covenants | 0.6935 | 0.5645 | 0.7753 | 5 | 18.56 | 0.3519 | 27/46 | 16/16 |
| Governing Law & Disputes | 0.9577 | 0.8873 | 0.9583 | 0 | 18.55 | 0.3497 | 56/58 | 12/13 |
| **All** | **0.8545** | **0.7727** | **0.8732** | 15 | 18.58 | 0.3481 | 222/262 (0.8473) | 60/68 (0.8824) |

Test split: the chosen config is the baseline, so test recall is unchanged (60/68). For
reference, k25_r5 has 59/68 on test.

**All configs, 75 contracts** (micro; cap-30 rows are REPORT-ONLY, never selectable; they show
the recall vs chars-sent trade-off of a larger cap for a later decision):

| config | clause hits | clause | excerpt | span | hop hits | mean cand | chars sent | high_caps chars | test hits |
|---|---|---|---|---|---|---|---|---|---|
| k15 / k15_r5 / k15_r10 | 282 | 0.8545 | 0.7727 | 0.8732 | 15 | 18.58 | 0.3481 | 0.2115 | 60 |
| k25 | 283 | 0.8576 | 0.7818 | 0.8691 | 0 | 19.78 | 0.3904 | 0.2437 | 59 |
| k25_r5 | 284 | 0.8606 | 0.7848 | 0.8793 | 6 | 19.78 | 0.3923 | 0.2469 | 59 |
| k25_r10 | 280 | 0.8485 | 0.7697 | 0.8712 | 11 | 19.78 | 0.3847 | 0.2377 | 61 |
| k40 | 285 | 0.8636 | 0.7879 | 0.8753 | 0 | 19.89 | 0.4117 | 0.262 | 60 |
| k40_r5 | 282 | 0.8545 | 0.7818 | 0.8814 | 5 | 19.89 | 0.4048 | 0.2609 | 60 |
| k40_r10 | 274 | 0.8303 | 0.7576 | 0.8507 | 8 | 19.89 | 0.3952 | 0.2565 | 59 |
| k15_c30 / k15_r5_c30 / k15_r10_c30 (report-only) | 286 | 0.8667 | 0.7848 | 0.8834 | 19 | 20.21 | 0.3678 | 0.238 | 61 |
| k25_c30 / k25_r5_c30 (report-only) | 289 | 0.8758 | 0.8 | 0.8896 | 6 | 22.43 | 0.4295 | 0.2883 | 61 |
| k25_r10_c30 (report-only) | 290 | 0.8788 | 0.8 | 0.8957 | 11 | 22.43 | 0.4245 | 0.283 | 62 |
| k40_c30 (report-only) | 291 | 0.8818 | 0.8061 | 0.8896 | 0 | 22.63 | 0.4517 | 0.3039 | 62 |
| k40_r5_c30 (report-only) | 291 | 0.8818 | 0.8061 | 0.9018 | 5 | 22.63 | 0.4467 | 0.3052 | 61 |
| k40_r10_c30 (report-only) | 287 | 0.8697 | 0.797 | 0.8978 | 8 | 22.63 | 0.4373 | 0.3014 | 62 |

k15 at cap 30 is the only report-only row that keeps every regime at or above 7a and stays
within the chars limits; it fails only mean_candidates (20.21 > 20.0). Whether 5 more slots are worth
~2 points of chars sent is a decision for later, not made here.

**Why reserving slots did not help.** At k >= 25 the reserve does not add slots, it moves them
from seeds of rank ~16-25 to hops, and those seeds are gold just as often: vs k15, k25_r5
gains 7 and loses 5 gold clauses, k25_r10 gains 4 and loses 6, k40_r10 gains 6 and loses 14.
The existing ranking puts tier before seed rank, so a reserved slot can go to a direct hop
(definition, outgoing reference) of a seed that was itself cut. Kept hops none of whose
source seeds is kept (`hops_from_cut_seeds` per row in `runs/eval_candidates_sweep_rows.csv`,
`all.hops_from_cut_seeds` / `share_hops_from_cut_seeds` per config in the sweep file):
k25_r5 22 of 1,155 (1.9 %), k25_r10 222 of 1,959 (11.3 %), k40_r5 92 of 911 (10.1 %), k40_r10
362 of 1,619 (22.4 %); 0 for every r = 0 config and at k = 15 (seeds rank before hops). An
earlier hand count gave 2 / 13 / 12 / 29 %; the reproducible numbers above replace it.

**Hop kind of the clauses lost to crowding** (user decision C; `<config>_hop_kind` in
`runs/eval_gold_clauses.csv` for the 7a configs and `runs/eval_gold_clauses_sweep.csv` for the
grid, `hits_by_hop_kind` in the summary and the sweep file). The 10 hop hits at k15
that are misses at k25, by the relation that brought them in at k = 15: reference_in 3,
definition 2, reference_in+subsection_of_reference 2, reference_out 1,
subsection_of_reference 1, definition+reference_in 1. 7 are in `ok` contracts, 3 in
`high_caps`; 7 dev, 3 test. k25_r5 recovers 5 of the 10 (as hops), k25_r10 recovers 9;
the A&C reference_in clause (high_caps) is lost in every k >= 25 config, cap 30 included. All 15 hop hits at
k = 15: definition 4, reference_in 3, reference_out 2, subsection_of_reference 2,
reference_in+subsection_of_reference 2, definition+reference_in 1, parent 1 (no
`subsection` hits).

**Ceiling.** 34 of 330 gold clauses are missed at every k of 7a (15/25/40), so no budget or
ordering change over those configs can exceed 296/330 = 0.897 micro clause recall. Across the
whole 18-config grid 28 are missed everywhere (union of cap-25 configs: 299/330 = 0.906).
These clauses are out of scope here (regime descriptions, excerpts, extraction).

**Confidentiality** (no gold): `extract_regimes.py --dry-run --regime Confidentiality` before
and after is identical (defaults unchanged): 75 calls, mean 18.59 candidates, 34.9 % of
contract chars sent, within the 20.0 / 0.39 ceilings. For reference k25_r5: 19.77 / 39.15 %.
The dry run needs no API key and makes no call (`runs/regime_calls.jsonl` unchanged, no
`runs/llm_cache/`).

Possible later levers (not done, need their own spec): rank hops of kept seeds before hops of
cut seeds; k = 15 at cap 30; the 34 always-missed clauses.
