# 0006 Candidate-config evaluation: hashed dev/test split, decision rule fixed in advance, report-only configs

Status: Accepted, 2026-11-20 (step 7a follow-up, `docs/known-issues.md` section 8b). Adds to
ADR 0005 without changing it; ADR 0004 and ADR 0005 stay Accepted.

## Context
ADR 0005 measures candidate recall, the free upper limit on step-6 LLM recall, and allows a
`--k` sweep that is "reported, not adopted". The first follow-up asked whether a different
candidate selection should become the default: more vector seeds (k), a number of the 25
slots reserved for graph-hop clauses (`hop_reserve`, `queries.reserve_hops`), or a larger
cap. Tuning such knobs on the same 75 contracts that report the result overfits: every
config is a guess scored on the data it was picked for. Nothing in ADR 0005 says how a
config is chosen, which contracts may be used to choose it, or what a config is called in
the outputs. The step-6 extraction script also accepted any selection without recording
it, so PART_OF / RELIES_ON edges could come from a selection no evaluation had accepted.

## Decision
- **Fixed dev/test split** (`evaluate.split_of`): a contract is `test` if
  `int(sha256(doc_id)[:8], 16) % 4 == 0`, else `dev`. No database, no randomness, no list
  to maintain: 59 dev / 16 test of the 75 loaded contracts (with gold: 57 dev contracts,
  262 gold clauses; 14 test contracts, 68 gold clauses). The rule **must not change**: 7b
  (scoring PART_OF edges) and 7c (local model) reuse it, and any change moves contracts
  that were used to choose a config into the test split. A test pins the split of a few
  synthetic ids.
- **Decision rule and adoption targets are fixed before a sweep runs** and live in code
  (`evaluate.DECISION_RULE`, `evaluate.ADOPTION_TARGETS`, locked by a test):
  1. On **dev** only (`evaluate.decide`): among eligible configs pick the highest micro
     clause recall; ties go to fewer chars sent, then grid order. Eligible = not
     report-only, dev `chars_sent_ratio <= 0.39`, dev `mean_candidates <= 20.0`, and no
     regime losing more than 1 dev gold clause against the baseline (k = 15, r = 0, cap 25,
     the ADR 0004 selection). The defaults change only if the best config beats the
     baseline by **at least 2 dev gold clauses**.
  2. **Adoption check** on all contracts (`evaluate.adoption_check`), only for a dev
     winner: at least 4 more gold clauses than the baseline, excerpt and span recall not
     lower, no regime's micro clause recall lower, at least one hop hit, `chars_sent_ratio
     <= 0.39`, `mean_candidates <= 20.0`, high_caps contracts `chars_sent_ratio <= 0.25`,
     and at most 1 test gold clause lost. Every target is listed with value, threshold and
     result.
  Changing a threshold after seeing results needs a new ADR.
- **Configs with a cap other than 25 (`regimes.MAX_CANDIDATES`) are report-only**: they are
  measured and listed (`report_only`), but `decide` never makes them eligible. A larger cap
  is a cost decision (more chars sent per call), not a tuning result.
- **One canonical config name** (`regimes.candidate_config_name(k, hop_reserve,
  max_candidates)`): `k{k}`, plus `_r{r}` unless r = 0, plus `_c{cap}` unless cap = 25.
  Examples: `k15` (the 7a default), `k25_r5`, `k15_c30`, `k40_r10_c30`. The elided values
  are fixed literals (`NAME_ELIDED_RESERVE`, `NAME_ELIDED_CAP`), not the current defaults,
  so a name keeps its meaning if a default changes later. The same config has the same name
  everywhere: `by_config` keys and the CSV `config` column / gold-CSV columns of
  `scripts/eval_candidates.py`, the grid and decision table of the sweep file, and
  `candidate_config` in `runs/regime_metrics.csv` and `runs/regime_calls.jsonl` of
  `scripts/extract_regimes.py`.
- **Summary key mapping**: the summary key `by_k.<k>` named in ADR 0005 and
  `docs/known-issues.md` section 8 is now `by_config.k<k>` (e.g. `by_k.15` ->
  `by_config.k15`); the numbers under it are unchanged.
- **A sweep never overwrites the committed 7a outputs**: with `--sweep-out` the script
  writes only the sweep file (committed: `runs/eval_candidates_sweep.json`, counts and ids,
  no text, no timestamp) and git-ignored per-row / per-gold-clause CSVs
  (`runs/eval_candidates_sweep_rows.csv`, `runs/eval_gold_clauses_sweep.csv`).
  `runs/eval_candidates.csv` and `runs/eval_candidates_summary.json` stay the 7a run
  (`--k 15 --k 25 --k 40`).
- **No edges from a non-adopted selection**: `scripts/extract_regimes.py` accepts a
  `--hop-reserve` other than the adopted default (`regimes.HOP_RESERVE`) only with
  `--dry-run`. Its metrics rows are keyed by (doc_id, model, prompt_version,
  candidate_config), so runs with different configs do not overwrite each other.
- **Every row reports `hops_from_cut_seeds`**: kept graph-hop candidates none of whose
  source seeds was kept (`Candidate.sources`), aggregated as `hops_from_cut_seeds` /
  `share_hops_from_cut_seeds` per group. With r = 0 it is always 0 (seeds rank before hops).

## Result of the first sweep
Grid `--k 15 --k 25 --k 40 --hop-reserve 0 --hop-reserve 5 --hop-reserve 10
--max-candidates 25 --max-candidates 30` (18 configs, 6 report-only),
`runs/eval_candidates_sweep.json`, details in `docs/known-issues.md` section 8b. **Nothing
was adopted; the defaults stay k = 15, `HOP_RESERVE = 0`, cap 25.** Only the three k = 15
configs at cap 25 are eligible (identical: at k = 15 at most 15 seeds compete for 25 slots,
so a reserve <= 10 changes nothing), and they gain 0 dev gold clauses. The closest
contender, `k25_r5` (+3 dev gold clauses), fails the dev chars limit (0.3956 > 0.39) and
would also have failed the adoption targets on all 75 contracts.

## Alternatives considered
- **Tune on all 75 contracts**: simplest, but the reported recall would then be the
  number the config was picked for, and 7b/7c would have no untouched contracts.
- **Random or stratified split stored as a list**: a list has to be versioned and
  maintained as contracts are added; a seeded random split depends on the order of the
  contract file. The hash rule needs neither and assigns a new contract without touching
  the others.
- **Choose by eye from the sweep table**: invites moving the threshold to fit a favourite
  config. Fixing the rule in code before the run, with a test on the constants, makes
  that visible in a diff.
- **Let larger caps compete**: a larger cap almost always raises recall, so it would win
  any recall-first rule; whether the extra cost is worth it is a separate decision.
- **Always spell out every parameter (`k15_r0_c25`)**: unambiguous, but renames the
  committed 7a keys (`k15`) and every reference to them. Names relative to the current
  defaults (the earlier draft, where `k15` meant `k15_r0` only when no other r was in the
  grid) give one config two names. Eliding fixed literals avoids both.
- **Later levers, not decided here** (each needs its own spec): rank the hops of kept seeds
  before the hops of cut seeds (a reserved slot can go to a hop whose seed was itself cut;
  `hops_from_cut_seeds`); k = 15 at cap 30 (report-only `k15_c30`: the only report-only row
  keeping every regime at or above 7a and within the chars limits, failing only
  `mean_candidates` 20.21 > 20.0); the clauses missed by every config.

## Consequences
- A future candidate change is decided on dev with a rule that cannot drift towards the
  result, and reported on test and on all contracts.
- The test split is small (14 contracts with gold, 68 gold clauses): it catches large
  regressions, not differences of one or two clauses; hence the "at most 1 lost" target.
- The split is permanent. Adding contracts adds to both sides; removing the rule or
  re-salting it would invalidate every dev/test comparison made so far.
- The committed 7a outputs keep their keys (`by_config.k15`, `k25`, `k40`); new columns
  (`hops_from_cut_seeds`) and fields were added without changing any 7a number (micro
  clause recall 282/330 = 0.8545 at k = 15).
- `regime_metrics.csv` rows written before `candidate_config` existed keep a blank config
  and are never replaced by a new run.
- `extract_regimes.py` can still write edges with a non-default `--k` or
  `--max-candidates`; those are now recorded by name in the metrics and call log, but not
  blocked.
