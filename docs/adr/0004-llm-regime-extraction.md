# 0004 LLM regime extraction: structured output, deterministic validation, minimised candidates

Status: Accepted, 2026-11-20 (proposed 2026-11-14; step 6)

## Context
The graph from steps 3 to 5 knows which clauses cite which, and which are similar in meaning.
It does not know which clauses together form one contractual regime, or what role each
one plays. For example, a liability regime is a cap, its carve-outs, the indemnity it
limits and the insurance amounts it points to. Deciding that needs understanding, so it
calls for an LLM (ADR 0002). Two risks follow:
- A legal reader cannot trust free-form LLM output.
- Contract text sent to an LLM API leaves the machine (ADR 0003).

## Decision
- **One provider-neutral module**, `src/clause_graph/llm.py`, selected by
  `LLM_PROVIDER=anthropic|ollama`. Anthropic is implemented now. Ollama comes in step 7c
  and will use its JSON-schema `format` mode, so the same schema and the same
  validation apply. Model ids: `LLM_MODEL=claude-sonnet-5-5`,
  `COMPARE_MODEL=claude-opus-5-5`, checked against platform.claude.com on 2026-11-14.
- **Six regimes** in `src/clause_graph/regimes.json`: Liability, Termination, Assignment &
  Control, Restrictive Covenants, Governing Law & Disputes, Confidentiality. Each has a
  description, which serves both as the vector query and as the definition in the
  prompt, and its CUAD categories for the step-7 eval.
- **Candidates without an LLM, which is also data minimisation.** For each (contract,
  regime), the system takes the top k=15 chunks for the regime description within the
  contract and groups them by clause. It then adds one graph hop: definitions,
  references in both directions, subsections and the parent. That makes at most 25
  candidates, each carrying the reason it was included. Only these excerpts are sent,
  never the whole contract, and every call logs the characters sent against the
  contract's total.
- **One call per (contract, regime), schema-constrained.** The call uses structured
  outputs (`output_config.format`, JSON schema) and is validated again with Pydantic.
  The output has `members` (section, role ∈ {rule, cap, exclusion, definition,
  procedure, remedy, other}, verbatim evidence, confidence) and `relies_on` (implied
  references such as "hereunder" or "the indemnifying party"). An empty member list is
  a valid answer. The prompt is a versioned file (`prompts/assign_regime_v1.md`), and
  every edge records `method`, `model` and `prompt_version`.
- **Deterministic validation, then escalation.** A member is stored with status
  `needs_review` (still stored, flagged with the failed checks) when any of these hold:
  - its section is not a candidate;
  - its evidence is not verbatim in that clause's full text (whitespace-normalised);
  - its evidence contains a "..." elision or a "[***]" redaction;
  - its confidence is below 0.6.

  Otherwise it is `accepted`. A section that does not exist in the contract, or a
  self-reference, cannot be stored and is counted as rejected. A schema violation,
  refusal or truncation fails the whole call, and old edges stay in place. This
  follows the clause-harness design: a fabricated citation is caught, and a correct
  paraphrase costs a minute of human review.
- **Graph**: `(:Regime {uid: doc_id#name})`,
  `(:Clause)-[:PART_OF {role, evidence, confidence, status, checks, method, model,
  prompt_version}]->(:Regime)`, and `(:Clause)-[:RELIES_ON {..., regime}]->(:Clause)`.
  A re-run replaces the edges of the same (contract, regime, model, prompt_version), so
  Sonnet and Opus results can sit side by side for comparison. `get_regime` (MCP)
  serves them, and `assemble_context` follows RELIES_ON edges with the reason
  "implied reference (LLM, status)".
- **Cost control**: every raw response is cached in `runs/llm_cache/<prompt_version>/
  <model>/`, keyed by a hash of the exact input (prompt, candidates, schema, model,
  effort), so a re-run never pays twice. Token counts and cost are recorded for every
  call. Prices (as of 2026-11-14, USD per MTok in/out): Sonnet 5.5 $2/$10, Opus 5.5
  $4/$20. The first run covers 10 contracts; all 75 run only after review.

## Alternatives considered
- **Forced tool (`tool_choice: {type: "tool"}`)**: this was the original plan. Sonnet 5.5
  and Opus 5.5 reject it with a 400 error; the docs recommend strict tools or structured
  outputs instead. Structured outputs give the same guarantee (schema-valid JSON, no
  prose), keep the code simpler, and correspond directly to Ollama's JSON-schema mode.
  Older models that accept forced tools (Sonnet 5, Opus 5) were not chosen, so as not
  to tie the design to a feature the current models dropped.
- **Extended thinking**: not used. Manual extended thinking is rejected on these models
  anyway. On Sonnet 5.5 the lowest setting is `thinking: between_tools`, which with no
  tools means no thinking. Opus 5.5 cannot turn thinking off, so its adaptive thinking
  tokens are billed as output tokens and show up in the comparison.
- **Whole contract in the prompt**: simpler and higher recall, but every clause of every
  contract would leave the machine. Candidate selection also makes the "why" of each
  candidate auditable.
- **Section ids as a schema enum per call**: this would make invented sections impossible
  at decoding time, but it hides how often the model would invent them. With the
  deterministic check, that rate is a measured number.
- **Keyword rules per regime**: deterministic, but regimes are defined by meaning and by
  implied references ("such claims"), which regex cannot follow (ADR 0002).

## Consequences
- Every LLM claim is checkable: a verbatim quote, a status, the failed checks and the
  model plus prompt version. `needs_review` is the escalation queue.
- Recall is bounded by candidate selection. A regime clause outside the top-k chunks
  and one hop is never seen. The step-7 eval against the CUAD gold spans measures this.
- Minimisation is weaker for small contracts with few, long clauses, and for contracts
  where section splitting failed (sections=0): there a few candidates already cover most
  of the text (see the measured ratio).
- **Data protection**: LLM API calls are allowed here only because CUAD is public. For
  real client contracts, German professional secrecy (§ 203 StGB; for lawyers § 43e
  BRAO on IT service providers) points to three safeguards:
  - data minimisation, as implemented here;
  - an EU-hosted model under a contract that meets § 43e BRAO;
  - a local model (the Ollama provider, step 7c).

  Provider choice is a configuration switch, not a rewrite.
