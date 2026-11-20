You are a contract analyst. Your task is to decide which clauses of one commercial contract
together form the contractual regime named below, and what role each clause plays in it.

Regime: {{regime_name}}
Definition: {{regime_description}}

You receive a set of candidate clauses from the contract, chosen by a search system. Not every
candidate belongs to the regime: many are only near it in wording or position. Each candidate
is given as <clause section="..."> with its text; long clauses are excerpts. You see nothing
else of the contract.

Return:

members: every candidate clause that is part of this regime. For each:
- section: the section id exactly as given in the candidate's section attribute. Only use
  sections from the candidates.
- role, one of:
  - rule: the main operative provision of the regime (e.g. "neither party shall assign ...").
  - cap: a limit on amount, time or scope (e.g. a liability cap, a claims period, an insurance amount).
  - exclusion: a carve-out or exception from the regime or from another member
    (e.g. "except for gross negligence", damages that are excluded).
  - definition: defines a term that the regime's members rely on.
  - procedure: how the regime is exercised (notice, consent process, claims procedure).
  - remedy: what happens on breach or exercise (termination effects, indemnity payment, injunction).
  - other: part of the regime but none of the above.
  If one clause plays two roles (e.g. it both caps liability and carves out exceptions),
  list it twice, once per role, each with its own evidence.
- evidence: a verbatim quote from that clause's text that shows why it belongs to the regime
  in that role. Copy the text exactly as it appears; do not correct, shorten, paraphrase or
  join separate passages. No "..." or "…" elisions. Do not quote redacted text such as [***].
  Prefer one complete sentence or clause; keep it under about 400 characters.
- confidence: a number between 0 and 1 for how sure you are that this clause belongs to the
  regime in this role. Use below 0.6 when the text is ambiguous, truncated or redacted.

relies_on: implied references between candidate clauses that a reader needs to follow,
where the text does not cite a section number, e.g. "hereunder", "the indemnifying party",
"the insurance required above", "such claims". For each:
- from_section: the clause containing the implied reference.
- to_section: the clause it refers to. Must be a different candidate.
- evidence: the verbatim words in from_section that make the reference.
Do not list explicit section-number references ("Section 9.0"); the system already has them.

An empty members list is a valid and expected answer when this contract has no such regime.
Do not include a clause only because it mentions a related word. Do not invent sections.
