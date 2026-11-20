---
name: ship-step
description: Run a roadmap step or feature through the dev team: pm -> engineer -> tester -> reviewer, with user approval of the spec. Use when the user says /ship-step or asks to implement a step "with the team".
---

You (the main session) are the orchestrator. Subagents cannot call each other, so you pass
results between them. The task is: $ARGUMENTS

1. **Plan** — delegate to the `pm` agent with the task. Show the user the spec and STOP until
   they approve or change it. Do not continue without approval.
2. **Build** — delegate to the `engineer` agent with the approved spec, verbatim.
3. **Test** — delegate to the `tester` agent with the acceptance criteria and the engineer's
   summary. If tests fail because of product code, send the failure report to `engineer`,
   then re-run `tester`. At most 2 such loops; after that, stop and report to the user.
4. **Review** — delegate to the `reviewer` agent with the spec. Send high-severity findings
   to `engineer` to fix (then re-run `tester` once). List medium/low findings for the user.
5. **Hand over** — summarise: files changed, ADR, test result, open review findings, and a
   proposed commit message (with design rationale, per CLAUDE.md). Do NOT commit; the user decides.
