# Known issues: deterministic extraction

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

## 4. Very long clauses (sub-chunking needed before step 5)

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

## 5. Minor: ALL-CAPS heading false positives

In a sample of 30 detected ALL-CAPS headings about 25 were real boundaries. The rest are mostly
signature-block company names (`FEDERATED INVESTMENT MANAGEMENT COMPANY` followed by `By:`)
and table rows. They create small extra clauses at the end of a contract and rarely cut a
clause mid-sentence; repeated page headers (>2 occurrences) are already filtered.
