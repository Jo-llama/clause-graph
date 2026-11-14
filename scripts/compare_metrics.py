"""Compare two extraction metrics CSVs (before/after an extractor change).

Usage: uv run scripts/compare_metrics.py [BEFORE] [AFTER]
Defaults: runs/extraction_metrics_v1.csv vs runs/extraction_metrics.csv.
"""

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEFORE = ROOT / "runs" / "extraction_metrics_v1.csv"
AFTER = ROOT / "runs" / "extraction_metrics.csv"


def load(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8") as f:
        return {
            r["doc_id"]: {k: float(v) for k, v in r.items() if k != "doc_id" and v != ""}
            for r in csv.DictReader(f)
        }


def clauses(m: dict[str, float]) -> float:
    """All clauses incl. preamble and synthetic ALL-CAPS sections."""
    return m["sections"] + m.get("caps_sections", 0) + (1 if m["preamble_chars"] else 0)


def summary(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    ms = list(rows.values())
    refs = sum(m["references"] for m in ms)
    unresolved = sum(m["unresolved_refs"] for m in ms)
    return {
        "contracts": len(ms),
        "median sections (numbered)": statistics.median(m["sections"] for m in ms),
        "median caps sections (H*)": statistics.median(m.get("caps_sections", 0) for m in ms),
        "contracts with sections=0": sum(m["sections"] == 0 for m in ms),
        "subsection links (total)": sum(m.get("subsection_links", 0) for m in ms),
        "references resolved (total)": refs,
        "unresolved share (total)": round(unresolved / (refs + unresolved), 3),
        "median preamble_chars": statistics.median(m["preamble_chars"] for m in ms),
        "max preamble_chars": max(m["preamble_chars"] for m in ms),
        "median max_clause_chars": statistics.median(m["max_clause_chars"] for m in ms),
        "max max_clause_chars": max(m["max_clause_chars"] for m in ms),
    }


def fmt(x: float) -> str:
    return f"{x:,.3f}" if isinstance(x, float) and not x.is_integer() else f"{int(x):,}"


def main() -> int:
    before_path = Path(sys.argv[1]) if len(sys.argv) > 1 else BEFORE
    after_path = Path(sys.argv[2]) if len(sys.argv) > 2 else AFTER
    before, after = load(before_path), load(after_path)

    b, a = summary(before), summary(after)
    print(f"{'metric':<30} {'before':>10} {'after':>10}")
    for key in b:
        print(f"{key:<30} {fmt(b[key]):>10} {fmt(a[key]):>10}")

    # Most changed: relative change in the number of clauses the contract is cut into.
    common = [d for d in after if d in before]
    ranked = sorted(
        common, key=lambda d: -abs(clauses(after[d]) - clauses(before[d])) / clauses(before[d])
    )
    print("\n5 contracts that changed most (clauses = preamble + numbered + H*)")
    print(f"{'doc_id':<45} {'clauses':>11} {'sections':>9} {'preamble':>15} {'max clause':>17}")
    for d in ranked[:5]:
        x, y = before[d], after[d]
        print(
            f"{d[:45]:<45} {int(clauses(x)):>4} ->{int(clauses(y)):>4} "
            f"{int(x['sections']):>3} ->{int(y['sections']):>3} "
            f"{int(x['preamble_chars']):>6} ->{int(y['preamble_chars']):>6} "
            f"{int(x['max_clause_chars']):>7} ->{int(y['max_clause_chars']):>7}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
