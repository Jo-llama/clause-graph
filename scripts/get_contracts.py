"""Download the CUAD golden set from clause-harness and write data/contracts.jsonl
and data/gold.jsonl.

contracts.jsonl rows: {"doc_id": ..., "text": ...}       (input to load_graph.py)
gold.jsonl rows:      {"doc_id": ..., "labels": {...}}   (input to eval_candidates.py, ADR 0005)
labels is copied unchanged: {<category>: {"present": bool, "evidence": [str, ...]}}.
contracts.jsonl is only rewritten when its content would change, so an existing load
is never touched by fetching the gold labels. Exits non-zero and writes nothing if an
upstream row has no labels object (it would otherwise become a contract without gold).
Standard library only.
"""

import json
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/Jo-llama/clause-harness/main/data/golden.jsonl"
DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "contracts.jsonl"
GOLD = DATA / "gold.jsonl"


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    with urllib.request.urlopen(URL, timeout=60) as resp:
        lines = resp.read().decode("utf-8").splitlines()

    seen: set[str] = set()
    rows, gold, no_labels = [], [], []
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["doc_id"] in seen:
            continue
        seen.add(rec["doc_id"])
        rows.append({"doc_id": rec["doc_id"], "text": rec["source_text"]})
        if not isinstance(rec.get("labels"), dict):
            no_labels.append(rec["doc_id"])
        gold.append({"doc_id": rec["doc_id"], "labels": rec.get("labels")})
    if no_labels:
        # Empty labels would silently become "no gold" for the contract; nothing is written.
        sys.exit(f"ERROR: {len(no_labels)} upstream row(s) have no 'labels' object, nothing "
                 f"written: {no_labels[:5]}")

    DATA.mkdir(parents=True, exist_ok=True)
    for path, content, n, what in ((OUT, jsonl(rows), len(rows), "contracts"),
                                   (GOLD, jsonl(gold), len(gold), "gold rows")):
        changed = write_if_changed(path, content)
        print(f"{'wrote' if changed else 'unchanged'}: {n} {what} in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
