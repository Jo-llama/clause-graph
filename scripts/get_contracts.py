"""Download the CUAD golden set from clause-harness and write data/contracts.jsonl.

Output rows: {"doc_id": ..., "text": ...}. Standard library only.
"""

import json
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/Jo-llama/clause-harness/main/data/golden.jsonl"
OUT = Path(__file__).resolve().parent.parent / "data" / "contracts.jsonl"


def main() -> int:
    with urllib.request.urlopen(URL, timeout=60) as resp:
        lines = resp.read().decode("utf-8").splitlines()

    seen: set[str] = set()
    rows = []
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["doc_id"] in seen:
            continue
        seen.add(rec["doc_id"])
        rows.append({"doc_id": rec["doc_id"], "text": rec["source_text"]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} contracts to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
