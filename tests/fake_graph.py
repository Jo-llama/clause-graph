"""Synthetic fixture graph for collect_context / candidates / assemble_context tests.

No database: answers the Cypher strings queries.py and regimes.py issue, by identity.
Clauses are "FIX#<n>" with section "<n>" and position n; text is synthetic.
"""

from clause_graph import queries, regimes

DOC = "FIX"


def uid(section: str | int) -> str:
    return f"{DOC}#{section}"


class Rec:
    def __init__(self, d):
        self.d = d

    def data(self):
        return self.d

    def __getitem__(self, key):
        return self.d[key]


class Result(tuple):
    """(records, summary, keys) like neo4j's EagerResult, which also has .records."""

    @property
    def records(self):
        return self[0]


def result(rows: list[dict]) -> Result:
    return Result(([Rec(r) for r in rows], None, None))


def seed_row(section: int, rank: int) -> dict:
    """Vector seed row as TOP_CHUNKS / _score_clauses_in_doc return it (best first)."""
    hit = {"score": round(0.95 - rank * 0.01, 4), "idx": 0, "start": 0}
    return {"section": str(section), "uid": uid(section), "best": hit, "hits": [hit],
            "text": text(section), "position": section, "doc_id": DOC, "chunks_matched": 1}


def text(section: int | str) -> str:
    return f"{section}.  Synthetic clause {section}. The parties agree to term {section}."


class FakeGraph:
    """seeds: clause numbers in vector rank order. edges: {kind: [(seed, target, term, via)]}
    for the expansion kinds in queries.EXPANSIONS."""

    def __init__(self, seeds: list[int], edges: dict[str, list[tuple]]):
        self.seeds = seeds
        self.edges = edges
        self.queries: list[str] = []
        self.by_cypher = {cypher: kind for _, kind, cypher, _ in queries.EXPANSIONS}

    def seed_rows(self, k: int | None = None) -> list[dict]:
        return [seed_row(s, i) for i, s in enumerate(self.seeds[:k])]

    def execute_query(self, query, params=None, routing_=None):
        self.queries.append(query)
        p = params or {}
        if query == regimes.TOP_CHUNKS:
            rows = [{key: r[key] for key in ("section", "uid", "best", "hits")}
                    for r in self.seed_rows(p["k"])]
        elif query in self.by_cypher:
            kind = self.by_cypher[query]
            rows = [{"seed": uid(a), "uid": uid(b), "term": term, "via": via}
                    for a, b, term, via in self.edges.get(kind, []) if uid(a) in p["seeds"]]
        elif "c.uid IN $uids" in query:
            rows = []
            for u in p["uids"]:
                s = u.rsplit("#", 1)[1]
                rows.append({"uid": u, "section": s, "text": text(s), "position": int(s)})
        else:
            raise AssertionError(f"unexpected query: {query[:80]}")
        return result(rows)


def small_graph() -> FakeGraph:
    """8 seeds; one hop of every kind; seed 1 references seed 6 (seed and hop target)."""
    return FakeGraph(
        seeds=[1, 2, 3, 4, 5, 6, 7, 8],
        edges={
            "definition": [(1, 20, "Losses", None)],
            "reference_out": [(1, 6, None, None), (2, 21, None, None), (8, 22, None, None)],
            "parent": [(3, 23, None, None)],
            "reference_in": [(4, 24, None, None)],
            "sibling": [(5, 25, None, "9")],
            "subsection": [(7, 26, None, None)],
        },
    )


def wide_graph(n_seeds: int = 30) -> FakeGraph:
    """n_seeds seeds, each referencing its own clause 100+i: many seeds and many hops."""
    seeds = list(range(1, n_seeds + 1))
    return FakeGraph(seeds, {"reference_out": [(s, 100 + s, None, None) for s in seeds]})
