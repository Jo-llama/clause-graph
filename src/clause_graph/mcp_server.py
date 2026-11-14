"""MCP server exposing the contract graph (stdio transport).

Run: uv run src/clause_graph/mcp_server.py

Tool design rules:
- verb_noun names; each description tells the model WHEN to use the tool and WHAT comes back.
- Read-only and parameterised: there is deliberately no tool that runs arbitrary Cypher.
- Every returned item carries provenance: doc_id, section, uid.
- Bounded output: clause text is cut to max_chars (default 1500) and flagged "truncated".
- doc_id accepts a unique prefix (CUAD ids are long); ambiguity returns the candidates.
- Failures come back as is_error results with a hint on how to fix the call, never a traceback.
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from clause_graph import queries
from clause_graph.queries import DEFAULT_MAX_CHARS, QueryError

ROOT = Path(__file__).resolve().parents[2]
READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

_driver: Driver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        load_dotenv(ROOT / ".env")
        password = os.environ.get("NEO4J_PASSWORD")
        if not password:
            raise QueryError("Server misconfigured: NEO4J_PASSWORD is not set in .env.")
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        _driver = GraphDatabase.driver(uri, auth=("neo4j", password))
    return _driver


def safe(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Turn failures into a ToolError: the client gets is_error=True plus a short
    message the model can act on, never a traceback."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return fn(*args, **kwargs)
        except QueryError as e:
            raise ToolError(str(e)) from e
        except ServiceUnavailable as e:
            raise ToolError("Neo4j is not reachable. Start it with: docker compose up -d") from e
        except AuthError as e:
            raise ToolError("Neo4j rejected the credentials; check NEO4J_PASSWORD in .env.") from e
        except Exception as e:  # last resort: log details server-side only
            print(f"{fn.__name__} failed: {e!r}", file=sys.stderr)
            raise ToolError(f"Internal error in {fn.__name__}: {type(e).__name__}: {e}") from e

    return wrapper


def build_instructions() -> str:
    """Server instructions with the contract count read from the graph at startup,
    so they never drift from what is actually loaded. If Neo4j is unreachable the
    count is omitted; the tools themselves then report the connection error."""
    try:
        n = queries.count_contracts(get_driver())
        corpus = f"{n} CUAD commercial contracts"
    except Exception as e:
        print(f"contract count unavailable at startup: {e!r}", file=sys.stderr)
        corpus = "CUAD commercial contracts"
    return (
        f"Knowledge graph of {corpus}: clauses, defined terms and "
        "cross-references, with local semantic search. Fastest path for a question about one "
        "contract: assemble_context(query, doc_id) - vector search plus one graph hop, every "
        "item says why it is included. Manual flow: list_contracts -> search_clauses (meaning) "
        "or find_clauses (exact wording) -> get_clause_context (the clause, definitions of the "
        "terms it uses, clauses it cites or is cited by, subsections) -> get_definition for "
        "any other term. Always cite doc_id and section from the results."
    )


server = MCPServer("clause-graph", instructions=build_instructions())


@server.tool(annotations=READ_ONLY)
@safe
def list_contracts() -> dict:
    """List all contracts in the graph with their extraction metrics.

    Use this first to pick a contract and get its doc_id. Returns doc_id plus
    metrics: sections, definitions, term_uses, hub_terms, references,
    unresolved_refs, unresolved_share. sections=0 means section splitting failed
    for that contract (the whole text is the preamble, section "0").
    Other tools accept any unique prefix of doc_id.
    """
    contracts = queries.list_contracts(get_driver())
    return {"count": len(contracts), "contracts": contracts}


@server.tool(annotations=READ_ONLY)
@safe
def find_clauses(
    doc_id: str, keyword: str, limit: int = 10, max_chars: int = DEFAULT_MAX_CHARS
) -> dict:
    """Find clauses in one contract whose text contains a keyword (case-insensitive substring).

    Use this to locate where a topic is handled, e.g. keyword "liability",
    "terminat", "indemn". This is literal matching, not semantic search: try word
    stems and synonyms if nothing comes back. Returns total_matches and up to
    `limit` (max 50) clauses in section order, each with doc_id, section, uid and
    text; long text is excerpted around the first hit and marked truncated.
    Follow up with get_clause_context on a section to see its full regime.
    """
    return queries.find_clauses(get_driver(), doc_id, keyword, limit, max_chars)


@server.tool(annotations=READ_ONLY)
@safe
def get_definition(doc_id: str, term: str, max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    """Get the contractual definition of a defined term (e.g. "Losses", "Products").

    Use this when a clause relies on a capitalised defined term and you need its
    exact meaning. Returns every defining clause (doc_id, section, uid, text excerpt
    starting at the definition). If the term is defined in more than one section,
    all are returned and a note says so - check which one governs. If the term is
    unknown, the error lists up to 5 similar defined terms from that contract.
    """
    return queries.get_definition(get_driver(), doc_id, term, max_chars)


@server.tool(annotations=READ_ONLY)
@safe
def get_clause_context(
    doc_id: str, section: str, max_chars: int = DEFAULT_MAX_CHARS
) -> dict:
    """Assemble the full regime around one clause - the core tool for answering questions.

    Use this after locating a relevant section (e.g. a liability cap) to collect
    everything needed to read it correctly. Returns:
    - clause: the clause itself (doc_id, section, uid, text) and the terms it defines;
    - definitions: for each non-hub defined term the clause uses, the defining
      clause(s) as excerpts starting at the definition;
    - references: clauses it cites (direction "outgoing", e.g. carve-outs it points
      to) and clauses citing it (direction "incoming", e.g. exclusions that override it);
    - unresolved_references: section numbers cited in the text that do not exist
      in the extracted graph - treat them as gaps, not as absent provisions.
    The clause and every referenced clause carry "subsections" when they have any
    (e.g. a citation of "Section 9.0" comes with 9.1, 9.2, ...), in document order.
    section is the number as written ("12", "12.3"); "0" is the preamble; H1, H2, ...
    are unnumbered ALL-CAPS headings (e.g. "WRITTEN AGREEMENT").
    Hub terms like "Agreement" are not expanded.
    """
    return queries.get_clause_context(get_driver(), doc_id, section, max_chars)


@server.tool(annotations=READ_ONLY)
@safe
def search_clauses(
    query: str, doc_id: str | None = None, k: int = 10, max_chars: int = DEFAULT_MAX_CHARS
) -> dict:
    """Semantic search: find clauses whose meaning matches a natural-language query.

    Use this when the contract may phrase a topic differently from your words, e.g.
    "limitation of liability" also finds "shall not exceed the insurance coverage".
    Omit doc_id to search all contracts. Returns up to k (max 50) clauses, best
    first, each with doc_id, section, uid, score (cosine similarity, roughly
    0.3 weak .. 0.6+ strong) and text excerpted around the best-matching chunk.
    Embeddings are computed locally; no text leaves the machine. For exact wording
    use find_clauses instead.
    """
    return queries.search_clauses(get_driver(), query, doc_id, k, max_chars)


@server.tool(annotations=READ_ONLY)
@safe
def assemble_context(
    query: str,
    doc_id: str,
    k: int = queries.DEFAULT_SEEDS,
    max_items: int = queries.DEFAULT_CONTEXT_ITEMS,
    max_chars: int = queries.DEFAULT_CONTEXT_CHARS,
) -> dict:
    """GraphRAG in one call: everything needed to answer a question about one contract.

    Use this first for questions like "how is liability limited?". Takes the k
    best semantic matches as seeds, then adds one graph hop around each seed:
    definitions of terms it uses, clauses it references and their subsections,
    clauses referencing it, its subsections, parent and siblings. Returns items in
    document order; each has doc_id, section, uid, role ("seed" or "context"),
    score, text, and "why" - the reasons it is included (e.g. "vector match 0.57
    (rank 1)", "defines 'Freight' used by 6.0", "referenced by 8.2"). At most
    max_items (default 25) items; "omitted" counts what was cut, by relation.
    """
    return queries.assemble_context(get_driver(), query, doc_id, k, max_items, max_chars)


if __name__ == "__main__":
    server.run()  # stdio
