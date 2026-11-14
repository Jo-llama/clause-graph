"""Split clauses into embedding chunks. Pure functions: no model, no database.

A clause up to MAX_TOKENS is one chunk. Longer clauses are cut at sentence
boundaries into chunks of at most MAX_TOKENS, each starting with up to
OVERLAP_TOKENS of the previous chunk's last sentences. Every chunk keeps its
character offsets into the clause text, so it can be traced back to
doc_id + section + source text.

How tokens are counted depends on the chunker version, stored on every Chunk
(ADR 0003):
  v1  chars / CHARS_PER_TOKEN on whitespace-collapsed text (estimate_tokens).
      Real bge-m3 counts run 0.94x this on average but up to 1.57x on dense text.
  v2  real model tokens; the caller passes the counter (embeddings.count_tokens).
Existing contracts keep the version they were chunked with, so reloads keep their
embeddings; new contracts get CURRENT_CHUNKER.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass

MAX_TOKENS = 1500
OVERLAP_TOKENS = 100
CHARS_PER_TOKEN = 4
CHUNKER_V1 = 1  # estimated tokens
CHUNKER_V2 = 2  # real model tokens
CURRENT_CHUNKER = CHUNKER_V2
MIN_LOWERCASE = 10  # fewer lowercase letters = bare heading ("9.0    INSURANCE"), not embedded

# Sentence end: . ; : ? ! followed by whitespace and a capital, digit, quote or
# bracket; or a blank line. Abbreviations like "U.S. Patent" may split - harmless.
SENTENCE_END_RE = re.compile(r"(?<=[.;:?!])\s+(?=[A-Z0-9(\"“])|\n[ \t]*\n\s*")


@dataclass(frozen=True)
class Chunk:
    idx: int
    start: int  # char offsets into the clause text
    end: int
    text: str


def collapse(text: str) -> str:
    """Whitespace-collapsed text: what is embedded and what tokens are estimated on."""
    return " ".join(text.split())


def text_hash(text: str) -> str:
    """Identity of a chunk's text; an embedding is valid only for the hash it was made from."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    return math.ceil(len(collapse(text)) / CHARS_PER_TOKEN)


def is_near_empty(text: str) -> bool:
    return sum(c.islower() for c in text) < MIN_LOWERCASE


Counter = Callable[[str], int]


def chunker_version(doc_id: str, existing: dict[str, int], rechunk: str | None = None) -> int:
    """Version to chunk a contract with: the one it already has (so reloads keep their
    embeddings), CURRENT_CHUNKER if it is new or matches the --rechunk prefix."""
    if rechunk is not None and doc_id.startswith(rechunk):
        return CURRENT_CHUNKER
    return existing.get(doc_id, CURRENT_CHUNKER)


def sentence_spans(
    text: str, max_tokens: int = MAX_TOKENS, count: Counter = estimate_tokens
) -> list[tuple[int, int]]:
    """(start, end) of each sentence; sentences over max_tokens are cut at whitespace."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in SENTENCE_END_RE.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))

    max_chars = max_tokens * CHARS_PER_TOKEN
    out: list[tuple[int, int]] = []
    for start, end in spans:
        while count(text[start:end]) > max_tokens:
            limit = start + max_chars
            while True:
                cut = text.rfind(" ", start, limit)
                cut = cut if cut > start else limit
                # Estimated tokens never exceed max_chars / 4; real ones can.
                if count(text[start:cut]) <= max_tokens or limit - start < 2:
                    break
                limit = start + (limit - start) * 9 // 10
            out.append((start, cut))
            start = cut
            while start < end and text[start].isspace():
                start += 1
        if start < end:
            out.append((start, end))
    return out


def chunk_clause(
    text: str,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    count: Counter = estimate_tokens,
) -> tuple[list[Chunk], int]:
    """Chunks of one clause and the number skipped as near-empty."""
    if count(text) <= max_tokens:
        if is_near_empty(text):
            return [], 1
        return [Chunk(0, 0, len(text), text)], 0

    spans = sentence_spans(text, max_tokens, count)
    groups: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for span in spans:
        candidate = current + [span]
        if current and count(text[candidate[0][0] : span[1]]) > max_tokens:
            groups.append(current)
            # Overlap: trailing sentences of the previous chunk, up to overlap_tokens.
            overlap: list[tuple[int, int]] = []
            for prev in reversed(current):
                if count(text[prev[0] : current[-1][1]]) > overlap_tokens:
                    break
                overlap.insert(0, prev)
            current = overlap + [span]
            if count(text[current[0][0] : span[1]]) > max_tokens:
                current = [span]
        else:
            current = candidate
    if current:
        groups.append(current)

    chunks: list[Chunk] = []
    skipped = 0
    for group in groups:
        start, end = group[0][0], group[-1][1]
        piece = text[start:end]
        if is_near_empty(piece):
            skipped += 1
            continue
        chunks.append(Chunk(len(chunks), start, end, piece))
    return chunks, skipped
