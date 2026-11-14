from clause_graph.chunking import (
    chunk_clause,
    collapse,
    estimate_tokens,
    sentence_spans,
)


def sentences(n: int, words: int = 20) -> str:
    """n distinct sentences of `words` words each (~5 chars per word)."""
    return " ".join(f"Sentence {i} " + " ".join(["word"] * (words - 2)) + "." for i in range(n))


def test_short_clause_is_one_chunk():
    text = "6.1    A Party shall be liable to the other for loss of Freight."
    chunks, skipped = chunk_clause(text)
    assert skipped == 0
    assert [(c.idx, c.start, c.end, c.text) for c in chunks] == [(0, 0, len(text), text)]


def test_bare_heading_is_skipped_and_counted():
    assert chunk_clause("9.0    INSURANCE") == ([], 1)
    assert chunk_clause("ARTICLE VII LIMITATION OF LIABILITY") == ([], 1)


def test_long_clause_splits_at_sentence_boundaries_within_budget():
    text = sentences(60)  # ~60 * 25 tokens = ~1,500+ tokens
    chunks, skipped = chunk_clause(text, max_tokens=300, overlap_tokens=40)
    assert skipped == 0 and len(chunks) > 1
    for c in chunks:
        assert estimate_tokens(c.text) <= 300
        assert c.text == text[c.start : c.end]  # provenance: offsets into the clause
        assert c.text.startswith("Sentence ") and c.text.endswith(".")
    assert [c.idx for c in chunks] == list(range(len(chunks)))
    # Every sentence is in at least one chunk.
    covered = " ".join(c.text for c in chunks)
    assert all(f"Sentence {i} " in covered for i in range(60))


def test_chunks_overlap_by_trailing_sentences():
    text = sentences(40)
    chunks, _ = chunk_clause(text, max_tokens=200, overlap_tokens=40)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start < prev.end  # overlap exists
        assert estimate_tokens(text[nxt.start : prev.end]) <= 40


def test_sentence_longer_than_budget_is_cut_at_whitespace():
    text = "Table " + " ".join(f"cell{i}" for i in range(2000))  # one "sentence", no periods
    spans = sentence_spans(text, max_tokens=100)
    assert len(spans) > 1
    assert all(estimate_tokens(text[s:e]) <= 100 for s, e in spans)
    assert all(not text[s].isspace() for s, _ in spans)


def test_token_estimate_uses_collapsed_whitespace():
    assert collapse("a    b\n\n c") == "a b c"
    assert estimate_tokens("abcd" + " " * 100 + "efgh") == 3  # "abcd efgh" = 9 chars


def test_text_hash_identifies_text():
    from clause_graph.chunking import text_hash

    assert text_hash("9.1 Cover.") == text_hash("9.1 Cover.")
    assert text_hash("9.1 Cover.") != text_hash("9.1 Cover. ")  # any change forces re-embedding
    assert len(text_hash("x")) == 16


def test_custom_counter_bounds_chunks_by_real_tokens():
    # A counter that sees 2x the estimate (dense text): chunks must still fit the budget.
    def dense(text: str) -> int:
        return 2 * estimate_tokens(text)

    text = sentences(60)
    v1, _ = chunk_clause(text, max_tokens=300, overlap_tokens=40)
    v2, _ = chunk_clause(text, max_tokens=300, overlap_tokens=40, count=dense)
    assert all(dense(c.text) <= 300 for c in v2)
    assert any(dense(c.text) > 300 for c in v1)  # what v1 would have produced
    assert all(c.text == text[c.start : c.end] for c in v2)


def test_long_sentence_cut_respects_real_counter():
    def dense(text: str) -> int:
        return 2 * estimate_tokens(text)

    text = "Table " + " ".join(f"cell{i}" for i in range(2000))
    spans = sentence_spans(text, max_tokens=100, count=dense)
    assert all(dense(text[s:e]) <= 100 for s, e in spans)
    assert "".join(text[s:e] for s, e in spans).replace(" ", "") == text.replace(" ", "")


def test_chunker_version_is_kept_for_existing_contracts():
    from clause_graph.chunking import CURRENT_CHUNKER, chunker_version

    existing = {"DYNAMEX": 1}
    assert chunker_version("DYNAMEX", existing) == 1  # reload keeps v1 and its embeddings
    assert chunker_version("NEWCO", existing) == CURRENT_CHUNKER
    assert chunker_version("DYNAMEX", existing, rechunk="DYN") == CURRENT_CHUNKER
    assert chunker_version("DYNAMEX", existing, rechunk="WEST") == 1


def test_real_token_count_matches_ollama():
    import pytest

    from clause_graph.embeddings import Config, EmbeddingError, count_tokens

    try:
        n = count_tokens("9.1 Cover.", Config("bge-m3", ""))
    except EmbeddingError as e:
        pytest.skip(f"bge-m3 tokenizer not available: {e}")
    assert n == 6  # 4 text tokens + <s> </s>, as counted by Ollama's prompt_eval_count
