"""Tests for chunk + rank retrieval helpers."""

from forecast_playground.retrieval import chunk_text, score_chunk, top_chunks


def test_chunk_text_splits_and_respects_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 50 for i in range(10))
    chunks = chunk_text(text, chunk_chars=400, overlap=50)
    assert len(chunks) > 1
    # No chunk wildly exceeds the target (allow overlap slack).
    assert all(len(c) <= 400 + 60 for c in chunks)


def test_top_chunks_ranks_relevant_passage_first():
    text = (
        "The history of cheese is long.\n\n"
        "SpaceX Starship reached orbit during a test flight.\n\n"
        "Bananas are yellow fruit."
    )
    out = top_chunks(text, "Starship orbit flight", k=1)
    assert "Starship" in out[0]


def test_top_chunks_falls_back_when_no_match():
    text = "alpha beta\n\ngamma delta"
    # Query terms absent -> return leading chunks rather than nothing.
    out = top_chunks(text, "zzzzz qqqqq", k=1)
    assert len(out) == 1


def test_score_chunk_zero_for_empty_query():
    assert score_chunk("some text here", []) == 0.0
