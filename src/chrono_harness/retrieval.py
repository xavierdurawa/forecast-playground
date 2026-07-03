"""Chunk + rank utilities for keeping retrieved text relevant and context-sized.

Returning whole articles overflows the model's context and buries the relevant
passage. These helpers split text into passages and rank them against the query by
keyword overlap — dependency-free and deterministic (no embeddings), which keeps
the harness light and the behaviour testable offline. A consumer that wants
semantic ranking can swap in its own scorer.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")
# Very common words contribute no relevance signal.
_STOP = frozenset(
    "the a an and or of to in on for is are was were be been being this that these "
    "those it its as at by with from will would can could may might has have had do "
    "does did but not no if then than into about over under between".split()
)


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP]


def chunk_text(text: str, chunk_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into overlapping character windows on paragraph boundaries.

    Splits on blank lines first, then packs paragraphs into ~``chunk_chars`` windows.
    Overlap carries a little trailing context into the next chunk so a passage split
    across a boundary is still scorable.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            # Start the next buffer with a tail of the previous for continuity.
            tail = buf[-overlap:] if buf and overlap else ""
            buf = f"{tail}\n\n{p}" if tail else p
            # A single oversized paragraph is hard-split.
            while len(buf) > chunk_chars:
                chunks.append(buf[:chunk_chars])
                buf = buf[chunk_chars - overlap :]
    if buf:
        chunks.append(buf)
    return chunks


def score_chunk(chunk: str, query_tokens: list[str]) -> float:
    """Relevance score: query-token hits in the chunk, normalized by chunk length.

    Length-normalization stops long chunks from winning purely by size.
    """
    if not query_tokens:
        return 0.0
    ctoks = _tokens(chunk)
    if not ctoks:
        return 0.0
    qset = set(query_tokens)
    hits = sum(1 for t in ctoks if t in qset)
    return hits / (len(ctoks) ** 0.5)


def top_chunks(
    text: str,
    query: str,
    k: int = 3,
    chunk_chars: int = 1500,
) -> list[str]:
    """Return the ``k`` chunks of ``text`` most relevant to ``query``.

    Falls back to the leading chunks if nothing scores (e.g. query terms absent),
    so the model always gets *something* rather than an empty result.
    """
    chunks = chunk_text(text, chunk_chars=chunk_chars)
    if not chunks:
        return []
    qtoks = _tokens(query)
    scored = sorted(chunks, key=lambda c: score_chunk(c, qtoks), reverse=True)
    best = scored[:k]
    if all(score_chunk(c, qtoks) == 0.0 for c in best):
        return chunks[:k]  # nothing matched — give the intro instead
    return best
