"""BM25, implemented directly rather than imported.

Written out because the parameters are the point of the exercise. ``k1`` and
``b`` are what a benchmark should be sweeping, and a black-box dependency
invites treating them as magic constants:

* **k1** controls term-frequency saturation. A document containing a query term
  twenty times is not twenty times more relevant than one containing it once,
  and k1 sets how quickly that extra evidence stops counting.
* **b** controls length normalisation. At b=1 a long document is fully
  penalised for its length; at b=0 not at all. This matters enormously here,
  because chunking *is* length manipulation - the same corpus split three ways
  produces three different length distributions.

The scoring function is the Robertson/Sparck-Jones formulation with the
standard +0.5 smoothing in the IDF term.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..corpus import Chunk
from .base import Retriever, ScoredChunk

TOKEN = re.compile(r"[a-z0-9]+")

# Removing these costs nothing and stops IDF from being dominated by words that
# appear in every document. Kept short and explicit rather than pulling in a
# stopword corpus, so the list is auditable.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "as",
        "it",
        "its",
        "into",
        "about",
        "over",
        "under",
        "between",
        "do",
        "does",
        "did",
        "done",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "will",
        "shall",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "them",
        "his",
        "her",
        "their",
        "our",
        "your",
        "my",
        "me",
        "us",
        "not",
        "no",
        "yes",
    ]
)


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    tokens = TOKEN.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS] if remove_stopwords else tokens


class BM25Retriever(Retriever):
    """Lexical retrieval. Strong on exact terms, blind to synonyms.

    Its failure mode is the mirror image of dense retrieval's: it will not
    connect "how do I turn on two-factor auth" to a document that only ever
    says "enable 2FA". That complementarity is why the hybrid retriever exists.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, remove_stopwords: bool = True) -> None:
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        self.name = f"bm25(k1={k1},b={b})"

        self._chunks: list[Chunk] = []
        self._term_frequencies: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._average_length = 0.0
        self._idf: dict[str, float] = {}

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._term_frequencies = [
            Counter(tokenize(c.text, remove_stopwords=self.remove_stopwords)) for c in chunks
        ]
        self._lengths = [sum(tf.values()) for tf in self._term_frequencies]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

        document_frequency: Counter[str] = Counter()
        for tf in self._term_frequencies:
            document_frequency.update(tf.keys())

        total = len(chunks)
        # The +0.5 smoothing keeps IDF finite for a term present in every
        # document; without it such a term scores -inf and poisons the sum.
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if not self._chunks:
            raise RuntimeError("index() must be called before search()")

        terms = tokenize(query, remove_stopwords=self.remove_stopwords)
        scores: list[tuple[float, int]] = []

        for position, tf in enumerate(self._term_frequencies):
            length = self._lengths[position]
            norm = (
                1 - self.b + self.b * length / self._average_length if self._average_length else 1.0
            )
            score = 0.0
            for term in terms:
                frequency = tf.get(term)
                if not frequency:
                    continue
                score += self._idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1) / (frequency + self.k1 * norm)
                )
            if score > 0:
                scores.append((score, position))

        scores.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            ScoredChunk(chunk=self._chunks[position], score=score) for score, position in scores[:k]
        ]
