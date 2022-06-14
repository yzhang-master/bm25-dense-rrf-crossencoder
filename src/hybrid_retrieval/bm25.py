import math
import re
from collections import Counter

from .types import Document, RankedDocument, validate_documents

_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase and split Unicode letter/number sequences, dropping underscores."""
    return [match.group(0).casefold() for match in _TOKEN.finditer(text)]


class BM25Index:
    """Exact Robertson/Sparck Jones BM25 with Lucene's positive idf transform."""

    def __init__(
        self,
        documents: list[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        validate_documents(documents)
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.documents = tuple(documents)
        self.k1 = float(k1)
        self.b = float(b)
        self._tokens = [tokenize(document.text) for document in documents]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / len(documents)
        if self._average_length == 0:
            raise ValueError("the corpus must contain at least one token")
        self._document_frequency = Counter(
            term for tokens in self._tokens for term in set(tokens)
        )

    def idf(self, term: str) -> float:
        n_docs = len(self.documents)
        frequency = self._document_frequency.get(term, 0)
        return math.log1p((n_docs - frequency + 0.5) / (frequency + 0.5))

    def score_document(self, query: str, doc_index: int) -> float:
        term_frequencies = self._term_frequencies[doc_index]
        length_normalization = 1 - self.b + self.b * (
            self._lengths[doc_index] / self._average_length
        )
        score = 0.0
        for term in set(tokenize(query)):
            frequency = term_frequencies.get(term, 0)
            if frequency == 0:
                continue
            saturation = frequency + self.k1 * length_normalization
            score += self.idf(term) * frequency * (self.k1 + 1) / saturation
        return score

    def search(self, query: str, top_k: int = 10) -> list[RankedDocument]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scored = [
            RankedDocument(document.doc_id, self.score_document(query, index))
            for index, document in enumerate(self.documents)
        ]
        return sorted(scored, key=lambda item: (-item.score, item.doc_id))[:top_k]
