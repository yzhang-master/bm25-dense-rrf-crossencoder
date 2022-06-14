import hashlib
import math
from collections import Counter

import torch

from .bm25 import tokenize
from .types import Document, RankedDocument, validate_documents


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("cosine inputs must be equal-sized vectors")
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if denominator.item() == 0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


class HashingDenseEncoder:
    """Dense sublinear-TF vectors built from seeded random token embeddings."""

    def __init__(self, dimension: int = 128, seed: int = 17) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.seed = seed
        self._cache: dict[str, torch.Tensor] = {}

    def token_vector(self, token: str) -> torch.Tensor:
        if token not in self._cache:
            digest = hashlib.blake2b(
                f"{self.seed}:{token}".encode(), digest_size=8
            ).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(digest, "big"))
            self._cache[token] = torch.randn(self.dimension, generator=generator)
        return self._cache[token]

    def encode(self, text: str) -> torch.Tensor:
        counts = Counter(tokenize(text))
        vector = torch.zeros(self.dimension)
        for token, count in counts.items():
            vector += (1 + math.log(count)) * self.token_vector(token)
        norm = torch.linalg.vector_norm(vector)
        if norm.item() > 0:
            vector = vector / norm
        return vector


class DenseIndex:
    """Exact cosine search over a dense matrix; no approximate index is hidden here."""

    def __init__(self, documents: list[Document], encoder: HashingDenseEncoder) -> None:
        validate_documents(documents)
        self.documents = tuple(documents)
        self.encoder = encoder
        self.embeddings = torch.stack([encoder.encode(document.text) for document in documents])

    def search(self, query: str, top_k: int = 10) -> list[RankedDocument]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = self.encoder.encode(query)
        similarities = self.embeddings @ query_vector
        scored = [
            RankedDocument(document.doc_id, float(similarities[index]))
            for index, document in enumerate(self.documents)
        ]
        return sorted(scored, key=lambda item: (-item.score, item.doc_id))[:top_k]
