import math

import pytest
import torch

from hybrid_retrieval.dense import DenseIndex, HashingDenseEncoder, cosine_similarity
from hybrid_retrieval.types import Document


def test_cosine_matches_hand_computed_value_and_handles_zero():
    left = torch.tensor([1.0, 0.0])
    right = torch.tensor([1.0, 1.0])
    assert cosine_similarity(left, right) == pytest.approx(2**-0.5)
    assert cosine_similarity(left, torch.zeros(2)) == 0.0


def test_dense_encoding_is_deterministic_normalized_and_sublinear_tf_weighted():
    encoder = HashingDenseEncoder(dimension=32, seed=9)
    one = encoder.encode("Alpha beta alpha")
    two = encoder.encode("alpha beta alpha")
    assert torch.equal(one, two)
    assert torch.linalg.vector_norm(one).item() == pytest.approx(1.0)
    expected = (1 + math.log(2)) * encoder.token_vector("alpha")
    expected += encoder.token_vector("beta")
    expected /= torch.linalg.vector_norm(expected)
    assert torch.equal(one, expected)
    other_seed = HashingDenseEncoder(dimension=32, seed=10).encode("alpha beta")
    assert not torch.equal(one, other_seed)


def test_dense_index_performs_exact_cosine_ranking():
    encoder = HashingDenseEncoder(dimension=64, seed=5)
    documents = [
        Document("target", "quantum entanglement"),
        Document("other", "bread baking"),
        Document("empty", ""),
    ]
    index = DenseIndex(documents, encoder)
    result = index.search("quantum entanglement", top_k=3)
    assert result[0].doc_id == "target"
    assert result[0].score == pytest.approx(1.0)
    assert result[-1].score == 0.0
    expected = index.embeddings @ encoder.encode("quantum entanglement")
    by_id = {item.doc_id: item.score for item in result}
    for position, document in enumerate(documents):
        assert by_id[document.doc_id] == pytest.approx(float(expected[position]))
