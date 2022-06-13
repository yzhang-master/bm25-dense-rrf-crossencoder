import math

import pytest

from hybrid_retrieval.bm25 import BM25Index, tokenize
from hybrid_retrieval.types import Document


def test_tokenize_unicode_casefold_and_punctuation():
    assert tokenize("Straße, CAFÉ_v2! 42") == ["strasse", "café", "v2", "42"]


def test_bm25_matches_hand_computed_robertson_formula():
    index = BM25Index(
        [Document("long", "a a b"), Document("short", "a")], k1=1.5, b=0.75
    )
    idf = math.log1p(0.5 / 2.5)
    expected_long = idf * (2 * 2.5) / (2 + 1.5 * 1.375)
    expected_short = idf * 2.5 / (1 + 1.5 * 0.625)
    assert index.score_document("a", 0) == pytest.approx(expected_long)
    assert index.score_document("a", 1) == pytest.approx(expected_short)
    assert [item.doc_id for item in index.search("a")] == ["short", "long"]


def test_length_normalization_is_active_and_term_frequency_saturates():
    documents = [
        Document("short", "topic concise"),
        Document("long", "topic padding padding padding padding"),
    ]
    normalized = BM25Index(documents, b=0.75)
    unnormalized = BM25Index(documents, b=0)
    assert normalized.score_document("topic", 0) > normalized.score_document("topic", 1)
    assert unnormalized.score_document("topic", 0) == pytest.approx(
        unnormalized.score_document("topic", 1)
    )


def test_duplicate_query_terms_do_not_multiply_score_and_ties_are_stable():
    index = BM25Index([Document("b", "same"), Document("a", "same")])
    assert index.score_document("same same", 0) == pytest.approx(index.score_document("same", 0))
    assert [item.doc_id for item in index.search("same")] == ["a", "b"]


def test_invalid_corpus_and_parameters_are_rejected():
    with pytest.raises(ValueError):
        BM25Index([])
    with pytest.raises(ValueError):
        BM25Index([Document("a", "same"), Document("a", "other")])
    with pytest.raises(ValueError):
        BM25Index([Document("a", "same")], k1=0)
    with pytest.raises(ValueError):
        BM25Index([Document("a", "same")], b=1.1)
    with pytest.raises(ValueError):
        BM25Index([Document("a", "same")]).search("same", top_k=0)
