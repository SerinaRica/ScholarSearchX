import pytest

from hybrid_search.index.inverted_index import DocMeta, InvertedIndex
from hybrid_search.search.boolean_query import boolean_retrieve


def _index():
    idx = InvertedIndex()
    idx.add_document(
        DocMeta(
            doc_id=0,
            title="A",
            abstract="llm reasoning with knowledge graph",
            authors=[],
            categories=[],
            url="",
        )
    )
    idx.add_document(
        DocMeta(
            doc_id=1,
            title="B",
            abstract="computer vision with transformers",
            authors=[],
            categories=[],
            url="",
        )
    )
    idx.finalize()
    return idx


def test_boolean_query_basic():
    idx = _index()
    res = boolean_retrieve(idx, "LLM AND (Reasoning OR Graph) NOT Vision")
    assert res.doc_ids == [0]


def test_boolean_query_mismatch_parentheses():
    idx = _index()
    with pytest.raises(ValueError):
        boolean_retrieve(idx, "LLM AND (Reasoning OR Graph")

