from hybrid_search.index.inverted_index import DocMeta, InvertedIndex
from hybrid_search.search.tfidf_ranker import tfidf_search


def test_tfidf_ranking_prefers_matching_doc():
    idx = InvertedIndex()
    idx.add_document(
        DocMeta(
            doc_id=0,
            title="A",
            abstract="knowledge graph reasoning",
            authors=[],
            categories=[],
            url="",
        )
    )
    idx.add_document(
        DocMeta(
            doc_id=1,
            title="B",
            abstract="computer vision segmentation",
            authors=[],
            categories=[],
            url="",
        )
    )
    idx.finalize()

    results, _ = tfidf_search(idx, "knowledge graph", top_k=10)
    assert results
    assert results[0].doc_id == 0

