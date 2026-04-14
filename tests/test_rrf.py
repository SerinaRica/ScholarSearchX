from hybrid_search.rag.hybrid import rrf_fuse
from hybrid_search.search.tfidf_ranker import ScoredDoc
from hybrid_search.vector.qdrant_store import DenseHit


def test_rrf_fuse_prefers_consensus():
    sparse = [ScoredDoc(doc_id=1, score=1.0), ScoredDoc(doc_id=2, score=0.9)]
    dense = [DenseHit(doc_id=2, score=0.8), DenseHit(doc_id=3, score=0.7)]
    kg = [2, 4]
    fused = rrf_fuse(sparse=sparse, dense=dense, kg=kg, top_n=10)
    assert fused[0].doc_id == 2
    assert "sparse" in fused[0].sources
    assert "dense" in fused[0].sources
    assert "kg" in fused[0].sources

