from hybrid_search.index.inverted_index import DocMeta, InvertedIndex
from hybrid_search.kg.graph import build_cooccurrence_graph


def test_kg_graph_neighbors():
    idx = InvertedIndex()
    idx.add_document(
        DocMeta(
            doc_id=0,
            title="A",
            abstract="knowledge graph reasoning knowledge graph",
            authors=[],
            categories=[],
            url="",
        )
    )
    idx.add_document(
        DocMeta(
            doc_id=1,
            title="B",
            abstract="knowledge graph retrieval",
            authors=[],
            categories=[],
            url="",
        )
    )
    idx.finalize()
    kg = build_cooccurrence_graph(index=idx, min_df=1, per_doc_terms=10, max_neighbors=10)
    nbs = dict(kg.neighbors("knowledg", top_n=10))
    assert "graph" in nbs

