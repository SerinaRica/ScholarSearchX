from __future__ import annotations

import math
from dataclasses import dataclass

from hybrid_search.index.inverted_index import InvertedIndex
from hybrid_search.preprocess.text import TextPreprocessor


@dataclass(frozen=True)
class ScoredDoc:
    doc_id: int
    score: float


def tfidf_search(
    index: InvertedIndex, query: str, *, top_k: int = 100
) -> tuple[list[ScoredDoc], list[str]]:
    pre = TextPreprocessor()
    q_terms = pre.preprocess(query)
    if not q_terms:
        return [], []
    n = max(index.doc_count, 1)

    q_tf: dict[str, int] = {}
    for t in q_terms:
        q_tf[t] = q_tf.get(t, 0) + 1

    q_weights: dict[str, float] = {}
    for t, tf in q_tf.items():
        df = index.get_df(t)
        if df <= 0:
            continue
        idf = math.log(n / df)
        q_weights[t] = (1.0 + math.log(tf)) * idf

    if not q_weights:
        return [], q_terms

    q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
    scores: dict[int, float] = {}
    for term, wq in q_weights.items():
        df = index.get_df(term)
        if df <= 0:
            continue
        idf = math.log(n / df)
        postings = index.get_postings(term)
        for doc_id, tf in postings:
            wd = (1.0 + math.log(tf)) * idf
            scores[doc_id] = scores.get(doc_id, 0.0) + (wq * wd)

    results: list[ScoredDoc] = []
    for doc_id, dot in scores.items():
        d_norm = index.get_doc_norm(doc_id)
        if d_norm <= 0.0 or q_norm <= 0.0:
            continue
        results.append(ScoredDoc(doc_id=doc_id, score=float(dot / (d_norm * q_norm))))
    results.sort(key=lambda x: x.score, reverse=True)
    return results[: int(top_k)], q_terms
