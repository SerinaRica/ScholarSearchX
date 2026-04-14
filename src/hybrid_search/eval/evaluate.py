from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hybrid_search.eval.metrics import (
    average_precision,
    mean_average_precision,
    mean_ndcg_at_k,
    ndcg_at_k,
)
from hybrid_search.index.inverted_index import InvertedIndex
from hybrid_search.search.query_expand import expand_query
from hybrid_search.search.tfidf_ranker import tfidf_search


@dataclass(frozen=True)
class EvalQuery:
    qid: str
    query: str


def load_queries(path: Path) -> list[EvalQuery]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: list[EvalQuery] = []
    for item in raw:
        out.append(EvalQuery(qid=str(item["qid"]), query=str(item["query"])))
    return out


def load_qrels_tsv(path: Path) -> dict[str, dict[int, int]]:
    qrels: dict[str, dict[int, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            qid, doc_id, rel = line.split("\t")
            qrels.setdefault(qid, {})[int(doc_id)] = int(rel)
    return qrels


def evaluate(
    *,
    index_dir: Path,
    queries_path: Path,
    qrels_path: Path,
    top_k: int = 100,
    ndcg_k: int = 10,
) -> dict[str, float]:
    index = InvertedIndex.load(index_dir)
    queries = load_queries(queries_path)
    qrels = load_qrels_tsv(qrels_path)

    run_base: dict[str, list[int]] = {}
    run_expand: dict[str, list[int]] = {}

    all_doc_ids = index.all_doc_ids()
    qrels_in_index: dict[str, dict[int, int]] = {}
    for qid, rels in qrels.items():
        qrels_in_index[qid] = {doc_id: r for doc_id, r in rels.items() if doc_id in all_doc_ids}

    for q in queries:
        base, _ = tfidf_search(index, q.query, top_k=top_k)
        run_base[q.qid] = [d.doc_id for d in base]

        expanded_query, _ = expand_query(
            index=index,
            query=q.query,
            top_doc_ids=run_base[q.qid][:10],
            methods=["rocchio", "kg"],
        )
        expanded, _ = tfidf_search(index, expanded_query, top_k=top_k)
        run_expand[q.qid] = [d.doc_id for d in expanded]

    per_query: dict[str, dict[str, float | int]] = {}
    for q in queries:
        rels = qrels.get(q.qid, {})
        rels_idx = qrels_in_index.get(q.qid, {})
        relevant = {doc_id for doc_id, r in rels_idx.items() if r > 0}
        base_ranked = run_base.get(q.qid, [])
        exp_ranked = run_expand.get(q.qid, [])

        per_query[q.qid] = {
            "judged": len(rels),
            "judged_in_index": len(rels_idx),
            "relevant": len({doc_id for doc_id, r in rels.items() if r > 0}),
            "relevant_in_index": len(relevant),
            "base.AP@100": average_precision(base_ranked[:top_k], relevant),
            f"base.NDCG@{ndcg_k}": ndcg_at_k(base_ranked, rels_idx, ndcg_k),
            "expand.AP@100": average_precision(exp_ranked[:top_k], relevant),
            f"expand.NDCG@{ndcg_k}": ndcg_at_k(exp_ranked, rels_idx, ndcg_k),
            "base.hits@10": len(set(base_ranked[:10]).intersection(relevant)),
            "expand.hits@10": len(set(exp_ranked[:10]).intersection(relevant)),
        }

    qrels_total = sum(len(v) for v in qrels.values())
    qrels_total_in_index = sum(len(v) for v in qrels_in_index.values())
    pos_total = sum(1 for rels in qrels.values() for r in rels.values() if r > 0)
    pos_total_in_index = sum(1 for rels in qrels_in_index.values() for r in rels.values() if r > 0)

    return {
        "MAP@100.base": mean_average_precision(run_base, qrels_in_index),
        f"NDCG@{ndcg_k}.base": mean_ndcg_at_k(run_base, qrels_in_index, ndcg_k),
        "MAP@100.expand": mean_average_precision(run_expand, qrels_in_index),
        f"NDCG@{ndcg_k}.expand": mean_ndcg_at_k(run_expand, qrels_in_index, ndcg_k),
        "qrels.judged_total": qrels_total,
        "qrels.judged_in_index": qrels_total_in_index,
        "qrels.relevant_total": pos_total,
        "qrels.relevant_in_index": pos_total_in_index,
        "per_query": per_query,
    }
