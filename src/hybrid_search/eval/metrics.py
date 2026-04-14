from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Qrel:
    qid: str
    doc_id: int
    relevance: int


def average_precision(ranked_doc_ids: list[int], relevant: set[int]) -> float:
    if not relevant:
        return 0.0
    hit = 0
    s = 0.0
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            hit += 1
            s += hit / i
    return s / len(relevant)


def mean_average_precision(run: dict[str, list[int]], qrels: dict[str, dict[int, int]]) -> float:
    aps: list[float] = []
    for qid, ranked in run.items():
        rel = {doc_id for doc_id, r in qrels.get(qid, {}).items() if r > 0}
        aps.append(average_precision(ranked, rel))
    if not aps:
        return 0.0
    return sum(aps) / len(aps)


def dcg_at_k(ranked_doc_ids: list[int], rels: dict[int, int], k: int) -> float:
    s = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        r = int(rels.get(doc_id, 0))
        if r <= 0:
            continue
        s += (2**r - 1) / math.log2(i + 1)
    return s


def ndcg_at_k(ranked_doc_ids: list[int], rels: dict[int, int], k: int) -> float:
    dcg = dcg_at_k(ranked_doc_ids, rels, k)
    ideal = sorted(rels.values(), reverse=True)
    ideal_dcg = 0.0
    for i, r in enumerate(ideal[:k], start=1):
        if r <= 0:
            continue
        ideal_dcg += (2**r - 1) / math.log2(i + 1)
    if ideal_dcg <= 0.0:
        return 0.0
    return dcg / ideal_dcg


def mean_ndcg_at_k(run: dict[str, list[int]], qrels: dict[str, dict[int, int]], k: int) -> float:
    vals: list[float] = []
    for qid, ranked in run.items():
        vals.append(ndcg_at_k(ranked, qrels.get(qid, {}), k))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)

