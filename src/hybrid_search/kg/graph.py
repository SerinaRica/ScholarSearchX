from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from hybrid_search.index.inverted_index import InvertedIndex


@dataclass(frozen=True)
class KgGraph:
    adjacency: dict[str, list[tuple[str, float]]]

    def neighbors(self, term: str, *, top_n: int = 12) -> list[tuple[str, float]]:
        items = self.adjacency.get(term, [])
        return items[: int(top_n)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> KgGraph:
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, KgGraph):
            raise TypeError("invalid KG graph file")
        return obj


def build_cooccurrence_graph(
    *,
    index: InvertedIndex,
    min_df: int = 3,
    per_doc_terms: int = 30,
    max_neighbors: int = 30,
) -> KgGraph:
    vocab = index.vocabulary
    filtered_vocab = {t for t in vocab if index.get_df(t) >= int(min_df)}

    weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for doc_id in index.all_doc_ids():
        terms = [t for t, _tf in index.get_doc_top_terms(doc_id)[: int(per_doc_terms)]]
        terms = [t for t in terms if t in filtered_vocab]
        uniq = list(dict.fromkeys(terms))
        for i in range(len(uniq)):
            a = uniq[i]
            for j in range(i + 1, len(uniq)):
                b = uniq[j]
                weights[a][b] += 1.0
                weights[b][a] += 1.0

    adjacency: dict[str, list[tuple[str, float]]] = {}
    for term, nbrs in weights.items():
        ranked = sorted(nbrs.items(), key=lambda x: x[1], reverse=True)
        adjacency[term] = ranked[: int(max_neighbors)]

    return KgGraph(adjacency=adjacency)
