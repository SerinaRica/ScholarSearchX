from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hybrid_search.index.inverted_index import InvertedIndex
from hybrid_search.kg.graph import KgGraph
from hybrid_search.preprocess.text import TextPreprocessor

ExpandMethod = Literal["rocchio", "kg"]


@dataclass(frozen=True)
class ExpansionInfo:
    methods: list[ExpandMethod]
    added_terms: list[str]


def expand_query(
    *,
    index: InvertedIndex,
    query: str,
    top_doc_ids: list[int],
    methods: list[ExpandMethod],
    kg_dict_path: Path | None = None,
    kg_graph_path: Path | None = None,
    rocchio_top_docs: int = 5,
    rocchio_add_terms: int = 8,
    kg_add_terms: int = 10,
) -> tuple[str, ExpansionInfo]:
    pre = TextPreprocessor()
    base_terms = pre.preprocess(query)
    added: list[str] = []

    if "rocchio" in methods:
        doc_ids = top_doc_ids[: int(rocchio_top_docs)]
        counter: Counter[str] = Counter()
        for doc_id in doc_ids:
            counter.update(dict(index.get_doc_top_terms(doc_id)))
        for term, _ in counter.most_common(int(rocchio_add_terms) * 2):
            if term not in base_terms and term not in added:
                added.append(term)
            if len(added) >= int(rocchio_add_terms):
                break

    if "kg" in methods:
        kg_dict = _load_kg_dict(kg_dict_path)
        kg_graph = _load_kg_graph(kg_graph_path)
        raw_tokens = query.lower().split()
        for tok in raw_tokens:
            ex = kg_dict.get(tok)
            if ex:
                for t in pre.preprocess(" ".join(ex)):
                    if t not in base_terms and t not in added:
                        added.append(t)

        if kg_graph is not None:
            candidate_terms: Counter[str] = Counter()
            for t in base_terms:
                for nb, w in kg_graph.neighbors(t, top_n=30):
                    if nb in base_terms or nb in added:
                        continue
                    candidate_terms[nb] += float(w)
            for nb, _w in candidate_terms.most_common(int(kg_add_terms) * 2):
                if nb not in base_terms and nb not in added:
                    added.append(nb)
                if len(added) >= int(rocchio_add_terms) + int(kg_add_terms):
                    break

    expanded_terms = base_terms + added
    expanded_query = " ".join(expanded_terms)
    return expanded_query, ExpansionInfo(methods=list(methods), added_terms=added)


def _load_kg_dict(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        default = Path("data") / "kg_dict.json"
        path = default if default.exists() else None
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, list[str]] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            out[str(k).lower()] = [str(x) for x in v]
        else:
            out[str(k).lower()] = [str(v)]
    return out


def _load_kg_graph(path: Path | None) -> KgGraph | None:
    if path is None:
        default = Path("data") / "kg" / "graph.pkl"
        path = default if default.exists() else None
    if path is None or not path.exists():
        return None
    return KgGraph.load(path)
