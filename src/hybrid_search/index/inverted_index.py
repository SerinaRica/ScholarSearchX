from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hybrid_search.index.compress_vbyte import decode_postings_vb, encode_postings_vb
from hybrid_search.preprocess.text import TextPreprocessor

Compression = Literal["none", "vbyte"]


@dataclass(frozen=True)
class DocMeta:
    doc_id: int
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    url: str

    @staticmethod
    def from_dict(d: dict) -> DocMeta:
        return DocMeta(
            doc_id=int(d["doc_id"]),
            title=str(d.get("title", "")),
            abstract=str(d.get("abstract", "")),
            authors=list(d.get("authors", [])),
            categories=list(d.get("categories", [])),
            url=str(d.get("url", "")),
        )


class InvertedIndex:
    def __init__(self, *, compression: Compression = "none", top_terms_per_doc: int = 200):
        self.compression: Compression = compression
        self.preprocessor = TextPreprocessor()
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._df: dict[str, int] = {}
        self._doc_count: int = 0
        self._docs: dict[int, DocMeta] = {}
        self._doc_norms: dict[int, float] = {}
        self._doc_top_terms: dict[int, list[tuple[str, int]]] = {}
        self._top_terms_per_doc = int(top_terms_per_doc)

        self._postings_bytes: dict[str, bytes] = {}

    @property
    def doc_count(self) -> int:
        return self._doc_count

    @property
    def vocabulary(self) -> set[str]:
        return set(self._df.keys())

    def all_doc_ids(self) -> set[int]:
        return set(self._docs.keys())

    def get_doc(self, doc_id: int) -> DocMeta:
        return self._docs[doc_id]

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        if self.compression == "vbyte":
            data = self._postings_bytes.get(term)
            if not data:
                return []
            doc_ids, tfs = decode_postings_vb(data)
            return list(zip(doc_ids, tfs, strict=True))
        return list(self._postings.get(term, []))

    def get_df(self, term: str) -> int:
        return int(self._df.get(term, 0))

    def get_doc_norm(self, doc_id: int) -> float:
        return float(self._doc_norms.get(doc_id, 0.0))

    def get_doc_top_terms(self, doc_id: int) -> list[tuple[str, int]]:
        return list(self._doc_top_terms.get(doc_id, []))

    def add_document(self, doc: DocMeta) -> None:
        tokens = self.preprocessor.preprocess(doc.abstract)
        tf = Counter(tokens)
        unique_terms = set(tf.keys())

        for term in unique_terms:
            self._postings[term].append((doc.doc_id, int(tf[term])))

        self._docs[doc.doc_id] = doc
        self._doc_top_terms[doc.doc_id] = tf.most_common(self._top_terms_per_doc)
        self._doc_count = max(self._doc_count, doc.doc_id + 1)

    def finalize(self) -> None:
        self._df = {t: len(p) for t, p in self._postings.items()}
        for _term, plist in self._postings.items():
            plist.sort(key=lambda x: x[0])

        self._doc_norms = self._compute_doc_norms()

        if self.compression == "vbyte":
            self._postings_bytes = {}
            for term, plist in self._postings.items():
                doc_ids = [d for d, _ in plist]
                tfs = [tf for _, tf in plist]
                self._postings_bytes[term] = encode_postings_vb(doc_ids, tfs)
            self._postings = defaultdict(list)

    def _compute_doc_norms(self) -> dict[int, float]:
        norms_sq: dict[int, float] = defaultdict(float)
        n = max(self._doc_count, 1)
        for _term, plist in self._postings.items():
            df = len(plist)
            if df == 0:
                continue
            idf = math.log(n / df)
            for doc_id, tf in plist:
                w = (1.0 + math.log(tf)) * idf
                norms_sq[doc_id] += w * w
        return {doc_id: math.sqrt(v) for doc_id, v in norms_sq.items()}

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "compression": self.compression,
            "doc_count": self._doc_count,
            "df": self._df,
            "doc_norms": self._doc_norms,
            "docs": {k: self._docs[k].__dict__ for k in sorted(self._docs.keys())},
            "doc_top_terms": self._doc_top_terms,
        }
        if self.compression == "vbyte":
            payload["postings_bytes"] = self._postings_bytes
        else:
            payload["postings"] = dict(self._postings)
        with (index_dir / "index.pkl").open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(index_dir: Path) -> InvertedIndex:
        with (index_dir / "index.pkl").open("rb") as f:
            payload = pickle.load(f)
        idx = InvertedIndex(compression=payload["compression"])
        idx._doc_count = int(payload["doc_count"])
        idx._df = {str(k): int(v) for k, v in payload["df"].items()}
        idx._doc_norms = {int(k): float(v) for k, v in payload["doc_norms"].items()}
        idx._docs = {int(k): DocMeta.from_dict(v) for k, v in payload["docs"].items()}
        idx._doc_top_terms = {
            int(k): [(str(t), int(tf)) for t, tf in v]
            for k, v in payload.get("doc_top_terms", {}).items()
        }
        if idx.compression == "vbyte":
            idx._postings_bytes = payload.get("postings_bytes", {})
        else:
            idx._postings = defaultdict(list, payload.get("postings", {}))
        return idx


def iter_corpus_jsonl(corpus_path: Path) -> Iterable[DocMeta]:
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield DocMeta.from_dict(json.loads(line))


def build_index(
    *,
    corpus_path: Path,
    index_dir: Path,
    compression: Compression = "none",
    top_terms_per_doc: int = 200,
) -> InvertedIndex:
    idx = InvertedIndex(compression=compression, top_terms_per_doc=top_terms_per_doc)
    for doc in iter_corpus_jsonl(corpus_path):
        idx.add_document(doc)
    idx.finalize()
    idx.save(index_dir)
    return idx
