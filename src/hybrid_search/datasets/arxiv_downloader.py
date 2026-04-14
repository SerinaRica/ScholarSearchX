from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import arxiv
from tqdm import tqdm


@dataclass(frozen=True)
class ArxivDoc:
    doc_id: int
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    url: str

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "published": self.published,
            "updated": self.updated,
            "url": self.url,
        }


def download_arxiv_corpus(query: str, max_results: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client(page_size=200, delay_seconds=3.0, num_retries=3)

    with out_path.open("w", encoding="utf-8") as f:
        for i, result in enumerate(tqdm(client.results(search), total=max_results)):
            if i >= max_results:
                break
            doc = _result_to_doc(doc_id=i, result=result)
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")


def _dt(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.isoformat()


def _result_to_doc(doc_id: int, result: arxiv.Result) -> ArxivDoc:
    return ArxivDoc(
        doc_id=doc_id,
        arxiv_id=str(result.get_short_id()),
        title=(result.title or "").replace("\n", " ").strip(),
        abstract=(result.summary or "").replace("\n", " ").strip(),
        authors=[a.name for a in (result.authors or [])],
        categories=list(result.categories or []),
        published=_dt(result.published),
        updated=_dt(result.updated),
        url=str(result.entry_id or ""),
    )

