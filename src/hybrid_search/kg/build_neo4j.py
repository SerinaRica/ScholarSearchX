from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hybrid_search.kg.neo4j_store import Neo4jConfig, init_schema, upsert_papers
from hybrid_search.preprocess.text import TextPreprocessor


@dataclass(frozen=True)
class KgLoadStats:
    papers: int
    terms_per_paper: int


def load_corpus_to_neo4j(
    *,
    corpus_path: Path,
    cfg: Neo4jConfig,
    terms_per_paper: int = 30,
    batch_size: int = 200,
) -> KgLoadStats:
    init_schema(cfg)
    pre = TextPreprocessor()

    batch: list[dict] = []
    paper_count = 0
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            doc_id = int(d["doc_id"])
            abstract = str(d.get("abstract", ""))
            title = str(d.get("title", ""))
            url = str(d.get("url", ""))
            authors = list(d.get("authors", []))
            categories = list(d.get("categories", []))
            terms = pre.preprocess(abstract)[: int(terms_per_paper)]
            batch.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "abstract": abstract,
                    "url": url,
                    "authors": authors,
                    "categories": categories,
                    "terms": terms,
                }
            )
            paper_count += 1
            if len(batch) >= int(batch_size):
                upsert_papers(cfg, batch)
                batch = []

    if batch:
        upsert_papers(cfg, batch)

    return KgLoadStats(papers=paper_count, terms_per_paper=int(terms_per_paper))

