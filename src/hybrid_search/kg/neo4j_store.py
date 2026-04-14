from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def require_neo4j():
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        raise RuntimeError(
            "缺少 neo4j 依赖。请安装: python -m pip install -e '.[kg_rag]'"
        ) from e
    return GraphDatabase


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str


def init_schema(cfg: Neo4jConfig) -> None:
    GraphDatabase = require_neo4j()
    driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
    try:
        with driver.session() as session:
            session.run(
                "CREATE CONSTRAINT paper_doc_id IF NOT EXISTS FOR (p:Paper) "
                "REQUIRE p.doc_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) "
                "REQUIRE a.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) "
                "REQUIRE c.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT term_term IF NOT EXISTS FOR (t:Term) REQUIRE t.term IS UNIQUE"
            )
    finally:
        driver.close()


def upsert_papers(cfg: Neo4jConfig, papers: list[dict[str, Any]]) -> None:
    GraphDatabase = require_neo4j()
    driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
    try:
        with driver.session() as session:
            session.run(
                """
UNWIND $papers AS p
MERGE (paper:Paper {doc_id: p.doc_id})
SET paper.title = p.title,
    paper.abstract = p.abstract,
    paper.url = p.url
WITH paper, p
FOREACH (a IN coalesce(p.authors, []) |
  MERGE (au:Author {name: a})
  MERGE (paper)-[:HAS_AUTHOR]->(au)
)
FOREACH (c IN coalesce(p.categories, []) |
  MERGE (cat:Category {name: c})
  MERGE (paper)-[:IN_CATEGORY]->(cat)
)
FOREACH (t IN coalesce(p.terms, []) |
  MERGE (term:Term {term: t})
  MERGE (paper)-[:MENTIONS]->(term)
)
""",
                papers=papers,
            )
    finally:
        driver.close()


def retrieve_by_terms(cfg: Neo4jConfig, terms: list[str], *, top_k: int = 50) -> list[int]:
    if not terms:
        return []
    GraphDatabase = require_neo4j()
    driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
    try:
        with driver.session() as session:
            res = session.run(
                """
MATCH (t:Term)
WHERE t.term IN $terms
MATCH (t)<-[:MENTIONS]-(p:Paper)
RETURN p.doc_id AS doc_id, count(*) AS score
ORDER BY score DESC
LIMIT $top_k
""",
                terms=terms,
                top_k=int(top_k),
            )
            return [int(r["doc_id"]) for r in res]
    finally:
        driver.close()
