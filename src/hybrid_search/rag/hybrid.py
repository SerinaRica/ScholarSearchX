from __future__ import annotations

from dataclasses import dataclass

from hybrid_search.index.inverted_index import InvertedIndex
from hybrid_search.kg.neo4j_store import Neo4jConfig, retrieve_by_terms
from hybrid_search.preprocess.text import TextPreprocessor
from hybrid_search.rag.ollama_client import embed_texts, generate
from hybrid_search.search.query_expand import expand_query
from hybrid_search.search.tfidf_ranker import ScoredDoc, tfidf_search
from hybrid_search.vector.qdrant_store import DenseHit
from hybrid_search.vector.qdrant_store import search as qdrant_search


@dataclass(frozen=True)
class RagConfig:
    qdrant_url: str
    qdrant_collection: str
    ollama_url: str
    embed_model: str
    chat_model: str
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None


@dataclass(frozen=True)
class FusedDoc:
    doc_id: int
    score: float
    sources: list[str]


def rrf_fuse(
    *,
    sparse: list[ScoredDoc],
    dense: list[DenseHit],
    kg: list[int],
    k: int = 60,
    top_n: int = 50,
) -> list[FusedDoc]:
    scores: dict[int, float] = {}
    sources: dict[int, set[str]] = {}

    def add(doc_id: int, rank: int, src: str):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        sources.setdefault(doc_id, set()).add(src)

    for rank, d in enumerate(sparse, start=1):
        add(d.doc_id, rank, "sparse")
    for rank, d in enumerate(dense, start=1):
        add(d.doc_id, rank, "dense")
    for rank, doc_id in enumerate(kg, start=1):
        add(int(doc_id), rank, "kg")

    fused: list[FusedDoc] = []
    for doc_id, s in scores.items():
        fused.append(
            FusedDoc(
                doc_id=int(doc_id),
                score=float(s),
                sources=sorted(list(sources.get(doc_id, set()))),
            )
        )
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[: int(top_n)]


def retrieve_hybrid(
    *,
    index: InvertedIndex,
    query: str,
    cfg: RagConfig,
    top_k: int = 30,
    use_sparse: bool = True,
    use_dense: bool = True,
    use_kg_expand: bool = True,
    use_kg_neo4j: bool = False,
) -> tuple[list[FusedDoc], dict]:
    sparse: list[ScoredDoc] = []
    dense: list[DenseHit] = []
    kg_docs: list[int] = []
    debug: dict = {}

    if use_sparse:
        sparse, q_terms = tfidf_search(index, query, top_k=top_k)
        debug["sparse_query_terms"] = q_terms

    expanded_query = query
    added_terms: list[str] = []
    if use_kg_expand:
        expanded_query, info = expand_query(
            index=index,
            query=query,
            top_doc_ids=[d.doc_id for d in sparse[:10]],
            methods=["kg"],
        )
        added_terms = info.added_terms
        debug["kg_added_terms"] = added_terms

    if use_dense:
        vec = embed_texts(base_url=cfg.ollama_url, model=cfg.embed_model, texts=[expanded_query])[0]
        dense = qdrant_search(
            qdrant_url=cfg.qdrant_url,
            collection=cfg.qdrant_collection,
            query_vector=vec,
            top_k=top_k,
        )

    kg_docs = []
    if use_kg_neo4j:
        pre = TextPreprocessor()
        terms = pre.preprocess(query)
        if cfg.neo4j_uri and cfg.neo4j_user and cfg.neo4j_password:
            neo4j_cfg = Neo4jConfig(
                uri=cfg.neo4j_uri, user=cfg.neo4j_user, password=cfg.neo4j_password
            )
            kg_docs = retrieve_by_terms(neo4j_cfg, terms, top_k=top_k)
        debug["neo4j_terms"] = terms
        debug["neo4j_hits"] = kg_docs[:20]

    fused = rrf_fuse(sparse=sparse, dense=dense, kg=kg_docs, top_n=top_k)
    debug["dense_hits"] = [{"doc_id": d.doc_id, "score": d.score} for d in dense]
    return fused, debug


def build_rag_prompt(*, question: str, docs: list[dict]) -> str:
    lines: list[str] = []
    lines.append("你是学术检索助手。请基于给定文献摘要回答问题，并在回答中引用来源编号。")
    lines.append("")
    lines.append(f"问题：{question}")
    lines.append("")
    lines.append("资料：")
    for i, d in enumerate(docs, start=1):
        title = d.get("title", "")
        abstract = d.get("abstract", "")
        url = d.get("url", "")
        lines.append(f"[{i}] {title}")
        if url:
            lines.append(f"URL: {url}")
        if abstract:
            lines.append(f"Abstract: {abstract}")
        lines.append("")
    lines.append("输出要求：")
    lines.append("1) 用中文回答，结构清晰")
    lines.append("2) 关键结论后标注引用，例如 [1][3]")
    lines.append("3) 最后给出“参考文献”列表，列出引用到的条目编号与标题")
    return "\n".join(lines)


def rag_answer(
    *,
    index: InvertedIndex,
    question: str,
    cfg: RagConfig,
    top_k: int = 10,
    use_sparse: bool = True,
    use_dense: bool = True,
    use_kg_expand: bool = True,
    use_kg_neo4j: bool = False,
) -> tuple[str, list[dict], dict]:
    fused, debug = retrieve_hybrid(
        index=index,
        query=question,
        cfg=cfg,
        top_k=max(top_k, 10),
        use_sparse=use_sparse,
        use_dense=use_dense,
        use_kg_expand=use_kg_expand,
        use_kg_neo4j=use_kg_neo4j,
    )
    picked = fused[: int(top_k)]
    docs: list[dict] = []
    for fd in picked:
        m = index.get_doc(fd.doc_id)
        docs.append(
            {
                "doc_id": m.doc_id,
                "title": m.title,
                "abstract": m.abstract,
                "url": m.url,
                "sources": fd.sources,
                "fused_score": fd.score,
            }
        )
    prompt = build_rag_prompt(question=question, docs=docs)
    try:
        answer = generate(base_url=cfg.ollama_url, model=cfg.chat_model, prompt=prompt)
    except Exception as e:
        answer = (
            "当前生成模型不可用或尚未下载完成（Ollama pull 失败/网络中断）。\n\n"
            "下面先给出可用的检索结果作为参考文献列表，你可以基于这些文献继续阅读。\n"
        )
        debug["generate_error"] = str(e)
    return answer, docs, debug
