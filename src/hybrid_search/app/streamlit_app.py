from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import streamlit as st

from hybrid_search.index.inverted_index import InvertedIndex
from hybrid_search.kg.graph import build_cooccurrence_graph
from hybrid_search.search.boolean_query import boolean_retrieve, format_boolean_query
from hybrid_search.search.query_expand import expand_query
from hybrid_search.search.tfidf_ranker import tfidf_search


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir", type=Path, required=True)
    p.add_argument("--kg-graph", type=Path, default=Path("data/kg/graph.pkl"))
    p.add_argument("--kg-dict", type=Path, default=Path("data/kg_dict.json"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    st.set_page_config(page_title="Hybrid Academic Search", layout="wide")
    st.title("基于混合检索与知识图谱增强的学术文献搜索系统")

    with st.sidebar:
        st.header("设置")
        index_dir = st.text_input("Index Dir", value=str(args.index_dir))
        kg_graph_path = st.text_input("KG Graph", value=str(args.kg_graph))
        kg_dict_path = st.text_input("KG Dict", value=str(args.kg_dict))
        st.divider()

        mode = st.radio("检索模式", options=["TF-IDF", "布尔检索"], index=0)
        top_k = st.slider("Top-K", min_value=5, max_value=50, value=20, step=5)

        expand_rocchio = st.checkbox("Rocchio 扩展", value=True)
        expand_kg = st.checkbox("KG 扩展", value=True)
        st.caption("KG 扩展会优先使用 data/kg/graph.pkl（若存在），其次使用 data/kg_dict.json")

        st.divider()
        if st.button("构建/更新 KG（从语料自动生成）", use_container_width=True):
            idx = _load_index(Path(index_dir))
            kg = build_cooccurrence_graph(index=idx)
            out = Path(kg_graph_path)
            kg.save(out)
            st.success(f"已生成: {out}")

        st.divider()
        st.header("KG + RAG（Docker 服务）")
        qdrant_url = st.text_input("Qdrant URL", value=os.environ.get("QDRANT_URL", "http://localhost:6333"))
        qdrant_collection = st.text_input(
            "Qdrant Collection", value=os.environ.get("QDRANT_COLLECTION", "arxiv_papers")
        )
        ollama_url = st.text_input("Ollama URL", value=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
        embed_model = st.text_input(
            "Embed Model", value=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        )
        chat_model = st.text_input(
            "Chat Model", value=os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1:8b-instruct")
        )
        neo4j_uri = st.text_input("Neo4j URI", value=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
        neo4j_user = st.text_input("Neo4j User", value=os.environ.get("NEO4J_USER", "neo4j"))
        neo4j_password = st.text_input(
            "Neo4j Password",
            value=os.environ.get("NEO4J_PASSWORD", "neo4j_password"),
            type="password",
        )

    index = _load_index(Path(index_dir))

    query = st.text_input("Query", value="kg rag hallucination mitigation")

    tab_search, tab_compare, tab_rag, tab_about = st.tabs(["检索", "对比", "RAG 问答", "说明"])

    methods: list[str] = []
    if mode == "TF-IDF":
        if expand_rocchio:
            methods.append("rocchio")
        if expand_kg:
            methods.append("kg")

    with tab_search:
        if mode == "布尔检索":
            try:
                res = boolean_retrieve(index, query)
                st.caption(f"Postfix: {format_boolean_query(res.postfix)}")
                _render_boolean_results(index, res.doc_ids[:top_k])
            except Exception as e:
                st.error(str(e))
        else:
            base_results, q_terms = tfidf_search(index, query, top_k=top_k)
            st.caption(f"Query terms: {' '.join(q_terms)}")
            _render_scored_results(index, base_results, title="Base: TF-IDF")

            if methods:
                expanded_query, info = expand_query(
                    index=index,
                    query=query,
                    top_doc_ids=[d.doc_id for d in base_results],
                    methods=methods,
                    kg_dict_path=Path(kg_dict_path),
                    kg_graph_path=Path(kg_graph_path),
                )
                st.divider()
                st.subheader("Enhanced: Expansion + TF-IDF")
                st.caption(f"Expanded query: {expanded_query}")
                st.caption(f"Added terms: {' '.join(info.added_terms)}")
                expanded_results, _ = tfidf_search(index, expanded_query, top_k=top_k)
                _render_scored_results(index, expanded_results, title=None)

    with tab_compare:
        if mode != "TF-IDF":
            st.info("对比页目前仅支持 TF-IDF 模式。")
        else:
            base_results, _ = tfidf_search(index, query, top_k=top_k)
            base_ids = [d.doc_id for d in base_results]
            left, right = st.columns(2)
            with left:
                st.subheader("Base")
                _render_scored_results(index, base_results, title=None)
            with right:
                st.subheader("Enhanced")
                if not methods:
                    st.info("请在侧边栏启用 Rocchio/KG 扩展。")
                else:
                    expanded_query, info = expand_query(
                        index=index,
                        query=query,
                        top_doc_ids=base_ids,
                        methods=methods,
                        kg_dict_path=Path(kg_dict_path),
                        kg_graph_path=Path(kg_graph_path),
                    )
                    st.caption(f"Added terms: {' '.join(info.added_terms)}")
                    expanded_results, _ = tfidf_search(index, expanded_query, top_k=top_k)
                    _render_scored_results(index, expanded_results, title=None)
                    _render_diff(base_ids, [d.doc_id for d in expanded_results])

    with tab_about:
        st.subheader("中期展示建议")
        st.write(
            "建议中期演示聚焦：索引已构建、检索可用、增强可解释（展示 Added terms）、界面可交互。"
        )
        st.write("如果要做严谨评测（MAP/NDCG），需要 qrels（人工或公开数据集）。")
        st.write("如果要做真正的 KG + RAG：启动 Docker（Neo4j/Qdrant/Ollama）。")
        st.write("并构建向量索引（Qdrant）与图谱（Neo4j KG）。")
        st.subheader("当前索引信息")
        st.code(
            json.dumps(
                {
                    "doc_count": index.doc_count,
                    "vocab_size": len(index.vocabulary),
                    "compression": index.compression,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    st.caption("Tip: 若运行时提示找不到 KG 图谱，先执行 build-kg 或在侧边栏点击“构建/更新 KG”。")
    with tab_rag:
        st.subheader("KG + RAG 问答（需要 Qdrant + Ollama，可选 Neo4j）")
        question = st.text_input("Question", value=query, key="rag_question")
        cols = st.columns(4)
        use_sparse = cols[0].checkbox("Sparse(TF-IDF)", value=True)
        use_dense = cols[1].checkbox("Dense(Vector)", value=True)
        use_kg_expand = cols[2].checkbox("KG Expand", value=True)
        use_kg_neo4j = cols[3].checkbox("Neo4j KG", value=False)
        topk = st.slider("Context Top-K", min_value=3, max_value=15, value=8, step=1)

        btn_cols = st.columns(3)
        if btn_cols[0].button("构建向量库（Qdrant）", use_container_width=True):
            try:
                from hybrid_search.rag.ollama_client import embed_texts
                from hybrid_search.vector.qdrant_store import ensure_collection, upsert_documents

                ids = sorted(index.all_doc_ids())
                texts: list[str] = []
                payloads: list[dict] = []
                doc_ids: list[int] = []
                batch = 32
                for doc_id in ids:
                    m = index.get_doc(doc_id)
                    texts.append(f"{m.title}\n\n{m.abstract}")
                    payloads.append({"title": m.title, "url": m.url})
                    doc_ids.append(int(doc_id))
                    if len(texts) >= batch:
                        vecs = embed_texts(base_url=ollama_url, model=embed_model, texts=texts)
                        ensure_collection(
                            qdrant_url=qdrant_url,
                            collection=qdrant_collection,
                            vector_size=len(vecs[0]),
                        )
                        upsert_documents(
                            qdrant_url=qdrant_url,
                            collection=qdrant_collection,
                            vectors=vecs,
                            doc_ids=doc_ids,
                            payloads=payloads,
                        )
                        texts, payloads, doc_ids = [], [], []
                if texts:
                    vecs = embed_texts(base_url=ollama_url, model=embed_model, texts=texts)
                    ensure_collection(
                        qdrant_url=qdrant_url,
                        collection=qdrant_collection,
                        vector_size=len(vecs[0]),
                    )
                    upsert_documents(
                        qdrant_url=qdrant_url,
                        collection=qdrant_collection,
                        vectors=vecs,
                        doc_ids=doc_ids,
                        payloads=payloads,
                    )
                st.success("向量库构建完成")
            except Exception as e:
                st.error(str(e))

        if btn_cols[1].button("导入 Neo4j KG", use_container_width=True):
            try:
                from hybrid_search.kg.build_neo4j import load_corpus_to_neo4j
                from hybrid_search.kg.neo4j_store import Neo4jConfig

                corpus = Path("data") / "corpus.jsonl"
                stats = load_corpus_to_neo4j(
                    corpus_path=corpus,
                    cfg=Neo4jConfig(uri=neo4j_uri, user=neo4j_user, password=neo4j_password),
                )
                st.success(f"Neo4j 导入完成: papers={stats.papers}")
            except Exception as e:
                st.error(str(e))

        if btn_cols[2].button("运行 RAG", use_container_width=True):
            try:
                from hybrid_search.rag.hybrid import RagConfig, rag_answer

                cfg = RagConfig(
                    qdrant_url=qdrant_url,
                    qdrant_collection=qdrant_collection,
                    ollama_url=ollama_url,
                    embed_model=embed_model,
                    chat_model=chat_model,
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                )
                answer, docs, debug = rag_answer(
                    index=index,
                    question=question,
                    cfg=cfg,
                    top_k=topk,
                    use_sparse=use_sparse,
                    use_dense=use_dense,
                    use_kg_expand=use_kg_expand,
                    use_kg_neo4j=use_kg_neo4j,
                )
                st.subheader("Answer")
                st.write(answer)
                st.subheader("Sources")
                st.dataframe(
                    [
                        {
                            "doc_id": d["doc_id"],
                            "title": d["title"],
                            "url": d["url"],
                            "sources": ",".join(d.get("sources", [])),
                            "fused_score": d.get("fused_score", 0.0),
                        }
                        for d in docs
                    ],
                    use_container_width=True,
                )
                with st.expander("Debug", expanded=False):
                    st.code(json.dumps(debug, ensure_ascii=False, indent=2))
            except Exception as e:
                st.error(str(e))


@st.cache_resource
def _load_index(index_dir: Path) -> InvertedIndex:
    return InvertedIndex.load(index_dir)


def _render_scored_results(index: InvertedIndex, results, *, title: str | None):
    if title:
        st.subheader(title)
    if not results:
        st.info("没有结果。")
        return
    for i, r in enumerate(results, start=1):
        doc = index.get_doc(r.doc_id)
        with st.expander(f"{i}. {doc.title}", expanded=(i <= 3)):
            cols = st.columns([1, 1, 2])
            cols[0].write(f"doc_id={doc.doc_id}")
            cols[1].write(f"score={r.score:.4f}")
            cols[2].markdown(f"[arXiv link]({doc.url})")
            if doc.authors:
                st.write(", ".join(doc.authors[:12]))
            if doc.categories:
                st.write("Categories: " + ", ".join(doc.categories))
            if doc.abstract:
                st.write(doc.abstract)


def _render_boolean_results(index: InvertedIndex, doc_ids: list[int]):
    if not doc_ids:
        st.info("没有命中文档。")
        return
    for i, doc_id in enumerate(doc_ids, start=1):
        doc = index.get_doc(doc_id)
        with st.expander(f"{i}. {doc.title}", expanded=(i <= 5)):
            st.write(f"doc_id={doc.doc_id}")
            st.markdown(f"[arXiv link]({doc.url})")
            if doc.authors:
                st.write(", ".join(doc.authors[:12]))
            if doc.categories:
                st.write("Categories: " + ", ".join(doc.categories))
            if doc.abstract:
                st.write(doc.abstract)


def _render_diff(base_doc_ids: list[int], enhanced_doc_ids: list[int]) -> None:
    base_top = base_doc_ids[:10]
    enh_top = enhanced_doc_ids[:10]
    new_in_top10 = [d for d in enh_top if d not in base_top]
    dropped_from_top10 = [d for d in base_top if d not in enh_top]
    st.divider()
    st.subheader("Top-10 变化")
    cols = st.columns(2)
    cols[0].write("Enhanced 新进入 Top-10 的 doc_id：")
    cols[0].code("\n".join(map(str, new_in_top10)) if new_in_top10 else "(无)")
    cols[1].write("Enhanced 中被挤出 Top-10 的 doc_id：")
    cols[1].code("\n".join(map(str, dropped_from_top10)) if dropped_from_top10 else "(无)")


if __name__ == "__main__":
    main()
