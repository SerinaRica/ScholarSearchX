from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hybrid_search.datasets.arxiv_downloader import download_arxiv_corpus
from hybrid_search.eval.evaluate import evaluate
from hybrid_search.index.inverted_index import InvertedIndex, build_index
from hybrid_search.kg.build_neo4j import load_corpus_to_neo4j
from hybrid_search.kg.graph import build_cooccurrence_graph
from hybrid_search.kg.neo4j_store import Neo4jConfig
from hybrid_search.preprocess.text import nltk_download
from hybrid_search.rag.hybrid import RagConfig, rag_answer
from hybrid_search.rag.ollama_client import embed_texts
from hybrid_search.search.boolean_query import boolean_retrieve, format_boolean_query
from hybrid_search.search.query_expand import expand_query
from hybrid_search.search.spell import suggest_term
from hybrid_search.search.tfidf_ranker import tfidf_search
from hybrid_search.vector.qdrant_store import ensure_collection, upsert_documents


def main() -> None:
    p = argparse.ArgumentParser(prog="hybrid-search")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download")
    p_dl.add_argument("--query", required=True)
    p_dl.add_argument("--max-results", type=int, default=2000)
    p_dl.add_argument("--out", type=Path, default=Path("data/corpus.jsonl"))

    sub.add_parser("nltk-download")

    p_build = sub.add_parser("build-index")
    p_build.add_argument("--corpus", type=Path, required=True)
    p_build.add_argument("--index-dir", type=Path, default=Path("data/index"))
    p_build.add_argument("--compress", choices=["none", "vbyte"], default="none")
    p_build.add_argument("--top-terms-per-doc", type=int, default=200)

    p_kg = sub.add_parser("build-kg")
    p_kg.add_argument("--index-dir", type=Path, default=Path("data/index"))
    p_kg.add_argument("--out", type=Path, default=Path("data/kg/graph.pkl"))
    p_kg.add_argument("--min-df", type=int, default=3)
    p_kg.add_argument("--per-doc-terms", type=int, default=30)
    p_kg.add_argument("--max-neighbors", type=int, default=30)

    p_neo = sub.add_parser("neo4j-load")
    p_neo.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    p_neo.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    p_neo.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p_neo.add_argument(
        "--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "neo4j_password")
    )
    p_neo.add_argument("--terms-per-paper", type=int, default=30)

    p_vec = sub.add_parser("vector-build")
    p_vec.add_argument("--index-dir", type=Path, default=Path("data/index"))
    p_vec.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    p_vec.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", "arxiv_papers"))
    p_vec.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    p_vec.add_argument(
        "--embed-model", default=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    )
    p_vec.add_argument("--batch", type=int, default=32)
    p_vec.add_argument("--limit", type=int, default=0)

    p_rag = sub.add_parser("rag-ask")
    p_rag.add_argument("--index-dir", type=Path, default=Path("data/index"))
    p_rag.add_argument("--question", required=True)
    p_rag.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    p_rag.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", "arxiv_papers"))
    p_rag.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    p_rag.add_argument(
        "--embed-model", default=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    )
    p_rag.add_argument(
        "--chat-model", default=os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1:8b-instruct")
    )
    p_rag.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI"))
    p_rag.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER"))
    p_rag.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD"))
    p_rag.add_argument("--topk", type=int, default=8)
    p_rag.add_argument("--no-sparse", action="store_true")
    p_rag.add_argument("--no-dense", action="store_true")
    p_rag.add_argument("--kg-expand", action="store_true")
    p_rag.add_argument("--kg-neo4j", action="store_true")

    p_search = sub.add_parser("search")
    p_search.add_argument("--index-dir", type=Path, default=Path("data/index"))
    p_search.add_argument("--mode", choices=["boolean", "tfidf"], default="tfidf")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--topk", type=int, default=20)
    p_search.add_argument("--expand", default="")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--index-dir", type=Path, default=Path("data/index"))
    p_eval.add_argument("--queries", type=Path, required=True)
    p_eval.add_argument("--qrels", type=Path, required=True)
    p_eval.add_argument("--topk", type=int, default=100)

    p_label = sub.add_parser("label")
    p_label.add_argument("--index-dir", type=Path, default=Path("data/index"))
    p_label.add_argument("--query", default="")
    p_label.add_argument("--queries", type=Path, default=None)
    p_label.add_argument("--topk", type=int, default=30)
    p_label.add_argument("--expand", default="")

    args = p.parse_args()

    if args.cmd == "download":
        download_arxiv_corpus(args.query, args.max_results, args.out)
        return

    if args.cmd == "nltk-download":
        nltk_download()
        return

    if args.cmd == "build-index":
        build_index(
            corpus_path=args.corpus,
            index_dir=args.index_dir,
            compression=args.compress,
            top_terms_per_doc=args.top_terms_per_doc,
        )
        return

    if args.cmd == "build-kg":
        index = InvertedIndex.load(args.index_dir)
        kg = build_cooccurrence_graph(
            index=index,
            min_df=args.min_df,
            per_doc_terms=args.per_doc_terms,
            max_neighbors=args.max_neighbors,
        )
        kg.save(args.out)
        return

    if args.cmd == "neo4j-load":
        cfg = Neo4jConfig(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
        stats = load_corpus_to_neo4j(
            corpus_path=args.corpus,
            cfg=cfg,
            terms_per_paper=args.terms_per_paper,
        )
        print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))
        return

    if args.cmd == "vector-build":
        index = InvertedIndex.load(args.index_dir)
        doc_ids = sorted(index.all_doc_ids())
        if args.limit and int(args.limit) > 0:
            doc_ids = doc_ids[: int(args.limit)]
        texts: list[str] = []
        payloads: list[dict] = []
        ids: list[int] = []

        batch_size = int(args.batch)
        total = len(doc_ids)
        done = 0
        for doc_id in doc_ids:
            m = index.get_doc(doc_id)
            text = f"{m.title}\n\n{m.abstract}"
            texts.append(text)
            payloads.append({"title": m.title, "url": m.url})
            ids.append(int(doc_id))
            if len(texts) >= batch_size:
                vectors = embed_texts(base_url=args.ollama_url, model=args.embed_model, texts=texts)
                ensure_collection(
                    qdrant_url=args.qdrant_url,
                    collection=args.collection,
                    vector_size=len(vectors[0]),
                )
                upsert_documents(
                    qdrant_url=args.qdrant_url,
                    collection=args.collection,
                    vectors=vectors,
                    doc_ids=ids,
                    payloads=payloads,
                )
                done += len(ids)
                print(json.dumps({"progress": f"{done}/{total}"}, ensure_ascii=False))
                texts, payloads, ids = [], [], []

        if texts:
            vectors = embed_texts(base_url=args.ollama_url, model=args.embed_model, texts=texts)
            ensure_collection(
                qdrant_url=args.qdrant_url,
                collection=args.collection,
                vector_size=len(vectors[0]),
            )
            upsert_documents(
                qdrant_url=args.qdrant_url,
                collection=args.collection,
                vectors=vectors,
                doc_ids=ids,
                payloads=payloads,
            )
            done += len(ids)
            print(json.dumps({"progress": f"{done}/{total}"}, ensure_ascii=False))
        print(
            json.dumps(
                {"docs_indexed": len(doc_ids), "collection": args.collection},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.cmd == "rag-ask":
        index = InvertedIndex.load(args.index_dir)
        cfg = RagConfig(
            qdrant_url=args.qdrant_url,
            qdrant_collection=args.collection,
            ollama_url=args.ollama_url,
            embed_model=args.embed_model,
            chat_model=args.chat_model,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
        )
        answer, docs, debug = rag_answer(
            index=index,
            question=args.question,
            cfg=cfg,
            top_k=args.topk,
            use_sparse=not args.no_sparse,
            use_dense=not args.no_dense,
            use_kg_expand=args.kg_expand,
            use_kg_neo4j=args.kg_neo4j,
        )
        print(
            json.dumps(
                {"answer": answer, "sources": docs, "debug": debug},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.cmd == "search":
        index = InvertedIndex.load(args.index_dir)
        if args.mode == "boolean":
            res = boolean_retrieve(index, args.query)
            out = {
                "mode": "boolean",
                "postfix": format_boolean_query(res.postfix),
                "hits": len(res.doc_ids),
                "doc_ids": res.doc_ids[: args.topk],
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return

        if args.mode == "tfidf":
            results, q_terms = tfidf_search(index, args.query, top_k=args.topk)
            sugg = _suggest_oov(index, q_terms)

            expand_methods = [m for m in args.expand.split(",") if m]
            expanded_query = ""
            added_terms: list[str] = []
            expanded_results = []
            if expand_methods:
                expanded_query, info = expand_query(
                    index=index,
                    query=args.query,
                    top_doc_ids=[d.doc_id for d in results],
                    methods=expand_methods,
                )
                added_terms = info.added_terms
                expanded_results, _ = tfidf_search(index, expanded_query, top_k=args.topk)

            payload = {
                "mode": "tfidf",
                "query_terms": q_terms,
                "suggestions": [s.__dict__ for s in sugg],
                "results": [_doc_payload(index, d.doc_id, d.score) for d in results],
            }
            if expand_methods:
                payload["expanded"] = {
                    "methods": expand_methods,
                    "expanded_query": expanded_query,
                    "added_terms": added_terms,
                    "results": [_doc_payload(index, d.doc_id, d.score) for d in expanded_results],
                }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return

    if args.cmd == "evaluate":
        metrics = evaluate(
            index_dir=args.index_dir,
            queries_path=args.queries,
            qrels_path=args.qrels,
            top_k=args.topk,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if args.cmd == "label":
        index = InvertedIndex.load(args.index_dir)
        items: list[tuple[str, str]] = []
        if args.queries is not None:
            with args.queries.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            for it in raw:
                items.append((str(it["qid"]), str(it["query"])))
        else:
            items.append(("q", str(args.query)))

        expand_methods = [m for m in args.expand.split(",") if m]
        for qid, qtext in items:
            results, _ = tfidf_search(index, qtext, top_k=args.topk)
            base_doc_ids = [d.doc_id for d in results]
            expanded_query = ""
            expanded_results = []
            added_terms: list[str] = []
            if expand_methods:
                expanded_query, info = expand_query(
                    index=index, query=qtext, top_doc_ids=base_doc_ids[:10], methods=expand_methods
                )
                added_terms = info.added_terms
                expanded_results, _ = tfidf_search(index, expanded_query, top_k=args.topk)

            print(f"# qid={qid}")
            print(f"# query={qtext}")
            if expand_methods:
                print(f"# expand_methods={','.join(expand_methods)}")
                print(f"# expanded_query={expanded_query}")
                print(f"# added_terms={' '.join(added_terms)}")
            print("rank\tdoc_id\tscore\ttitle\turl")
            for rank, d in enumerate(results, start=1):
                meta = index.get_doc(d.doc_id)
                print(f"{rank}\t{meta.doc_id}\t{d.score:.6f}\t{meta.title}\t{meta.url}")
            if expand_methods:
                print("# expanded")
                print("rank\tdoc_id\tscore\ttitle\turl")
                for rank, d in enumerate(expanded_results, start=1):
                    meta = index.get_doc(d.doc_id)
                    print(f"{rank}\t{meta.doc_id}\t{d.score:.6f}\t{meta.title}\t{meta.url}")
            print()
        return


def _doc_payload(index: InvertedIndex, doc_id: int, score: float) -> dict:
    d = index.get_doc(doc_id)
    return {
        "doc_id": d.doc_id,
        "score": score,
        "title": d.title,
        "url": d.url,
        "authors": d.authors,
        "categories": d.categories,
    }


def _suggest_oov(index: InvertedIndex, query_terms: list[str]):
    out = []
    vocab = index.vocabulary
    for t in query_terms:
        s = suggest_term(t, vocab)
        if s is not None:
            out.append(s)
    return out


if __name__ == "__main__":
    main()
