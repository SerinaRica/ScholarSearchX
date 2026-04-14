from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DenseHit:
    doc_id: int
    score: float


def require_qdrant():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, PointStruct, VectorParams
    except Exception as e:
        raise RuntimeError(
            "缺少 qdrant 依赖。请安装: python -m pip install -e '.[kg_rag]'"
        ) from e
    return QdrantClient, Distance, PointStruct, VectorParams


def ensure_collection(*, qdrant_url: str, collection: str, vector_size: int) -> None:
    QdrantClient, Distance, _PointStruct, VectorParams = require_qdrant()
    client = QdrantClient(url=qdrant_url, check_compatibility=False)
    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=int(vector_size), distance=Distance.COSINE),
    )


def upsert_documents(
    *,
    qdrant_url: str,
    collection: str,
    vectors: list[list[float]],
    doc_ids: list[int],
    payloads: list[dict[str, Any]],
) -> None:
    QdrantClient, _Distance, PointStruct, _VectorParams = require_qdrant()
    if len(vectors) != len(doc_ids) or len(vectors) != len(payloads):
        raise ValueError("vectors/doc_ids/payloads 长度必须一致")
    client = QdrantClient(url=qdrant_url, check_compatibility=False)
    points = [
        PointStruct(id=int(doc_id), vector=vec, payload=payload)
        for doc_id, vec, payload in zip(doc_ids, vectors, payloads, strict=True)
    ]
    client.upsert(collection_name=collection, points=points)


def search(
    *,
    qdrant_url: str,
    collection: str,
    query_vector: list[float],
    top_k: int,
) -> list[DenseHit]:
    QdrantClient, _Distance, _PointStruct, _VectorParams = require_qdrant()
    client = QdrantClient(url=qdrant_url, check_compatibility=False)
    if hasattr(client, "search"):
        res = client.search(
            collection_name=collection, query_vector=query_vector, limit=int(top_k)
        )
        out: list[DenseHit] = []
        for p in res:
            out.append(DenseHit(doc_id=int(p.id), score=float(p.score)))
        return out

    resp = client.query_points(collection_name=collection, query=query_vector, limit=int(top_k))
    out: list[DenseHit] = []
    for p in resp.points:
        out.append(DenseHit(doc_id=int(p.id), score=float(p.score)))
    return out
