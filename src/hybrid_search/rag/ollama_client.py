from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def require_httpx():
    try:
        import httpx
    except Exception as e:
        raise RuntimeError("缺少 httpx 依赖。请安装: python -m pip install -e '.[kg_rag]'") from e
    return httpx


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    embed_model: str
    chat_model: str


def embed_texts(*, base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    httpx = require_httpx()
    with httpx.Client(base_url=base_url, timeout=120.0, trust_env=False) as client:
        r = client.post("/api/embed", json={"model": model, "input": texts})
        if r.status_code >= 400:
            out: list[list[float]] = []
            for t in texts:
                r2 = client.post("/api/embeddings", json={"model": model, "prompt": t})
                r2.raise_for_status()
                data2: dict[str, Any] = r2.json()
                vec2 = data2.get("embedding")
                if not isinstance(vec2, list):
                    raise RuntimeError("Ollama embeddings 返回格式不正确")
                out.append([float(x) for x in vec2])
            return out
        r.raise_for_status()
        data: dict[str, Any] = r.json()
        vecs = data.get("embeddings")
        if not isinstance(vecs, list):
            raise RuntimeError("Ollama embed 返回格式不正确")
        out = []
        for vec in vecs:
            if not isinstance(vec, list):
                raise RuntimeError("Ollama embed 返回格式不正确")
            out.append([float(x) for x in vec])
        return out


def generate(*, base_url: str, model: str, prompt: str) -> str:
    httpx = require_httpx()
    with httpx.Client(base_url=base_url, timeout=120.0, trust_env=False) as client:
        r = client.post("/api/generate", json={"model": model, "prompt": prompt, "stream": False})
        r.raise_for_status()
        data: dict[str, Any] = r.json()
        resp = data.get("response", "")
        return str(resp)
