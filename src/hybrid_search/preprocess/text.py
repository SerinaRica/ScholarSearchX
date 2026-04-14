from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


@dataclass(frozen=True)
class TextPreprocessor:
    lowercase: bool = True
    use_stemmer: bool = True

    def tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return _TOKEN_RE.findall(text)

    def normalize_tokens(self, tokens: Iterable[str]) -> list[str]:
        tokens_list = [t for t in tokens if t]
        stop = _load_stopwords()
        tokens_list = [t for t in tokens_list if t not in stop]
        if not self.use_stemmer:
            return tokens_list
        stemmer = _load_stemmer()
        return [stemmer.stem(t) for t in tokens_list]

    def preprocess(self, text: str) -> list[str]:
        return self.normalize_tokens(self.tokenize(text))


def nltk_download() -> None:
    import nltk

    nltk.download("stopwords", quiet=False)


def _load_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        return {
            "a",
            "an",
            "the",
            "and",
            "or",
            "not",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "as",
            "by",
            "at",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "it",
            "this",
            "that",
        }


def _load_stemmer():
    try:
        from nltk.stem import PorterStemmer

        return PorterStemmer()
    except Exception:
        return _NoOpStemmer()


class _NoOpStemmer:
    def stem(self, token: str) -> str:
        return token
