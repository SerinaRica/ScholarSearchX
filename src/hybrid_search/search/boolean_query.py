from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from hybrid_search.index.inverted_index import InvertedIndex
from hybrid_search.preprocess.text import TextPreprocessor

_OPS = {"AND", "OR", "NOT", "(", ")"}


@dataclass(frozen=True)
class BooleanQueryResult:
    doc_ids: list[int]
    postfix: list[str]


def boolean_retrieve(index: InvertedIndex, query: str) -> BooleanQueryResult:
    tokens = _tokenize_query(query)
    postfix = _to_postfix(tokens)
    doc_ids = _eval_postfix(index, postfix)
    return BooleanQueryResult(doc_ids=doc_ids, postfix=postfix)


def _tokenize_query(query: str) -> list[str]:
    q = query.replace("(", " ( ").replace(")", " ) ")
    parts = [p for p in q.split() if p]
    out: list[str] = []
    for p in parts:
        up = p.upper()
        if up in _OPS:
            out.append(up)
        else:
            out.append(p)
    return _insert_implicit_and(out)


def _insert_implicit_and(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if i > 0:
            prev = tokens[i - 1]
            prev_is_term = prev not in _OPS or prev == ")"
            cur_is_term = tok not in _OPS or tok == "(" or tok == "NOT"
            if prev_is_term and cur_is_term and prev not in {"AND", "OR"}:
                out.append("AND")
        out.append(tok)
    return out


def _to_postfix(tokens: list[str]) -> list[str]:
    prec = {"NOT": 3, "AND": 2, "OR": 1}
    right_assoc = {"NOT"}
    out: list[str] = []
    stack: list[str] = []
    for tok in tokens:
        if tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                out.append(stack.pop())
            if not stack or stack[-1] != "(":
                raise ValueError("mismatched parentheses")
            stack.pop()
        elif tok in prec:
            while stack and stack[-1] in prec:
                top = stack[-1]
                if (top in right_assoc and prec[top] > prec[tok]) or (
                    top not in right_assoc and prec[top] >= prec[tok]
                ):
                    out.append(stack.pop())
                else:
                    break
            stack.append(tok)
        else:
            out.append(tok)
    while stack:
        if stack[-1] in {"(", ")"}:
            raise ValueError("mismatched parentheses")
        out.append(stack.pop())
    return out


def _eval_postfix(index: InvertedIndex, postfix: list[str]) -> list[int]:
    pre = TextPreprocessor()
    universe = index.all_doc_ids()
    stack: list[set[int]] = []
    for tok in postfix:
        if tok == "NOT":
            if not stack:
                raise ValueError("NOT requires one operand")
            a = stack.pop()
            stack.append(universe.difference(a))
        elif tok in {"AND", "OR"}:
            if len(stack) < 2:
                raise ValueError(f"{tok} requires two operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(a.intersection(b) if tok == "AND" else a.union(b))
        else:
            terms = pre.preprocess(tok)
            if not terms:
                stack.append(set())
                continue
            term = terms[0]
            docs = {doc_id for doc_id, _ in index.get_postings(term)}
            stack.append(docs)
    if len(stack) != 1:
        raise ValueError("invalid boolean expression")
    return sorted(stack[0])


def format_boolean_query(postfix: Iterable[str]) -> str:
    return " ".join(postfix)
