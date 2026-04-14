from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Suggestion:
    original: str
    suggestion: str
    distance: int


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def suggest_term(
    term: str,
    vocabulary: set[str],
    *,
    max_distance: int = 2,
    max_candidates: int = 2000,
) -> Suggestion | None:
    term = term.strip()
    if not term or term in vocabulary:
        return None
    candidates = _candidate_filter(term, vocabulary, max_candidates=max_candidates)
    best: tuple[int, str] | None = None
    for cand in candidates:
        d = levenshtein(term, cand)
        if d > max_distance:
            continue
        if best is None or d < best[0] or (d == best[0] and cand < best[1]):
            best = (d, cand)
            if d == 1:
                break
    if best is None:
        return None
    return Suggestion(original=term, suggestion=best[1], distance=best[0])


def _candidate_filter(term: str, vocabulary: set[str], *, max_candidates: int) -> list[str]:
    first = term[0]
    target_len = len(term)
    out: list[str] = []
    for v in vocabulary:
        if not v:
            continue
        if v[0] != first:
            continue
        if abs(len(v) - target_len) > 2:
            continue
        out.append(v)
        if len(out) >= max_candidates:
            break
    return out

