from __future__ import annotations

from collections.abc import Iterable


def vb_encode_number(n: int) -> bytes:
    if n < 0:
        raise ValueError("vbyte only supports non-negative integers")
    bytes_list: list[int] = []
    while True:
        bytes_list.insert(0, n % 128)
        if n < 128:
            break
        n //= 128
    bytes_list[-1] += 128
    return bytes(bytes_list)


def vb_encode_stream(numbers: Iterable[int]) -> bytes:
    out = bytearray()
    for n in numbers:
        out.extend(vb_encode_number(int(n)))
    return bytes(out)


def vb_decode_stream(data: bytes) -> list[int]:
    numbers: list[int] = []
    n = 0
    for b in data:
        if b < 128:
            n = 128 * n + b
        else:
            n = 128 * n + (b - 128)
            numbers.append(n)
            n = 0
    if n != 0:
        raise ValueError("incomplete vbyte stream")
    return numbers


def encode_postings_vb(doc_ids: list[int], tfs: list[int]) -> bytes:
    if len(doc_ids) != len(tfs):
        raise ValueError("doc_ids and tfs must have same length")
    if not doc_ids:
        return b""
    gaps: list[int] = []
    prev = 0
    for i, d in enumerate(doc_ids):
        if i == 0:
            gaps.append(d)
        else:
            gaps.append(d - prev)
        prev = d
    interleaved: list[int] = []
    for g, tf in zip(gaps, tfs, strict=True):
        interleaved.append(g)
        interleaved.append(tf)
    return vb_encode_stream(interleaved)


def decode_postings_vb(data: bytes) -> tuple[list[int], list[int]]:
    nums = vb_decode_stream(data)
    if len(nums) % 2 != 0:
        raise ValueError("postings stream must contain (gap, tf) pairs")
    gaps = nums[0::2]
    tfs = nums[1::2]
    doc_ids: list[int] = []
    prev = 0
    for i, g in enumerate(gaps):
        if i == 0:
            doc_ids.append(g)
            prev = g
        else:
            prev = prev + g
            doc_ids.append(prev)
    return doc_ids, tfs
