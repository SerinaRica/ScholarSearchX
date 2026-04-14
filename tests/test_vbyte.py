from hybrid_search.index.compress_vbyte import decode_postings_vb, encode_postings_vb


def test_vbyte_postings_roundtrip():
    doc_ids = [3, 10, 11, 120, 121]
    tfs = [1, 2, 10, 3, 1]
    data = encode_postings_vb(doc_ids, tfs)
    d2, tf2 = decode_postings_vb(data)
    assert d2 == doc_ids
    assert tf2 == tfs

