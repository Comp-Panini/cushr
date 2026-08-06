#!/usr/bin/env python3
"""SLP1 <-> IAST, losslessly, for talking to IAST-trained models.

Our corpus is SLP1 (`cikzepa me suto rAjan`); ByT5-Sanskrit is IAST
(`cikṣepa me suto rājan`). Inputs must be converted going in and outputs coming
back, and a silent bug here is indistinguishable from a model difference -- so
`assert_roundtrip` is called on the real data before any inference runs.

Why not reuse `ingest.normalize_lemma`: it is IAST->SLP1 only and it is
deliberately LOSSY -- it deletes avagraha (`'`), which is right for comparing
lemmas and wrong for round-tripping sentences. Inverting a lossy map by
reversing its dict silently drops characters. This module keeps its own table.

SLP1 is one character per phoneme, so SLP1->IAST is a plain per-character
substitution. IAST->SLP1 is not: `kh`, `ai` and friends are two characters that
map to one, so multi-character sequences must be consumed before single ones.
"""

# SLP1 char -> IAST string. Everything absent maps to itself (k, g, c, ...).
SLP1_TO_IAST = {
    'A': 'ā', 'I': 'ī', 'U': 'ū',
    'f': 'ṛ', 'F': 'ṝ', 'x': 'ḷ', 'X': 'ḹ',
    'E': 'ai', 'O': 'au',
    'K': 'kh', 'G': 'gh', 'C': 'ch', 'J': 'jh',
    'W': 'ṭh', 'Q': 'ḍh', 'T': 'th', 'D': 'dh',
    'P': 'ph', 'B': 'bh',
    'N': 'ṅ', 'Y': 'ñ', 'w': 'ṭ', 'q': 'ḍ', 'R': 'ṇ',
    'S': 'ś', 'z': 'ṣ', 'H': 'ḥ', 'M': 'ṃ',
}

# Longest-first so 'ṭh' is consumed before 'ṭ', and 'ai' before 'a'.
IAST_TO_SLP1 = sorted(((v, k) for k, v in SLP1_TO_IAST.items()),
                      key=lambda p: -len(p[0]))
_MAX_IAST = max(len(v) for v in SLP1_TO_IAST.values())


def slp1_to_iast(s):
    return ''.join(SLP1_TO_IAST.get(ch, ch) for ch in s)


def iast_to_slp1(s):
    out = []
    i = 0
    n = len(s)
    while i < n:
        for src, dst in IAST_TO_SLP1:
            if s.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            out.append(s[i])
            i += 1
    return ''.join(out)


def assert_roundtrip(strings, label="input"):
    """SLP1 -> IAST -> SLP1 must be the identity. Raises on the first failure.

    Run on the actual corpus rather than a hand-written sample: the characters
    that break a transliterator are the rare ones, and a sample of clean text
    will not contain them.
    """
    bad = []
    for s in strings:
        if iast_to_slp1(slp1_to_iast(s)) != s:
            bad.append(s)
            if len(bad) >= 5:
                break
    if bad:
        raise AssertionError(
            f"{label}: SLP1->IAST->SLP1 is not lossless on {len(bad)}+ strings. "
            f"First: {bad[0]!r} -> {slp1_to_iast(bad[0])!r} -> "
            f"{iast_to_slp1(slp1_to_iast(bad[0]))!r}")
    return True


if __name__ == "__main__":
    import csv
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sighum_test_4200.tsv"
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig"), delimiter="\t"))
    ins = [r["input"] for r in rows]
    outs = [r["output"] for r in rows]
    assert_roundtrip(ins, "input column")
    assert_roundtrip(outs, "output column")
    print(f"round-trip lossless on {len(ins):,} inputs and {len(outs):,} outputs")
    print(f"  {ins[0]}")
    print(f"  {slp1_to_iast(ins[0])}")
