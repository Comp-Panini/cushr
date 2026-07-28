#!/usr/bin/env python3
"""Reconstruct the raw surface text of each chunk from lattice node spans.

The SIGHUM distribution ships only graphml lattices -- there is no raw text
file anywhere under After_graphml/. But every node carries an exact character
span into its chunk:

    chunk_no    (d1)  which whitespace-delimited chunk this candidate sits in
    position    (d3)  0-based start offset of the candidate within that chunk
    length_word (d6)  span length in characters
    word        (d2)  the candidate's own (unsandhied) spelling

Candidates overlap heavily, so most character positions are covered by several
nodes. Where sandhi did not alter a character, every covering node agrees on
it; where it did, the analyses disagree -- but only at span edges, and the
majority still reports the surface character. A per-character majority vote
therefore recovers the surface string.

Measured over 3,000 files: ~20% of positions receive more than one proposed
character, but that figure is dominated by candidates whose `word` is a
shifted substring rather than by genuine sandhi, and the vote resolves them --
output is correct SLP1 (tatra, sarvatra, vaktavyam, manyante, SAstfakovidAH).
About 4% of positions end up with no evidence at all and are emitted as '?':
these are junction characters that sandhi consumed, so no analysis spells
them. They are marked rather than dropped so a caller can see the gap instead
of silently reading a shifted string. Run `python surface.py --verify` to
re-measure.

Scope note: the scalar featurizers do *not* depend on this module -- they read
each node's own `word` attribute and its declared span. Character-level
reconstruction exists for Hellwig-style n-gram split probabilities, which need
the raw string and for which '?' is an acceptable unknown-character marker.
"""

import argparse
import collections
import glob
import os
import xml.etree.ElementTree as ET

GRAPHML_NS = "{http://graphml.graphdrawing.org/xmlns}"


def reconstruct_chunks(node_attrs):
    """Majority-vote the surface string of every chunk.

    node_attrs: iterable of dicts carrying 'chunk_no', 'position',
                'length_word' and 'word' (the graphml node attributes, exactly
                as ingest.py already holds them).

    Returns {chunk_no: surface_string}. Positions no node covers become '?'
    so a caller can spot holes instead of silently shifting later characters.
    """
    votes, ambiguous, total, extent = _tally(node_attrs)
    out = {}
    for chunk, positions in votes.items():
        # Take the chunk's length from the widest span any candidate *declares*,
        # not from the widest position actually voted on. Otherwise a chunk
        # whose final character only appears in sandhied form (surface
        # `kOravyaH` analysed as `kOravya`) is silently truncated instead of
        # reporting the gap.
        width = max(extent.get(chunk, 0), (max(positions) + 1) if positions else 0)
        text = []
        for p in range(width):
            counter = positions.get(p)
            text.append(counter.most_common(1)[0][0] if counter else "?")
        out[chunk] = "".join(text)
    return out


# Weight of a vote from a candidate whose spelling length matches its declared
# span exactly (direct surface evidence) versus one where it does not.
EXACT_WEIGHT = 4
PREFIX_WEIGHT = 1


def _tally(node_attrs):
    """Shared counting core: {chunk: {pos: Counter(char)}}, plus vote stats."""
    votes = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    extent = collections.defaultdict(int)   # chunk -> declared width
    ambiguous = 0
    total = 0
    for attrs in node_attrs:
        try:
            chunk = int(attrs.get("chunk_no"))
            pos = int(attrs.get("position"))
            length = int(attrs.get("length_word"))
        except (TypeError, ValueError):
            continue
        word = attrs.get("word") or ""
        # ~15% of candidates spell a word whose length disagrees with the span
        # it claims, because sandhi rewrote the junction (surface `kOravyaH`,
        # span 8, analysed as `kOravya`). Dropping them outright left 5% of
        # positions with no evidence at all. Sandhi acts on word *endings*, so
        # the shared prefix is still trustworthy: lay down the overlapping
        # prefix at a lower weight, letting exact-length candidates outvote it
        # wherever both exist.
        weight = EXACT_WEIGHT if len(word) == length else PREFIX_WEIGHT
        extent[chunk] = max(extent[chunk], pos + length)
        for i, ch in enumerate(word[:length]):
            votes[chunk][pos + i][ch] += weight
    for positions in votes.values():
        for counter in positions.values():
            total += 1
            if len(counter) > 1:
                ambiguous += 1
    return votes, ambiguous, total, extent


def chunk_stats(node_attrs):
    """(n_positions, n_ambiguous) -- for the --verify audit."""
    _, ambiguous, total, _ = _tally(node_attrs)
    return total, ambiguous


def read_graphml_nodes(path):
    """Yield node attribute dicts from a graphml file (verify path only).

    ingest.py goes through networkx; this is a dependency-free reader so the
    audit can run standalone.
    """
    root = ET.parse(path).getroot()
    keys = {k.get("id"): k.get("attr.name") for k in root.findall(f"{GRAPHML_NS}key")}
    for node in root.iter(f"{GRAPHML_NS}node"):
        yield {keys.get(d.get("key")): (d.text or "") for d in node}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphml-dir", default="../../SIGHUM_database/After_graphml")
    ap.add_argument("--verify", action="store_true",
                    help="report ambiguity rate and print sample reconstructions")
    ap.add_argument("--limit", type=int, default=0,
                    help="only read the first N files (0 = all)")
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.graphml_dir, "*.graphml")))
    if not files:
        raise SystemExit(f"no graphml files under {args.graphml_dir}")
    if args.limit:
        files = files[:args.limit]

    total = ambiguous = holes = n_chunks = 0
    samples = []
    for path in files:
        attrs = list(read_graphml_nodes(path))
        t, a = chunk_stats(attrs)
        total += t
        ambiguous += a
        for chunk, text in sorted(reconstruct_chunks(attrs).items()):
            n_chunks += 1
            holes += text.count("?")
            if len(samples) < args.samples:
                samples.append((os.path.basename(path), chunk, text))

    print(f"files:            {len(files):,}")
    print(f"chunks:           {n_chunks:,}")
    print(f"char positions:   {total:,}")
    rate = (ambiguous / total) if total else 0.0
    print(f"ambiguous:        {ambiguous:,}  ({rate:.1%} -- resolved by majority vote)")
    print(f"uncovered ('?'):  {holes:,}")
    if args.verify:
        print("\nsample reconstructions:")
        for name, chunk, text in samples:
            print(f"  {name} chunk {chunk}: {text}")


if __name__ == "__main__":
    main()
