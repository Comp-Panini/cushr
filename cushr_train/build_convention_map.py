#!/usr/bin/env python3
"""Derive a joint (SHR lemma, cng) -> (DCS lemma, cng) convention table.

WHY A JOINT TABLE AND NOT TWO SEPARATE ONES
-------------------------------------------
`build_lemma_map.py` translates the lemma alone, which lifted ORACLE L from
70.07 to 89.80. It could not touch M, because rewriting a lemma string does not
change the `cng` of the node that was selected — so S+M stayed at 68.23.

The two disagree together, not independently: of 1,544 oracle tag errors, 1,438
(93%) coincide with a lemma error and **100% of those want a negative `cng`**.
One phenomenon — SHR reads the word as a nominal (`dfzwa`, `nom. sg. n.`), DCS
reads it as a participle (`dfS`, `-190`) — showing up in two fields. So it wants
one table over the pair.

Measured on 4,000 training sentences, keying on the pair is not merely as
functional as keying on the lemma but *more* so, because `cng` disambiguates:

    SHR lemma        -> DCS lemma          99.26% over  4,748 keys
    (SHR lemma, cng) -> (DCS lemma, cng)   99.75% over 10,259 keys

WHAT THIS IS AND IS NOT
-----------------------
It is a *translation between two annotation traditions*, both of which describe
the word correctly — "nominative of the participial stem dfzwa" and "past
passive participle of root dfS" are the same claim in different vocabularies.
Applying it at output time is normalisation, not correction.

It is NOT a fix for wrong node selection. Where the model picked the wrong node
outright — 74.3% of its errors are same-form, same-lemma, wrong `cng`, mostly
`nom. sg. n.` vs `acc. sg. n.` — translating a wrong analysis yields a wrong
analysis in the other vocabulary. Those need a scorer change, not a table.

TRAINING SPLIT ONLY; dev/test membership is asserted absent, not assumed.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ingest"))
import ingest as ig  # noqa: E402

SEP = "\t"   # JSON keys must be strings; "<lemma>\t<cng>" is unambiguous
             # because neither field can contain a tab.

NS = "{http://graphml.graphdrawing.org/xmlns}"


def node_cng(path):
    """Ordered [(node id, cng)] without building a networkx graph.

    Only the `cng` attribute is needed here, and `nx.read_graphml` pays to
    construct an entire graph object. Verified identical to the networkx path
    on 200 files (0 mismatches) at 2.7x the speed -- graphml parsing is ~95% of
    this script's runtime, so that is what makes a 25k-sentence build practical.

    Node order must match `sorted(G.nodes(), key=ig._node_id)` exactly, because
    the caller maps position i to global node `base + 1 + i`. Getting that order
    wrong would silently pair every cng with the wrong node.
    """
    key = None
    out = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == NS + "key" and el.get("attr.name") == "cng":
            key = el.get("id")
        elif el.tag == NS + "node":
            c = ""
            for d in el.findall(NS + "data"):
                if d.get("key") == key:
                    c = d.text or ""
                    break
            out.append((el.get("id"), c))
            el.clear()
    try:
        out.sort(key=lambda p: ig._node_id(p[0]))
    except ValueError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data/cushr_data_g95.npz")
    ap.add_argument("--index", default="../data/sentence_index_repaired.json")
    ap.add_argument("--splits", default="splits95_ex4200.json")
    ap.add_argument("--lemma-vocab", default="../data/lemma_vocabulary.txt")
    ap.add_argument("--graphml-dir", default="../../SIGHUM_database/After_graphml")
    ap.add_argument("--p-dir", default="../../SIGHUM_database_gold_path/DCS_pick")
    ap.add_argument("--out", default="convention_map.json")
    ap.add_argument("--sample", type=int, default=25000)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--min-ratio", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index = json.load(open(args.index))
    splits = json.load(open(args.splits))
    train = sorted(set(splits["train"]))
    forbidden = set(splits.get("dev", [])) | set(splits.get("test", []))

    z = np.load(args.raw)
    lid, cs = z["node_lemma_id"], z["node_char_start"]
    go = z["gold_path_offsets"].astype(np.int64)
    gn = z["gold_path_nodes"].astype(np.int64)
    soff = z["sentenceoffsets"]
    nsn = np.append(soff[1:], len(lid)) - soff
    lv = [l.split("\t", 1)[1].rstrip("\n") if "\t" in l else ""
          for l in open(args.lemma_vocab, encoding="utf-8")]

    random.seed(args.seed)
    sample = random.sample(train, min(args.sample, len(train)))
    leaked = [s for s in sample if s in forbidden]
    if leaked:
        raise SystemExit(f"{len(leaked)} sampled ids are in dev/test: {leaked[:5]}")

    pairs = defaultdict(Counter)
    used = 0
    for s in sample:
        if go[s + 1] - go[s] == 0:
            continue
        try:
            with open(os.path.join(args.p_dir, index[s] + ".p"), "rb") as f:
                d = ig._DCSUnpickler(f, encoding="utf-8").load()
        except Exception:
            continue
        g_lem = [ig.normalize_lemma(str(w))
                 for cl in getattr(d, "lemmas", []) for w in cl]
        g_cng = [str(c) for cl in getattr(d, "cng", []) for c in cl]
        nodes = sorted(gn[go[s]:go[s + 1]].tolist(), key=lambda x: int(cs[x]))
        if len(nodes) != len(g_lem):
            continue
        # cng lives in the graphml, not the archive -- same source eval_slm uses
        nl = node_cng(os.path.join(args.graphml_dir, index[s] + ".graphml"))
        base = int(soff[s])
        cm = {base + 1 + i: c
              for i, (_, c) in enumerate(nl) if base + 1 + i < base + nsn[s] - 1}
        used += 1
        if used % 2000 == 0:
            print(f"  ... {used:,} sentences, {len(pairs):,} keys", flush=True)
        for nd, tl, tc in zip(nodes, g_lem, g_cng):
            pairs[(lv[lid[nd]], cm.get(nd, ""))][(tl, tc)] += 1

    table, ambiguous, identity = {}, 0, 0
    for key, c in pairs.items():
        top, cnt = c.most_common(1)[0]
        if len(c) > 1:
            ambiguous += 1
        if top == key:
            identity += 1
            continue          # no-op; keeps the table small and auditable
        if cnt >= args.min_count and cnt / sum(c.values()) >= args.min_ratio:
            table[key[0] + SEP + key[1]] = [top[0], top[1]]

    n_lem = sum(1 for k, v in table.items() if k.split(SEP)[0] != v[0])
    n_cng = sum(1 for k, v in table.items() if k.split(SEP)[1] != v[1])
    json.dump({"_provenance": {
                   "direction": "(SHR lemma, SHR cng) -> (DCS lemma, DCS cng)",
                   "built_from": "training split only",
                   "splits_file": args.splits,
                   "train_sentences_used": used,
                   "min_count": args.min_count,
                   "min_ratio": args.min_ratio,
                   "keys_seen": len(pairs),
                   "keys_already_identical": identity,
                   "keys_ambiguous": ambiguous,
                   "rules": len(table),
                   "rules_changing_lemma": n_lem,
                   "rules_changing_cng": n_cng},
               "map": table},
              open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, sort_keys=True)

    print(f"train sentences used   : {used:,}")
    print(f"(lemma, cng) keys seen : {len(pairs):,}")
    print(f"  already identical    : {identity:,}")
    print(f"  ambiguous            : {ambiguous:,}")
    print(f"rules kept             : {len(table):,}  "
          f"({n_lem:,} change the lemma, {n_cng:,} change the cng)")
    ex = list(table.items())[:6]
    for k, v in ex:
        a, b = k.split(SEP)
        print(f"    ({a}, {b})  ->  ({v[0]}, {v[1]})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
