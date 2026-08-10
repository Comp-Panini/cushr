#!/usr/bin/env python3
"""Derive the SHR -> DCS lemma rewrite table from the training split.

cuSHR reports the lemma stored on the lattice node it selected, which follows
SHR's convention. The reference follows DCS's. For participles the two disagree
systematically -- SHR gives the participial stem, DCS the verbal root:

    ukta -> vac      gata -> gam      kfta -> kf      smfta -> smf

Measured on the test oracle, that one pattern accounts for most of a 20-point
sentence-level L gap against ByT5: per-word lemma accuracy is 94.83% (cuSHR
oracle) vs 99.34% (ByT5), and perfect match compounds it over ~6.78 words.

98.7% of SHR lemma types map to exactly one DCS lemma, so the correspondence is
close to a function and a lookup applied at output time recovers most of the
gap without touching the lattice, the model, or the search.

TRAINING SPLIT ONLY. Deriving this from test and then scoring on test would be
leakage and would overstate the achievable gain. The split file is read
directly and test/dev ids are asserted absent, rather than assumed absent.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ingest"))
import ingest as ig  # noqa: E402


def load_vocab(path):
    return [l.split("\t", 1)[1].rstrip("\n") if "\t" in l else ""
            for l in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data/cushr_data_g95.npz")
    ap.add_argument("--index", default="../data/sentence_index_repaired.json")
    ap.add_argument("--splits", default="splits95_ex4200.json")
    ap.add_argument("--lemma-vocab", default="../data/lemma_vocabulary.txt")
    ap.add_argument("--p-dir", default="../../SIGHUM_database_gold_path/DCS_pick")
    ap.add_argument("--out", default="lemma_map.json")
    ap.add_argument("--sample", type=int, default=25000,
                    help="training sentences to read; pickle IO is the "
                         "bottleneck, and the table saturates well before the "
                         "full split")
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
    lv = load_vocab(args.lemma_vocab)

    random.seed(args.seed)
    sample = random.sample(train, min(args.sample, len(train)))
    # Leakage guard: asserted, not assumed.
    leaked = [s for s in sample if s in forbidden]
    if leaked:
        raise SystemExit(f"{len(leaked)} sampled ids are in dev/test: "
                         f"{leaked[:5]}")

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
        gold = [ig.normalize_lemma(str(w))
                for cl in getattr(d, "lemmas", []) for w in cl]
        nodes = sorted(gn[go[s]:go[s + 1]].tolist(), key=lambda x: int(cs[x]))
        pred = [lv[lid[x]] for x in nodes]
        if len(pred) != len(gold):
            continue
        used += 1
        for a, b in zip(pred, gold):
            pairs[a][b] += 1

    table, ambiguous = {}, 0
    for shr, c in pairs.items():
        top, cnt = c.most_common(1)[0]
        if len(c) > 1:
            ambiguous += 1
        # Only confident, non-identity rewrites. An identity entry is a no-op,
        # and a low-support or inconsistent one risks corrupting a correct lemma.
        if top != shr and cnt >= args.min_count and \
                cnt / sum(c.values()) >= args.min_ratio:
            table[shr] = top

    blob = {
        "_provenance": {
            "built_from": "training split only",
            "splits_file": args.splits,
            "train_sentences_sampled": len(sample),
            "train_sentences_used": used,
            "min_count": args.min_count,
            "min_ratio": args.min_ratio,
            "shr_lemma_types_seen": len(pairs),
            "types_with_one_dcs_lemma": len(pairs) - ambiguous,
            "rules": len(table),
        },
        "map": table,
    }
    json.dump(blob, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, sort_keys=True)

    print(f"train sentences used     : {used:,}")
    print(f"SHR lemma types seen     : {len(pairs):,}")
    print(f"  map to one DCS lemma   : {len(pairs) - ambiguous:,} "
          f"({100 * (len(pairs) - ambiguous) / len(pairs):.1f}%)")
    print(f"rewrite rules kept       : {len(table):,}")
    print("examples                 : " +
          ", ".join(f"{k}->{v}" for k, v in list(table.items())[:10]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
