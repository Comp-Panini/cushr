#!/usr/bin/env python3
"""Evaluate a model on an explicit list of DCS ids, split by our own train/dev/test.

Written for external benchmark sets (e.g. sighum_test_4200.tsv) whose sentences
were assigned to splits by somebody else. Our splits come from md5 bucketing on
sentence index (`prepare.bucket`), so an external test set will in general
overlap our TRAINING data. Reporting a single number over such a set is
meaningless; this script therefore always breaks the result down by which of our
splits each sentence landed in, so contamination is visible rather than averaged
away.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from compare_featurizers import evaluate_detailed, train_form_counts
from dataset import LatticeStore
from model import BiaffineEdgeScorer
from prepare import bucket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tsv", required=True, help="TSV with a DCS-ID column")
    ap.add_argument("--index", default="../data/sentence_index_repaired.json")
    ap.add_argument("--raw", default="../data/cushr_data_repaired.npz")
    ap.add_argument("--dev-pct", type=int, default=5)
    ap.add_argument("--test-pct", type=int, default=5)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.tsv, encoding="utf-8-sig"),
                               delimiter="\t"))
    want = [r["DCS-ID"].strip() for r in rows]

    index = json.load(open(args.index))
    pos = {s: i for i, s in enumerate(index)}
    store = LatticeStore(args.cache)
    have_gold = (store.gold_off[1:] - store.gold_off[:-1]) > 0

    groups = {"train (SEEN in training)": [], "dev (used for selection)": [],
              "test (truly held out)": []}
    missing = no_gold = 0
    for d in want:
        if d not in pos:
            missing += 1
            continue
        s = pos[d]
        if not have_gold[s]:
            no_gold += 1
            continue
        b = bucket(s)
        if b < args.test_pct:
            groups["test (truly held out)"].append(s)
        elif b < args.test_pct + args.dev_pct:
            groups["dev (used for selection)"].append(s)
        else:
            groups["train (SEEN in training)"].append(s)

    print(f"{args.tsv}: {len(want):,} sentences")
    print(f"  not in our corpus     : {missing:,}")
    print(f"  no resolved gold path : {no_gold:,}  (cannot be scored here)")
    for k, v in groups.items():
        print(f"  {k:<26}: {len(v):,}")

    dev = torch.device("cpu")
    m = np.load(args.model)
    net = BiaffineEdgeScorer(int(m["feat_dim"]), int(m["hidden"])).to(dev)
    net.src_proj.weight.data = torch.as_tensor(m["src_proj"], device=dev)
    net.dst_proj.weight.data = torch.as_tensor(m["dst_proj"], device=dev)
    net.bias.data = torch.as_tensor(np.asarray(m["bias"]).reshape(1), device=dev)
    net.eval()
    form_counts, node_form_id = train_form_counts(args.raw, args.cache)

    print(f"\n{'group':<28} {'F1':>8} {'PM':>8} {'n':>7}")
    allids = []
    for k, v in groups.items():
        if not v:
            continue
        allids += v
        r = evaluate_detailed(net, store, np.asarray(v), dev,
                              form_counts, node_form_id)
        print(f"{k:<28} {r['f1']:8.4f} {r['perfect_match']:8.4f} {r['n']:7,}")
    if allids:
        r = evaluate_detailed(net, store, np.asarray(allids), dev,
                              form_counts, node_form_id)
        print(f"{'ALL (contaminated)':<28} {r['f1']:8.4f} "
              f"{r['perfect_match']:8.4f} {r['n']:7,}")


if __name__ == "__main__":
    main()
