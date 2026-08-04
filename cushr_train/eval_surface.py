#!/usr/bin/env python3
"""Score predictions as a word sequence, the way segmentation papers do.

Our internal metric (`compare_featurizers.evaluate_detailed`) compares sets of
LATTICE NODE IDS. A node is (chunk, lemma, cng, form), so a prediction only
counts if the lemma and the morphological tag are also correct -- it measures
joint segmentation + morphological analysis. Published Sanskrit segmentation
numbers score the output WORD STRING. Ours is the strictly harder event, so the
two are not comparable and ours will read lower.

This scores the segmentation alone: decode, map each predicted node to its
surface form, and compare that sequence against the reference `output` column.

It also removes a second bias. The node metric can only score sentences whose
gold path our ingest managed to resolve, silently dropping the rest (567 of the
4,200 here) -- precisely the hard ones. A string metric scores whatever the
decoder emits, so every reference sentence counts.

Reported:
  PM        exact match of the whole word sequence (the headline)
  P/R/F1    multiset overlap of word tokens
"""
import argparse
import csv
import json

import numpy as np
import torch

from dataset import LatticeStore, collate
from model import BiaffineEdgeScorer
from train import batches, to_torch
from viterbi import viterbi, predicted_nodes


def load_forms(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        out.append(line.split("\t", 1)[1] if "\t" in line else "")
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tsv", default="sighum_test_4200.tsv")
    ap.add_argument("--index", default="../data/sentence_index_repaired.json")
    ap.add_argument("--raw", default="../data/cushr_data_repaired.npz")
    ap.add_argument("--vocab", default="../data/form_vocabulary.txt")
    ap.add_argument("--dump", default="", help="write per-sentence predictions")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.tsv, encoding="utf-8-sig"),
                               delimiter="\t"))
    index = json.load(open(args.index))
    pos = {s: i for i, s in enumerate(index)}
    forms = load_forms(args.vocab)
    _raw = np.load(args.raw)
    raw_fid = _raw["node_form_id"]
    # `predicted_nodes` walks the path edges as viterbi backtracked them, which
    # comes out reversed (and collate has already reordered edges by dst topo
    # level, so path-edge order is not reading order in general). Sorting by the
    # node's character offset recovers the sentence order regardless.
    char_start = _raw["node_char_start"]

    pairs = [(pos[r["DCS-ID"].strip()], r["output"].split())
             for r in rows if r["DCS-ID"].strip() in pos]
    ref = dict(pairs)
    sids = np.asarray([s for s, _ in pairs], dtype=np.int64)
    print(f"{args.tsv}: {len(rows):,} rows, {len(sids):,} located in our corpus")

    store = LatticeStore(args.cache)
    dev = torch.device("cpu")
    m = np.load(args.model)
    net = BiaffineEdgeScorer(int(m["feat_dim"]), int(m["hidden"])).to(dev)
    net.src_proj.weight.data = torch.as_tensor(m["src_proj"], device=dev)
    net.dst_proj.weight.data = torch.as_tensor(m["dst_proj"], device=dev)
    net.bias.data = torch.as_tensor(np.asarray(m["bias"]).reshape(1), device=dev)
    net.eval()

    pm = n = 0
    tp = fp = fn = 0
    # TransLIST (Sandhan et al. 2022, §3) reports MACRO-averaged word-level
    # P/R/F -- computed per sentence, then averaged over sentences. Micro
    # (pooling tp/fp/fn corpus-wide) gives a different, usually higher number on
    # this task because long sentences dominate. Both are reported here so the
    # comparison against their table uses the matching one.
    mp, mr, mf = [], [], []
    # Split by whether ingest resolved a gold path for the sentence. The node
    # metric can only score the resolved ones; this metric scores all of them,
    # and the difference between the two groups is exactly the bias that
    # dropping them would introduce.
    has_gold = (store.gold_off[1:] - store.gold_off[:-1]) > 0
    by_group = {"gold-resolved": [0, 0], "no-gold-path": [0, 0]}
    dump = []
    for chunk in batches(sids, 128, shuffle=False):
        b = collate(store, chunk)
        t = to_torch(b, dev)
        pe, pmask, _ = viterbi(t, net(t["feats"], t["src"], t["dst"],
                                      t.get("ids"), t))
        pred_local = predicted_nodes(t, pe, pmask)
        gn = np.asarray(b["global_node"])
        for i, s in enumerate(chunk):
            nodes = sorted((int(gn[x]) for x in pred_local[i]),
                           key=lambda g: int(char_start[g]))
            words = [forms[raw_fid[x]] for x in nodes]
            words = [w for w in words if w]
            gold = ref[int(s)]
            pm += int(words == gold)
            n += 1
            # multiset token overlap
            from collections import Counter
            cp, cg = Counter(words), Counter(gold)
            inter = sum((cp & cg).values())
            tp += inter
            fp += sum(cp.values()) - inter
            fn += sum(cg.values()) - inter
            p_i = inter / len(words) if words else 0.0
            r_i = inter / len(gold) if gold else 0.0
            mp.append(p_i)
            mr.append(r_i)
            mf.append(2 * p_i * r_i / (p_i + r_i) if p_i + r_i else 0.0)
            bucket = ("gold-resolved" if has_gold[int(s)] else "no-gold-path")
            by_group[bucket][0] += int(words == gold)
            by_group[bucket][1] += 1
            if args.dump:
                dump.append({"id": index[int(s)], "pred": words, "gold": gold,
                             "exact": words == gold, "group": bucket})

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"\nsurface-string metric over {n:,} sentences "
          f"(every reference sentence scored, gold path or not)")
    print(f"  perfect match     : {100 * pm / n:.2f}   ({pm:,}/{n:,})")
    print(f"  word P/R/F MACRO  : {100 * np.mean(mp):.2f} / "
          f"{100 * np.mean(mr):.2f} / {100 * np.mean(mf):.2f}   "
          f"<- matches TransLIST's protocol")
    print(f"  word P/R/F micro  : {100 * prec:.2f} / {100 * rec:.2f} / "
          f"{100 * f1:.2f}")

    print("  by ingest gold-path status:")
    for k, (h, tot) in by_group.items():
        if tot:
            print(f"    {k:<16} PM {100 * h / tot:6.2f}   n={tot:,}")

    if args.dump:
        json.dump(dump, open(args.dump, "w"), ensure_ascii=False)
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
