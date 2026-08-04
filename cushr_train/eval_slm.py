#!/usr/bin/env python3
"""Score at S, S+L, S+M and S+L+M levels, matching ByT5-Sanskrit's task ladder.

ByT5-Sanskrit (Nehrdich et al. 2024, Table 7) reports sentence-level perfect
match separately for segmentation (S), lemmatisation (L), and combinations with
morphosyntactic tagging (M). Our lattice nodes carry all three -- a node is
(surface form, lemma, cng) -- so selecting a path commits to all of them at
once, and we can report the same ladder by deciding how much of each node has to
match.

  S        surface form sequence only
  S+L      form and lemma
  S+M      form and morph tag (cng)
  S+L+M    the whole node identity

Reference is OUR reconstructed gold path, not the TSV, because the TSV carries
only segmentation. That restricts this to sentences whose gold path ingest could
resolve, which is why the sentence count here is lower than in eval_surface.py
and why S here is not identical to S there.
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tsv", default="sighum_test_4200.tsv")
    ap.add_argument("--index", default="../data/sentence_index_repaired.json")
    ap.add_argument("--raw", default="../data/cushr_data_repaired.npz")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.tsv, encoding="utf-8-sig"),
                               delimiter="\t"))
    index = json.load(open(args.index))
    pos = {s: i for i, s in enumerate(index)}

    z = np.load(args.raw)
    fid, lid, tag = z["node_form_id"], z["node_lemma_id"], z["node_features"]
    cstart = z["node_char_start"]

    store = LatticeStore(args.cache)
    have = (store.gold_off[1:] - store.gold_off[:-1]) > 0
    sids = np.asarray([pos[r["DCS-ID"].strip()] for r in rows
                       if r["DCS-ID"].strip() in pos
                       and have[pos[r["DCS-ID"].strip()]]], dtype=np.int64)
    print(f"{args.tsv}: {len(rows):,} rows; {len(sids):,} have a resolved gold "
          f"path and can be scored at L/M level "
          f"({len(rows) - len(sids):,} dropped)")

    dev = torch.device("cpu")
    m = np.load(args.model)
    net = BiaffineEdgeScorer(int(m["feat_dim"]), int(m["hidden"])).to(dev)
    net.src_proj.weight.data = torch.as_tensor(m["src_proj"], device=dev)
    net.dst_proj.weight.data = torch.as_tensor(m["dst_proj"], device=dev)
    net.bias.data = torch.as_tensor(np.asarray(m["bias"]).reshape(1), device=dev)
    net.eval()

    levels = {"S": lambda g: fid[g],
              "S+L": lambda g: (fid[g], lid[g]),
              "S+M": lambda g: (fid[g], tag[g]),
              "S+L+M": lambda g: (fid[g], lid[g], tag[g])}
    hit = {k: 0 for k in levels}
    n = 0
    for chunk in batches(sids, 128, shuffle=False):
        b = collate(store, chunk)
        t = to_torch(b, dev)
        pe, pmask, _ = viterbi(t, net(t["feats"], t["src"], t["dst"],
                                      t.get("ids"), t))
        pred_local = predicted_nodes(t, pe, pmask)
        gn = np.asarray(b["global_node"])
        for i, s in enumerate(chunk):
            # sort by character offset: predicted_nodes returns the path as
            # viterbi backtracked it, which is reversed.
            pred = sorted((int(gn[x]) for x in pred_local[i]),
                          key=lambda g: int(cstart[g]))
            gold = sorted(store.gold_nodes[store.gold_off[s]:store.gold_off[s + 1]]
                          .tolist(), key=lambda g: int(cstart[g]))
            for k, f in levels.items():
                hit[k] += int([f(g) for g in pred] == [f(g) for g in gold])
            n += 1

    print(f"\nsentence-level perfect match over {n:,} sentences")
    for k in ("S", "S+L", "S+M", "S+L+M"):
        print(f"  {k:<6} {100 * hit[k] / n:6.2f}")


if __name__ == "__main__":
    main()
