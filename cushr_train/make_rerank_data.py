#!/usr/bin/env python3
"""Dump K-best candidate lists with correctness labels, for reranker training.

WHAT THE LABEL IS, AND WHY IT IS THE GOLD PATH RATHER THAN DCS
--------------------------------------------------------------
A candidate is labelled correct when its node sequence equals OUR OWN GOLD
PATH exactly. It is tempting to label against DCS instead, since S+M is what
gets reported -- but the division of labour in this pipeline makes the gold
path the right target:

  - the reranker's only job is to choose a node sequence;
  - the SHR->DCS convention maps then translate whatever it chose at output
    time (eval_slm.py's --lemma-map / --convention-map).

So "pick the node the gold path picks" is precisely the reranker's share of the
work, and the 22-point oracle gap (ORACLE S+M 77.65) is the maps' share. Those
were already measured to be independent: the convention tables raised floor and
ceiling by the same factor, leaving %-of-ceiling unmoved at 67%.

It is also the only target available. The external SIGHUM reference covers the
4,200 test sentences only; the gold path exists for 96.61% of the corpus, which
is where the 104,159 training sentences are.

The claim that this transfers -- that fitting gold-path match raises reported
S+M -- is not assumed. `rerank.py` measures S+M on test through the unmodified
eval_slm scorer, and that is the number that counts.

THE JACKKNIFE QUESTION
----------------------
Decoding the train split with the scorer trained on it can yield
unrealistically clean candidate lists: if the base model has memorised train,
the gold path sits at rank 1 far more often there than at test time, the
reranker sees too few examples that need fixing, and it learns nothing beyond
"trust the base score."

Rather than assume it, this script prints recall@1 and recall@K per split. If
train recall@1 is close to dev recall@1 the effect is small and train is usable
directly; a large gap means 5-fold jackknifed decoding is required.
"""
import argparse

import numpy as np
import torch

from dataset import LatticeStore, collate
from kbest import kbest, predicted_nodes_k
from model import BiaffineEdgeScorer
from train import batches, to_torch


def load_net(model_path, dev):
    m = np.load(model_path)
    net = BiaffineEdgeScorer(int(m["feat_dim"]), int(m["hidden"])).to(dev)
    net.src_proj.weight.data = torch.as_tensor(m["src_proj"], device=dev)
    net.dst_proj.weight.data = torch.as_tensor(m["dst_proj"], device=dev)
    net.bias.data = torch.as_tensor(np.asarray(m["bias"]).reshape(1), device=dev)
    net.eval()
    return net


@torch.no_grad()
def dump(store, net, dev, ids, K, cstart, fid, forms, batch_size=128,
         progress_every=20000):
    nodes_flat, off, c_sent, c_score, c_label = [], [0], [], [], []
    sent_ids, n_hit1, n_hitK = [], 0, 0
    for chunk in batches(ids, batch_size, shuffle=False):
        b = collate(store, chunk)
        t = to_torch(b, dev)
        w = net(t["feats"], t["src"], t["dst"], t.get("ids"), t)
        ke, km, sc, va = kbest(t, w, K)
        cands = predicted_nodes_k(t, ke, km, va)
        gn = np.asarray(b["global_node"])
        for i, s in enumerate(chunk):
            s = int(s)
            gold = sorted(
                store.gold_nodes[store.gold_off[s]:store.gold_off[s + 1]].tolist(),
                key=lambda g: int(cstart[g]))
            gold = tuple(g for g in gold if forms[fid[g]])
            si = len(sent_ids)
            sent_ids.append(s)
            hit1 = hitK = False
            for k, loc in enumerate(cands[i]):
                seq = sorted((int(gn[x]) for x in loc),
                             key=lambda g: int(cstart[g]))
                seq = [g for g in seq if forms[fid[g]]]
                ok = tuple(seq) == gold
                nodes_flat.extend(seq)
                off.append(len(nodes_flat))
                c_sent.append(si)
                c_score.append(float(sc[i, k]))
                c_label.append(ok)
                hitK |= ok
                hit1 |= ok and k == 0
            n_hit1 += hit1
            n_hitK += hitK
            if len(sent_ids) % progress_every == 0:
                print(f"    ... {len(sent_ids):,} sentences", flush=True)
    return {
        "cand_nodes": np.asarray(nodes_flat, dtype=np.int32),
        "cand_off": np.asarray(off, dtype=np.int64),
        "cand_sent": np.asarray(c_sent, dtype=np.int32),
        "cand_score": np.asarray(c_score, dtype=np.float32),
        "cand_label": np.asarray(c_label, dtype=bool),
        "sent_ids": np.asarray(sent_ids, dtype=np.int32),
    }, n_hit1, n_hitK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache95_ctx_ex4200")
    ap.add_argument("--model", default="model95_ctx_ex4200.npz")
    ap.add_argument("--raw", default="../data/cushr_data_g95.npz")
    ap.add_argument("--form-vocab", default="../data/form_vocabulary.txt")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap sentences per split (0 = all). Sampled at RANDOM, "
                         "not truncated: sentence ids run in corpus order, so "
                         "the first N of each split are different source texts "
                         "and comparing splits on them measures the texts "
                         "rather than the model.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="rerank")
    args = ap.parse_args()

    store = LatticeStore(args.cache)
    dev = torch.device("cpu")
    net = load_net(args.model, dev)
    z = np.load(args.raw)
    cstart, fid = z["node_char_start"], z["node_form_id"]
    forms = [l.split("\t", 1)[1].rstrip("\n") if "\t" in l else ""
             for l in open(args.form_vocab, encoding="utf-8")]

    print(f"K={args.k}   label = candidate node sequence equals the gold path\n")
    print(f"{'split':<8}{'sentences':>11}{'recall@1':>10}{'recall@'+str(args.k):>11}")
    rows = {}
    for sp in args.splits.split(","):
        ids = store.trainable(sp)
        if args.limit and args.limit < len(ids):
            rng = np.random.default_rng(args.seed)
            ids = np.sort(rng.choice(ids, args.limit, replace=False))
        data, h1, hK = dump(store, net, dev, ids, args.k, cstart, fid, forms)
        n = len(data["sent_ids"])
        rows[sp] = (100 * h1 / n, 100 * hK / n)
        print(f"{sp:<8}{n:>11,}{100*h1/n:>9.2f}%{100*hK/n:>10.2f}%")
        out = f"{args.out_prefix}_{sp}.npz"
        np.savez_compressed(out, **data)
        print(f"        -> {out}  ({len(data['cand_sent']):,} candidates)")

    if "train" in rows and "dev" in rows:
        gap = rows["train"][0] - rows["dev"][0]
        print(f"\njackknife check: train recall@1 - dev recall@1 = {gap:+.2f} pts")
        if abs(gap) < 3:
            print("  Small. The base scorer has not memorised the train split, so")
            print("  its candidate lists there are representative and can be used")
            print("  to train the reranker directly -- no 5-fold decoding needed.")
        else:
            print("  LARGE. Train candidate lists are unrepresentatively clean;")
            print("  a reranker fitted on them will just learn to trust the base")
            print("  score. Use 5-fold jackknifed decoding, or train on dev only.")


if __name__ == "__main__":
    main()
