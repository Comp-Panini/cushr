#!/usr/bin/env python3

# goal: compare F1 against other scoring methods
# uniform = .49, length = .48, log linear = .57, biaffine = .79

import argparse
import json

import numpy as np
import torch

from dataset import LatticeStore, collate
from train import evaluate as eval_with_weights  
from train import f1_counts, to_torch, batches
from viterbi import viterbi, predicted_nodes


def fit_log_linear(store, train_ids):
    nodes = np.concatenate([
        np.arange(store.sent_off[s], store.sent_off[s + 1], dtype=np.int64)
        for s in train_ids])
    gold = np.zeros(len(nodes), dtype=bool)
    pos = {int(n): i for i, n in enumerate(nodes)}
    for s in train_ids:
        for g in store.gold_nodes[store.gold_off[s]:store.gold_off[s + 1]]:
            gold[pos[int(g)]] = True
    feats = np.asarray(store.node_features[nodes], dtype=np.float32)
    eps = 1e-6
    w = np.log((feats[gold].mean(0) + eps) / (feats[~gold].mean(0) + eps))
    return w.astype(np.float32)


def eval_scorer(store, ids, edge_w_fn, batch_size=128):
    tp = fp = fn = pm = n = 0
    for chunk in batches(ids, batch_size, shuffle=False):
        b = collate(store, chunk)
        t = to_torch(b, torch.device("cpu"))
        w = torch.as_tensor(edge_w_fn(store, b), dtype=torch.float32)
        pe, pmask, _ = viterbi(t, w)
        gn = np.asarray(b["global_node"])
        pred = [[int(gn[x]) for x in p]
                for p in predicted_nodes(t, pe, pmask)]
        gold = [store.gold_nodes[store.gold_off[s]:store.gold_off[s + 1]].tolist()
                for s in chunk]
        a, c, d, e = f1_counts(pred, gold)
        tp += a; fp += c; fn += d; pm += e; n += len(chunk)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "perfect_match": pm / n if n else 0.0, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--out", default="baseline_results.json")
    args = ap.parse_args()

    store = LatticeStore(args.cache)
    train_ids = store.trainable("train")
    test_ids = store.trainable("test")

    w_ll = fit_log_linear(store, train_ids)
    print("log_linear weights (train split):")
    print("  " + ",".join(f"{x:.4f}" for x in w_ll))

    def uniform(_s, b):
        return np.ones(b["num_edges"], dtype=np.float32)

    def length(s, b):
        dst = np.asarray(b["global_node"])[b["dst"]]
        return np.log1p(np.asarray(s.word_len[dst], dtype=np.float32))

    def log_linear(s, b):
        dst = np.asarray(b["global_node"])[b["dst"]]
        return np.asarray(s.node_features[dst], dtype=np.float32) @ w_ll

    out = {}
    for name, fn in (("uniform", uniform), ("length", length),
                     ("log_linear", log_linear)):
        r = eval_scorer(store, test_ids, fn)
        out[name] = r
        print(f"{name:<12} test F1 {r['f1']:.4f}  P {r['precision']:.4f}  "
              f"R {r['recall']:.4f}  PM {r['perfect_match']:.4f}")

    out["log_linear_weights"] = [float(x) for x in w_ll]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
