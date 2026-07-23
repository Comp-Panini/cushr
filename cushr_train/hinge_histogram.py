#!/usr/bin/env python3
"""hinge_histogram.py -- distribution of per-sentence hinge loss.

train.py logs only the mean loss and the fraction of sentences with a non-zero
hinge (90.6% at the end of training). That single number hides the shape: a
corpus where every sentence misses by a hair is a very different problem from
one where a tenth of them miss catastrophically. This recomputes the per-
sentence hinge with the trained model and plots how it breaks down.

Sentences with hinge exactly 0 are the ones the model already gets right by
more than the margin; they are reported as a number rather than drawn, because
a spike at 0 would swamp every other bar.

Usage:
    python hinge_histogram.py --model model_biaffine.npz --split train
"""

import argparse
import json

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import LatticeStore, collate, FEAT_DIM
from model import BiaffineEdgeScorer
from train import to_torch, batches
from viterbi import viterbi, path_score, gold_score


@torch.no_grad()
def hinge_values(model, store, ids, dev, margin, batch_size=128):
    out = []
    for chunk in batches(ids, batch_size, shuffle=False):
        b = collate(store, chunk)
        t = to_torch(b, dev)
        w = model(t["feats"], t["src"], t["dst"])
        cost = margin * (~t["gold_node"][t["dst"]]).to(w.dtype)
        pe, pmask, _ = viterbi(t, w + cost)
        s_pred = path_score(w, pe, pmask)
        hamming = (cost[pe] * pmask.to(w.dtype)).sum(-1)
        s_gold = gold_score(w, t["gold_edge"], t["gold_edge_ptr"], len(chunk))
        out.append(torch.relu(hamming + s_pred - s_gold).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--model", default="model_biaffine.npz")
    ap.add_argument("--split", default="train")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--png", default="hinge_histogram.png")
    ap.add_argument("--stats", default="hinge_stats.json")
    args = ap.parse_args()

    dev = torch.device("cpu")
    store = LatticeStore(args.cache)
    ids = store.trainable(args.split)

    m = np.load(args.model)
    model = BiaffineEdgeScorer(int(m["feat_dim"]), int(m["hidden"])).to(dev)
    with torch.no_grad():
        model.src_proj.weight.copy_(torch.as_tensor(m["src_proj"]))
        model.dst_proj.weight.copy_(torch.as_tensor(m["dst_proj"]))
        model.bias.copy_(torch.as_tensor(m["bias"]).reshape(1))
    model.eval()

    h = hinge_values(model, store, ids, dev, args.margin)
    n = len(h)
    zero = h == 0.0
    nz = h[~zero]
    pct = lambda x: 100.0 * x / n

    stats = {
        "split": args.split, "n_sentences": int(n),
        "pct_zero": pct(int(zero.sum())),
        "pct_nonzero": pct(int((~zero).sum())),
        "mean_all": float(h.mean()), "mean_nonzero": float(nz.mean()),
        "median_nonzero": float(np.median(nz)), "max": float(h.max()),
    }
    for p in (25, 50, 75, 90, 95, 99):
        stats[f"p{p}_nonzero"] = float(np.percentile(nz, p))
    # How much of the total loss comes from the worst decile?
    order = np.sort(h)[::-1]
    stats["pct_of_loss_from_worst_10pct"] = float(
        100.0 * order[: max(1, n // 10)].sum() / order.sum())

    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)

    # ---- plot ----------------------------------------------------------
    # y is % of ALL sentences, so the drawn bars sum to pct_nonzero and the
    # reader can compare against the zero bucket quoted in the caption.
    plt.figure(figsize=(5.6, 3.6))
    hi = float(np.percentile(nz, 99.5))
    counts, edges = np.histogram(nz, bins=args.bins, range=(0.0, hi))
    widths = np.diff(edges)
    plt.bar(edges[:-1], 100.0 * counts / n, width=widths, align="edge",
            color="#2b6cb0", edgecolor="white", linewidth=0.5)
    plt.xlabel("Per-sentence hinge loss")
    plt.ylabel("% of all sentences")
    plt.title("Hinge loss distribution (non-zero sentences)")
    plt.xlim(0, hi)
    plt.ylim(0, None)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(args.png, dpi=140)
    plt.close()

    print(f"wrote {args.png} and {args.stats}")
    print(f"  {n} sentences ({args.split})")
    print(f"  zero hinge     : {stats['pct_zero']:.1f}%")
    print(f"  non-zero hinge : {stats['pct_nonzero']:.1f}%")
    print(f"  non-zero median{stats['median_nonzero']:>8.3f}   "
          f"mean {stats['mean_nonzero']:.3f}   max {stats['max']:.3f}")
    for p in (25, 50, 75, 90, 95, 99):
        print(f"  p{p:<3} (non-zero): {stats[f'p{p}_nonzero']:.3f}")
    print(f"  worst 10% of sentences carry "
          f"{stats['pct_of_loss_from_worst_10pct']:.1f}% of total loss")


if __name__ == "__main__":
    main()
