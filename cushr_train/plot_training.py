#!/usr/bin/env python3
"""plot_training.py -- training-loss curve for WEEK9_TRAINING.md.

Reads train_log.json (written by train.py) and plots mean hinge loss per epoch.
Both axes are anchored at 0, matching the convention in cushr_gpu's plots.

Usage:  python plot_training.py [--log train_log.json] [--png training_loss.png]
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="train_log.json")
    ap.add_argument("--png", default="training_loss.png")
    args = ap.parse_args()

    with open(args.log) as f:
        blob = json.load(f)
    hist = blob["history"]
    xs = [h["epoch"] for h in hist]
    ys = [h["train_loss"] for h in hist]

    plt.figure(figsize=(5.2, 3.4))
    # One series, so no legend box -- the title names it (a legend for a single
    # line is pure chrome).
    plt.plot(xs, ys, "o-", color="#2b6cb0", linewidth=2, markersize=6)
    plt.xlabel("Epoch")
    plt.ylabel("Mean hinge loss (train)")
    plt.title("Training loss vs epoch")
    # Anchored at 0 on both axes. The curve reads as nearly flat at this scale,
    # which is the honest picture: the loss falls 3.66 -> 3.24 and then stalls.
    plt.xlim(0, max(xs) + 0.5)
    plt.ylim(0, max(ys) * 1.15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.png, dpi=140)
    plt.close()

    print(f"wrote {args.png}")
    print(f"  epochs {min(xs)}-{max(xs)}  loss {ys[0]:.4f} -> {ys[-1]:.4f} "
          f"({100.0 * (ys[0] - ys[-1]) / ys[0]:.1f}% drop)")


if __name__ == "__main__":
    main()
