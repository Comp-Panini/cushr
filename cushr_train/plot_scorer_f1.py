#!/usr/bin/env python3
"""plot_scorer_f1.py -- F1 across scorers on the held-out test split.

Reads baseline_results.json (the three week-3 scorers, from baseline.py) and
the biaffine test row from train_log.json, and draws a grouped bar chart of
F1 / precision / recall. All on the SAME test split, so the bars are directly
comparable -- which is the whole point of running baseline.py on the week-9
partition rather than quoting the README's whole-corpus number.

Usage:
    python plot_scorer_f1.py            # writes scorer_f1.png
"""

import argparse
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One hue per metric, assigned in fixed order (identity, not magnitude).
# Blue matches the other cushr plots; the three are CVD-separable.
METRIC_COLORS = {"F1": "#2b6cb0", "Precision": "#dd6b20", "Recall": "#38a169"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="baseline_results.json")
    ap.add_argument("--train-log", default="train_log.json")
    ap.add_argument("--png", default="scorer_f1.png")
    args = ap.parse_args()

    base = json.load(open(args.baseline))
    biaffine = json.load(open(args.train_log))["test"]

    # ordered worst -> best so the trained model reads as the endpoint
    scorers = [
        ("uniform", base["uniform"]),
        ("length", base["length"]),
        ("log_linear", base["log_linear"]),
        ("biaffine", biaffine),
    ]
    labels = [name for name, _ in scorers]
    metrics = ["F1", "Precision", "Recall"]
    keys = {"F1": "f1", "Precision": "precision", "Recall": "recall"}

    x = np.arange(len(scorers))
    width = 0.26

    plt.figure(figsize=(6.4, 3.8))
    for j, metric in enumerate(metrics):
        vals = [s[keys[metric]] for _, s in scorers]
        bars = plt.bar(x + (j - 1) * width, vals, width,
                       label=metric, color=METRIC_COLORS[metric],
                       edgecolor="white", linewidth=0.5)
        # direct-label F1 only -- it's the headline; labeling all three clutters.
        if metric == "F1":
            for xi, v in zip(x + (j - 1) * width, vals):
                plt.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                         fontsize=8, color="#2d3748")

    plt.xticks(x, labels)
    plt.ylabel("Score (test split)")
    plt.title("Segmentation accuracy by scorer (held-out test)")
    # Anchored at 0 so bar heights read as true magnitudes, matching the other
    # cushr plots; headroom for the F1 labels.
    plt.ylim(0, 1.0)
    plt.legend(frameon=False, ncol=3, loc="upper left")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(args.png, dpi=140)
    plt.close()

    print(f"wrote {args.png}")
    for name, s in scorers:
        print(f"  {name:<12} F1 {s['f1']:.4f}  P {s['precision']:.4f}  "
              f"R {s['recall']:.4f}")


if __name__ == "__main__":
    main()
