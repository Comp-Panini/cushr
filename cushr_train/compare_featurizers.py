#!/usr/bin/env python3
"""Compare trained models across featurizations and write a markdown report.

Two F1 numbers tell you which featurizer won but not why, which is the question
that decides what to build next. So this also buckets errors along the axis the
featurizations actually differ on -- how often a gold word was seen in training.
`scalars64` adds corpus frequency as an explicit feature, so if its advantage is
concentrated in the common-word buckets and absent in the unseen bucket, the win
is coming from frequency rather than from the position or character-class
blocks, and the next move is form identity (embedding tables). If the advantage
is flat across buckets, the extra columns are helping generally and capacity,
not features, is the limit.

    python compare_featurizers.py \
        --run morph43=cache_morph43:model_morph43.npz \
        --run scalars64=cache_scalars64:model_scalars64.npz \
        --out FEATURIZER_COMPARISON.md
"""

import argparse
import json
import os

import numpy as np
import torch

from dataset import LatticeStore, collate
from model import BiaffineEdgeScorer
from train import batches, to_torch
from viterbi import viterbi, predicted_nodes

# Gold words are bucketed by how many times their surface form occurs in the
# training split. "unseen" is the out-of-vocabulary case a frequency feature
# cannot help with; "common" is where it should help most.
BUCKETS = [("unseen", 0, 0), ("rare", 1, 4), ("mid", 5, 49), ("common", 50, None)]


def bucket_of(count):
    for name, lo, hi in BUCKETS:
        if count >= lo and (hi is None or count <= hi):
            return name
    return "common"


@torch.no_grad()
def evaluate_detailed(model, store, ids, dev, form_counts, node_form_id,
                      batch_size=128):
    """Overall P/R/F1 plus per-frequency-bucket recall on gold words."""
    model.eval()
    tp = fp = fn = pm = n = 0
    b_tp = {b[0]: 0 for b in BUCKETS}
    b_fn = {b[0]: 0 for b in BUCKETS}
    for chunk in batches(ids, batch_size, shuffle=False):
        b = collate(store, chunk)
        t = to_torch(b, dev)
        w = model(t["feats"], t["src"], t["dst"])
        pe, pmask, _ = viterbi(t, w)
        pred_local = predicted_nodes(t, pe, pmask)
        gn = np.asarray(b["global_node"])
        for i, s in enumerate(chunk):
            pred = {int(gn[x]) for x in pred_local[i]}
            gold = set(store.gold_nodes[store.gold_off[s]:store.gold_off[s + 1]].tolist())
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
            pm += int(pred == gold)
            n += 1
            # Bucket each gold word by its training frequency, then record
            # whether the decoder recovered it.
            for node in gold:
                cnt = int(form_counts[node_form_id[node]])
                key = bucket_of(cnt)
                if node in pred:
                    b_tp[key] += 1
                else:
                    b_fn[key] += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    model.train()
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "perfect_match": pm / n if n else 0.0, "n": n,
        "tp": tp, "fp": fp, "fn": fn,
        "bucket_recall": {
            k: (b_tp[k] / (b_tp[k] + b_fn[k]) if (b_tp[k] + b_fn[k]) else float("nan"))
            for k in b_tp},
        "bucket_support": {k: b_tp[k] + b_fn[k] for k in b_tp},
    }


def train_form_counts(raw_npz, cache, dev_pct=5, test_pct=5):
    """Training-split occurrence count per surface form, and each node's form id.

    Recomputed here rather than read back from the featurizer so the bucketing
    is independent of whichever featurizer is being scored -- both models get
    bucketed by the same yardstick.
    """
    from prepare import bucket
    z = np.load(raw_npz)
    form_ids = z["node_form_id"].astype(np.int64)
    sent_off = np.append(z["sentenceoffsets"], len(form_ids)).astype(np.int64)
    gold_off = z["gold_path_offsets"].astype(np.int64)
    has_gold = np.diff(gold_off) > 0
    mask = np.zeros(len(form_ids), dtype=bool)
    for s in range(len(sent_off) - 1):
        if has_gold[s] and bucket(s) >= dev_pct + test_pct:
            mask[sent_off[s]:sent_off[s + 1]] = True
    counts = np.bincount(form_ids[mask], minlength=int(form_ids.max()) + 1)
    return counts, form_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Delimiter is '=' throughout, not ':', because Windows paths carry a
    # drive-letter colon that would split in the wrong place.
    ap.add_argument("--run", action="append", required=True,
                    metavar="NAME=CACHE=MODEL",
                    help="repeatable; e.g. scalars64=./cache_s=model_s.npz")
    ap.add_argument("--raw", required=True,
                    help="raw ingest .npz (for frequency bucketing)")
    ap.add_argument("--out", default="FEATURIZER_COMPARISON.md")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    print("computing training-split form frequencies ...")
    form_counts, node_form_id = train_form_counts(args.raw, None)

    results = {}
    for spec in args.run:
        parts = spec.split("=")
        if len(parts) != 3:
            raise SystemExit(f"--run must be NAME=CACHE=MODEL, got {spec!r}")
        name, cache, model_path = parts
        print(f"\n=== {name} ===  cache={cache} model={model_path}")
        store = LatticeStore(cache)
        m = np.load(model_path)
        net = BiaffineEdgeScorer(int(m["feat_dim"]), int(m["hidden"])).to(dev)
        net.src_proj.weight.data = torch.as_tensor(m["src_proj"], device=dev)
        net.dst_proj.weight.data = torch.as_tensor(m["dst_proj"], device=dev)
        net.bias.data = torch.as_tensor(np.asarray(m["bias"]).reshape(1), device=dev)
        ids = store.trainable(args.split)
        r = evaluate_detailed(net, store, ids, dev, form_counts, node_form_id)
        r["featurizer"] = store.featurizer_name
        r["feat_dim"] = store.feat_dim
        r["params"] = net.num_params()
        log = os.path.join(os.path.dirname(model_path) or ".", "train_log.json")
        alt = model_path.replace("model_", "log_").replace(".npz", ".json")
        for cand in (alt, log):
            if os.path.exists(cand):
                with open(cand) as f:
                    r["log"] = json.load(f)
                break
        results[name] = r
        print(f"  F1 {r['f1']:.4f}  P {r['precision']:.4f}  R {r['recall']:.4f}  "
              f"PM {r['perfect_match']:.4f}  n={r['n']}")

    write_report(args.out, results, args.split)
    print(f"\nwrote {args.out}")


def write_report(path, results, split):
    names = list(results)
    L = []
    L.append("# Featurizer comparison\n")
    L.append(f"Word-level top-1 accuracy on the **{split}** split, decoded with "
             "Viterbi over the learned biaffine edge scores.\n")

    L.append("\n## Headline\n")
    L.append("| featurizer | F1 | precision | recall | perfect match | sentences | params |")
    L.append("|---|---|---|---|---|---|---|")
    for n in names:
        r = results[n]
        L.append(f"| `{n}` | **{r['f1']:.4f}** | {r['precision']:.4f} | "
                 f"{r['recall']:.4f} | {r['perfect_match']:.4f} | {r['n']:,} | "
                 f"{r['params']:,} |")

    L.append("\n## Recall by training frequency of the gold word\n")
    L.append("Each gold word is bucketed by how often its surface form occurs in "
             "the training split. This isolates whether an advantage comes from "
             "the frequency feature or from the rest of the vector.\n")
    header = "| featurizer | " + " | ".join(
        f"{b[0]} (n={results[names[0]]['bucket_support'][b[0]]:,})" for b in BUCKETS) + " |"
    L.append(header)
    L.append("|---" * (len(BUCKETS) + 1) + "|")
    for n in names:
        r = results[n]
        cells = " | ".join(f"{r['bucket_recall'][b[0]]:.4f}" for b in BUCKETS)
        L.append(f"| `{n}` | {cells} |")
    if len(names) == 2:
        a, b = names
        L.append(f"\nDelta (`{b}` - `{a}`):\n")
        L.append("| bucket | " + " | ".join(x[0] for x in BUCKETS) + " |")
        L.append("|---" * (len(BUCKETS) + 1) + "|")
        d = " | ".join(
            f"{results[b]['bucket_recall'][x[0]] - results[a]['bucket_recall'][x[0]]:+.4f}"
            for x in BUCKETS)
        L.append(f"| recall | {d} |")

    L.append("\n## Error counts\n")
    L.append("| featurizer | TP | FP | FN |")
    L.append("|---|---|---|---|")
    for n in names:
        r = results[n]
        L.append(f"| `{n}` | {r['tp']:,} | {r['fp']:,} | {r['fn']:,} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
