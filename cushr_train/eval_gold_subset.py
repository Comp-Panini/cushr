#!/usr/bin/env python3
"""Split a test set into originally-resolvable vs newly-recovered sentences.

The 49%-gold and 75%-gold corpora do not share a test split. Because the repair
only ever ADDS resolved sentences, the old test set is a strict subset of the
new one, so a headline F1 drop between the two reports can mean either

  (a) the model got worse, or
  (b) the test set got harder,

and the two are not distinguishable from the headline alone. This script
evaluates one 75%-trained model on the two halves of its own test split:

  old  -- sentences that already had a gold path before the orphan repair
  new  -- sentences the repair recovered

If `old` matches the 49% report's number, the model is unchanged and the drop is
entirely test-set composition.

Usage:
  python eval_gold_subset.py --cache ./cache75_hybrid_tag \
      --model model75_hybrid_tag.npz --old-npz ../data/new_cushr_data_fixed_USE_THIS.npz
"""
import argparse
import json
import os

import numpy as np
import torch

from compare_featurizers import evaluate_detailed, train_form_counts
from dataset import LatticeStore
from model import BiaffineEdgeScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw", default="../data/cushr_data_repaired.npz")
    ap.add_argument("--old-npz", default="../data/new_cushr_data_fixed_USE_THIS.npz")
    ap.add_argument("--split", default="test")
    ap.add_argument("--json-out", default="",
                    help="also append the three result rows to this JSON file, "
                         "keyed by --name")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    store = LatticeStore(args.cache)
    # store.trainable() is what the comparison report evaluates on; using the
    # raw splits.json list instead would include sentences the loader drops.
    ids = list(store.trainable(args.split))

    # Which sentences carried a gold path BEFORE the repair?
    old = np.load(args.old_npz)
    had_gold = np.diff(np.asarray(old["gold_path_offsets"], dtype=np.int64)) > 0

    old_ids = [s for s in ids if s < len(had_gold) and had_gold[s]]
    new_ids = [s for s in ids if not (s < len(had_gold) and had_gold[s])]

    # Load exactly the way compare_featurizers.py:139-143 does, so this script
    # and the comparison report cannot silently disagree about the weights.
    dev = torch.device("cpu")
    m = np.load(args.model)
    model = BiaffineEdgeScorer(int(m["feat_dim"]), int(m["hidden"])).to(dev)
    model.src_proj.weight.data = torch.as_tensor(m["src_proj"], device=dev)
    model.dst_proj.weight.data = torch.as_tensor(m["dst_proj"], device=dev)
    model.bias.data = torch.as_tensor(np.asarray(m["bias"]).reshape(1), device=dev)
    model.eval()

    form_counts, node_form_id = train_form_counts(args.raw, args.cache)

    print(f"{args.split}: {len(ids):,} sentences "
          f"= {len(old_ids):,} pre-repair + {len(new_ids):,} recovered\n")
    out = {}
    for name, subset in (("all", ids), ("old", old_ids), ("new", new_ids)):
        if not subset:
            continue
        r = evaluate_detailed(model, store, subset, dev, form_counts, node_form_id)
        out[name] = {k: r[k] for k in
                     ("f1", "precision", "recall", "perfect_match", "n")}
        print(f"  {name:<20} F1 {r['f1']:.4f}  P {r['precision']:.4f}  "
              f"R {r['recall']:.4f}  PM {r['perfect_match']:.4f}  n={r['n']:,}")

    if args.json_out:
        blob = {}
        if os.path.exists(args.json_out):
            blob = json.load(open(args.json_out))
        blob[args.name or args.model] = out
        json.dump(blob, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
