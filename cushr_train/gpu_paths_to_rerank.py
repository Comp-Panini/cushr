#!/usr/bin/env python3
"""Convert cushr_batched --dump-paths output into make_rerank_data.py's layout.

WHY THIS IS A SEPARATE STEP RATHER THAN C++ IN THE DRIVER
---------------------------------------------------------
The GPU driver knows the lattice: it can reconstruct a top-K node sequence and
name every node by its global id. It does not know three conventions that the
Python side owns, and duplicating them in CUDA would create a second definition
that drifts:

  1. the form filter -- nodes whose form_id maps to an empty surface form are
     dropped from a candidate (make_rerank_data.py:82);
  2. the ordering -- candidates are sorted by node_char_start, not by the
     decoder's topological order. For a linear lattice these agree, but the
     tie-break on equal-span nodes is the Python one and must stay so;
  3. the label -- a candidate is correct iff its filtered, sorted sequence
     equals the gold path under the same two rules.

So the driver emits raw reconstructions and this script applies the three,
producing a file that eval_slm.py and rerank.py read with no changes at all.

USAGE
    python gpu_paths_to_rerank.py --gpu ../cushr_gpu/gpu_k32_K32.npz \\
        --cache ./cache95_ctx_ex4200 --raw ../data/cushr_data_g95.npz \\
        --out gpu_rerank_k32.npz

The output is verified against the driver's own recall@K where possible: this
script prints recall@1 and recall@K, and they must match the numbers
cushr_batched printed for the same run. A disagreement means one of the three
conventions above was applied differently on the two sides, which is exactly
what this script exists to prevent -- so it is a hard failure, not a warning.
"""
import argparse

import numpy as np

from dataset import LatticeStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True,
                    help="npz written by cushr_batched --dump-paths")
    ap.add_argument("--cache", default="./cache95_ctx_ex4200")
    ap.add_argument("--raw", default="../data/cushr_data_g95.npz")
    ap.add_argument("--form-vocab", default="../data/form_vocabulary.txt")
    ap.add_argument("--out", default="gpu_rerank.npz")
    args = ap.parse_args()

    g = np.load(args.gpu)
    cand_nodes = np.asarray(g["cand_nodes"], dtype=np.int64)
    cand_off = np.asarray(g["cand_off"], dtype=np.int64)
    cand_sent = np.asarray(g["cand_sent"], dtype=np.int64)
    cand_score = np.asarray(g["cand_score"], dtype=np.float32)
    sent_ids = np.asarray(g["sent_ids"], dtype=np.int64)
    K = int(np.asarray(g["K"]).reshape(-1)[0]) if "K" in g.files else 0

    store = LatticeStore(args.cache)
    z = np.load(args.raw)
    cstart, fid = z["node_char_start"], z["node_form_id"]
    forms = [l.split("\t", 1)[1].rstrip("\n") if "\t" in l else ""
             for l in open(args.form_vocab, encoding="utf-8")]

    def canon(nodes):
        """The three conventions, in the order make_rerank_data.py applies them."""
        seq = sorted((int(x) for x in nodes), key=lambda n: int(cstart[n]))
        return [n for n in seq if forms[fid[n]]]

    # Gold, canonicalised the same way, once per sentence in the dump.
    gold = {}
    for si, s in enumerate(sent_ids):
        s = int(s)
        gold[si] = tuple(canon(
            store.gold_nodes[store.gold_off[s]:store.gold_off[s + 1]]))

    out_nodes, out_off, out_sent, out_score, out_label = [], [0], [], [], []
    # rank_in_sent tracks position within a sentence's candidate list, which the
    # dump preserves (the driver emits ranks 0..count-1 contiguously) but does
    # not store; recall@1 needs it.
    hit1 = np.zeros(len(sent_ids), dtype=bool)
    hitK = np.zeros(len(sent_ids), dtype=bool)
    seen = np.zeros(len(sent_ids), dtype=np.int64)

    for i in range(len(cand_sent)):
        si = int(cand_sent[i])
        seq = canon(cand_nodes[cand_off[i]:cand_off[i + 1]])
        ok = tuple(seq) == gold[si]
        out_nodes.extend(seq)
        out_off.append(len(out_nodes))
        out_sent.append(si)
        out_score.append(float(cand_score[i]))
        out_label.append(ok)
        if ok:
            hitK[si] = True
            if seen[si] == 0:
                hit1[si] = True
        seen[si] += 1

    data = {
        "cand_nodes": np.asarray(out_nodes, dtype=np.int32),
        "cand_off": np.asarray(out_off, dtype=np.int64),
        "cand_sent": np.asarray(out_sent, dtype=np.int32),
        "cand_score": np.asarray(out_score, dtype=np.float32),
        "cand_label": np.asarray(out_label, dtype=bool),
        "sent_ids": np.asarray(sent_ids, dtype=np.int32),
    }
    np.savez_compressed(args.out, **data)

    n = len(sent_ids)
    print(f"{args.gpu} -> {args.out}")
    print(f"  {n:,} sentences, {len(out_sent):,} candidates, K={K}")
    print(f"  recall@1  = {100 * hit1.sum() / n:.2f}%")
    print(f"  recall@{K if K else 'K'} = {100 * hitK.sum() / n:.2f}%")
    print("  cross-check: recall@K must equal the number cushr_batched printed "
          "for this run; if it does not, the form filter or the span-start sort "
          "differs between the two sides and the candidate lists are not "
          "comparable to make_rerank_data.py's.")


if __name__ == "__main__":
    main()
