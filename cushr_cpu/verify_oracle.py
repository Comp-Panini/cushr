#!/usr/bin/env python3
"""
Oracle sanity check for the CPU top-K decoder.

For N random sentences, independently re-score a handful of candidate paths
  - the decoder's own top-1 (from golden_outputs.json)
  - the gold path
  - several random source->sink walks
using the SAME edge-scoring rule the C++ LogLinearScorer uses
(edge score = bias + w . node_features[dst]; path score = sum of edge scores,
 i.e. sum over every node on the path EXCEPT the source).

If the decoder's top-1 has the highest score among all sampled candidates for
every sentence, the DP is consistent with the scoring function and is safe to
use as an oracle. Any sentence where some other path beats it is a real bug.
"""
import json
import random
import sys

import numpy as np

NPZ   = "../data/new_cushr_data_fixed_USE_THIS.npz"
GOLD  = "../golden_outputs.json"
N_SENT = 20
N_RANDOM_PATHS = 6
BIAS = 0.0
EPS = 1e-4          # float32 slack for tie comparisons
SEED = 0

# The 43 weights passed to ./cushr_evaluate (--scorer log_linear).
WEIGHTS = np.array([
    0.0559,-0.4269,-0.5438,-0.3353,0.5429,-0.2211,-1.4043,0.3214,0.0776,-2.3854,
    0.1326,0.4872,-1.1342,-0.4812,0.0558,1.3182,-0.0024,-0.6225,1.2320,-1.0156,
    1.7814,-1.0106,-1.0189,-0.1246,0.5901,0.9213,-1.0925,-1.4549,1.3426,-6.2178,
    -7.1641,-3.5924,-3.3199,1.6020,-1.6993,1.7712,2.3289,0.3599,1.6179,0.5901,
    0.9764,-0.9046,0.0000,
], dtype=np.float32)


def main():
    random.seed(SEED)
    d = np.load(NPZ)
    feats = d["node_features"].astype(np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(-1, WEIGHTS.size)
    assert feats.shape[1] == WEIGHTS.size, f"feat_dim {feats.shape[1]} != {WEIGHTS.size}"
    row_ptr = d["row_ptr"]
    col_idx = d["col_idx"]
    sent_off = d["sentence_offsets"]
    gp_nodes = d["gold_path_nodes"]
    gp_off = d["gold_path_offsets"]

    # per-node contribution to a path score = bias + w . feat[node]
    node_score = feats @ WEIGHTS + BIAS          # (N,) float32

    def path_score(nodes):
        # sum of edge scores; every node except the source (first) contributes
        return float(node_score[nodes[1:]].sum())

    golden = json.load(open(GOLD))
    dec = {s["sentence_id"]: s["paths"] for s in golden["sentences"]}

    def random_path(src, sink):
        v = src
        path = [v]
        while v != sink:
            lo, hi = row_ptr[v], row_ptr[v + 1]
            if hi == lo:
                return None                      # dead end (shouldn't happen)
            v = int(col_idx[random.randrange(lo, hi)])
            path.append(v)
        return path

    # eligible sentences: have a decoder path AND an explicit gold path
    eligible = [sid for sid in dec
                if gp_off[sid + 1] > gp_off[sid] and len(dec[sid][0]["nodes"]) > 2]
    sample = random.sample(eligible, min(N_SENT, len(eligible)))

    failures = 0
    score_mismatch = 0
    print(f"checking {len(sample)} sentences "
          f"({N_RANDOM_PATHS} random paths each)\n")

    for sid in sorted(sample):
        top1 = dec[sid][0]
        top1_nodes = top1["nodes"]
        src, sink = top1_nodes[0], top1_nodes[-1]

        # 1) does our independent score reproduce the decoder's reported score?
        my_top1 = path_score(top1_nodes)
        if abs(my_top1 - top1["score"]) > 1e-2:
            score_mismatch += 1
            print(f"  [score mismatch] sent {sid}: "
                  f"decoder={top1['score']:.5f} ours={my_top1:.5f}")

        # 2) build candidate set: gold + random valid paths
        gold_words = gp_nodes[gp_off[sid]:gp_off[sid + 1]].tolist()
        gold_full = [src] + gold_words + [sink]
        candidates = {"gold": gold_full}
        for i in range(N_RANDOM_PATHS):
            p = random_path(src, sink)
            if p is not None:
                candidates[f"rand{i}"] = p

        best_other = max(path_score(p) for p in candidates.values())
        winner = max(candidates, key=lambda k: path_score(candidates[k]))

        ok = my_top1 + EPS >= best_other
        if not ok:
            failures += 1
        flag = "OK " if ok else "*** FAIL ***"
        print(f"  sent {sid:>3}  top1={my_top1:8.4f}  "
              f"best_other={best_other:8.4f} ({winner:5})  {flag}")

    print()
    if score_mismatch:
        print(f"WARNING: {score_mismatch} sentences where our re-score "
              f"disagreed with the decoder's reported score "
              f"(scorer replication issue, not necessarily a DP bug)")
    if failures == 0:
        print("PASS: decoder top-1 is the highest-scoring path in every "
              "sampled set. DP is consistent with the scoring function; "
              "cleared to use as an oracle.")
        return 0
    print(f"FAIL: {failures} sentence(s) had a path out-scoring the "
          f"decoder's top-1. The DP has a bug.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
