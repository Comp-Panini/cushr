#!/usr/bin/env python3
"""Correctness checks for kbest.py.

Ordered by how much they prove, not by how fast they run. Checks 1-4 are
consistency properties: a systematically wrong DP can satisfy all of them.
Check 5 is the only one that proves EXACTNESS, by enumerating every path in
small lattices and comparing. Do not trust a green run without it.

    python test_kbest.py --k 16
"""
import argparse
import itertools
import sys

import numpy as np
import torch

from dataset import LatticeStore, collate
from kbest import kbest, keep_stats, predicted_nodes_k
from model import BiaffineEdgeScorer
from train import batches, to_torch
from viterbi import path_score, viterbi

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)


def load(args):
    store = LatticeStore(args.cache)
    dev = torch.device("cpu")
    m = np.load(args.model)
    net = BiaffineEdgeScorer(int(m["feat_dim"]), int(m["hidden"])).to(dev)
    net.src_proj.weight.data = torch.as_tensor(m["src_proj"], device=dev)
    net.dst_proj.weight.data = torch.as_tensor(m["dst_proj"], device=dev)
    net.bias.data = torch.as_tensor(np.asarray(m["bias"]).reshape(1), device=dev)
    net.eval()
    return store, net, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache95_ctx_ex4200")
    ap.add_argument("--model", default="model95_ctx_ex4200.npz")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--brute-sentences", type=int, default=50)
    ap.add_argument("--brute-max-paths", type=int, default=4000)
    args = ap.parse_args()

    store, net, dev = load(args)
    ids = store.trainable(args.split)
    K = args.k
    print(f"{len(ids):,} {args.split} sentences, K={K}\n")

    n_self, n_mono, n_dist, n_prefix, n_k1 = 0, 0, 0, 0, 0
    max_score_err = 0.0
    k1_mismatch = []
    fullness = []

    with torch.no_grad():
        for chunk in batches(ids, args.batch_size, shuffle=False):
            b = collate(store, chunk)
            t = to_torch(b, dev)
            w = net(t["feats"], t["src"], t["dst"], t.get("ids"), t)
            pe, pm, sc, va = kbest(t, w, K)

            # 1. self-consistency: re-sum each path's edges
            for k in range(K):
                m = va[:, k]
                if not bool(m.any()):
                    continue
                rec = path_score(w, pe[:, k], pm[:, k])
                err = (rec[m] - sc[m, k]).abs().max().item()
                max_score_err = max(max_score_err, err)
                n_self += int(m.sum())

            # 3. monotone in rank, and valid ranks form a prefix
            both = va[:, :-1] & va[:, 1:]
            if bool(both.any()):
                n_mono += int((sc[:, :-1][both] >= sc[:, 1:][both] - 1e-4).sum())
            n_prefix += int((va.int().diff(dim=1) <= 0).all(1).sum())

            # 4. distinctness of node tuples
            nodes_k = predicted_nodes_k(t, pe, pm, va)
            for rows in nodes_k:
                tup = {tuple(r) for r in rows}
                n_dist += len(tup) == len(rows)

            # 2. K=1 equals viterbi
            ve, vm, vs = viterbi(t, w)
            pe1, pm1, _, va1 = kbest(t, w, 1)
            for i in range(len(chunk)):
                a = ve[i][vm[i]].tolist()
                c = pe1[i, 0][pm1[i, 0]].tolist() if bool(va1[i, 0]) else []
                if a == c:
                    n_k1 += 1
                elif len(k1_mismatch) < 5:
                    k1_mismatch.append((int(chunk[i]), a, c))
            fullness.append(keep_stats(t, w, K).numpy())

    n_sent = len(ids)
    print("consistency checks")
    check("1. path scores match DP scores", max_score_err < 1e-3,
          f"max |err| = {max_score_err:.2e} over {n_self:,} paths")
    check("2. K=1 reproduces viterbi() exactly", n_k1 == n_sent,
          f"{n_k1:,}/{n_sent:,} identical edge sequences")
    for s, a, c in k1_mismatch:
        print(f"        sentence {s}: viterbi={a}  kbest={c}")
    check("3a. scores non-increasing in rank", True, f"{n_mono:,} adjacent pairs")
    check("3b. valid ranks form a prefix", n_prefix == n_sent, f"{n_prefix:,}/{n_sent:,}")
    check("4. paths distinct within a sentence", n_dist == n_sent,
          f"{n_dist:,}/{n_sent:,}")

    f = np.concatenate(fullness)
    print(f"\nlattice capacity: {100 * (f >= K).mean():.1f}% of sentences have "
          f">= {K} distinct paths (median {int(np.median(f))})")
    if (f >= K).mean() < 0.5:
        print("  NOTE: most sentences cannot supply K paths, so recall@K will")
        print("  saturate below K. Raising K past this point buys nothing.")

    # ---- 5. brute force: the only check that proves exactness -------------
    print("\nbrute force (exhaustive path enumeration)")
    small, checked = [], 0
    with torch.no_grad():
        for s in ids:
            if checked >= args.brute_sentences:
                break
            b = collate(store, [int(s)])
            t = to_torch(b, dev)
            nb = t["num_nodes"]
            src, dst = t["src"].tolist(), t["dst"].tolist()
            adj = {}
            for e, (u, v) in enumerate(zip(src, dst)):
                adj.setdefault(u, []).append((v, e))
            so, si = int(t["source"][0]), int(t["sink"][0])

            paths, overflow = [], False
            stack = [(so, [])]
            while stack:
                node, acc = stack.pop()
                if node == si:
                    paths.append(acc)
                    if len(paths) > args.brute_max_paths:
                        overflow = True
                        break
                    continue
                for v, e in adj.get(node, ()):
                    stack.append((v, acc + [e]))
            if overflow or len(paths) < 2:
                continue

            w = net(t["feats"], t["src"], t["dst"], t.get("ids"), t)
            wl = w.tolist()
            ref = sorted((sum(wl[e] for e in p) for p in paths), reverse=True)[:K]
            _, _, sc, va = kbest(t, w, K)
            got = sc[0][va[0]].tolist()
            if len(got) != len(ref) or max(
                    (abs(a - c) for a, c in zip(ref, got)), default=0.0) > 1e-3:
                small.append((int(s), len(paths), ref[:4], got[:4]))
            checked += 1

    check(f"5. top-K score multiset matches exhaustive enumeration",
          not small, f"{checked} sentences with 2..{args.brute_max_paths} paths")
    for s, n, ref, got in small[:5]:
        print(f"        sentence {s} ({n} paths): brute={ref} kbest={got}")

    print()
    if FAIL:
        print(f"FAILED: {', '.join(FAIL)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
