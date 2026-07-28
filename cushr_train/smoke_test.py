#!/usr/bin/env python3

# purpose: check to see that the data pipeline is correct

import argparse
import itertools

import numpy as np

from dataset import LatticeStore, collate

NEG_INF = -1e30


def np_viterbi(b, w):
    nb = b["num_nodes"]
    best = np.full(nb, NEG_INF, dtype=np.float64)
    best[b["source"]] = 0.0
    lp = b["level_ptr"]
    for l in range(len(lp) - 1):
        lo, hi = int(lp[l]), int(lp[l + 1])
        if hi <= lo:
            continue
        cand = best[b["src"][lo:hi]] + w[lo:hi]
        np.maximum.at(best, b["dst"][lo:hi], cand)
    return best


def brute_force(b, w, s):
    src, dst = b["src"], b["dst"]
    out = {}
    for u, v, e in zip(src, dst, itertools.count()):
        out.setdefault(int(u), []).append((int(v), e))
    sink = int(b["sink"][s])
    best = {}

    def rec(u, depth):
        if u == sink:
            return 0.0
        if u in best:
            return best[u]
        r = NEG_INF
        for v, e in out.get(u, []):
            sub = rec(v, depth + 1)
            if sub > NEG_INF / 2:
                r = max(r, w[e] + sub)
        best[u] = r
        return r

    return rec(int(b["source"][s]), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    store = LatticeStore(args.cache)
    print(f"store: {store.meta['num_sentences']} sentences, "
          f"{store.meta['num_with_gold']} with gold, "
          f"feat_dim={store.feat_dim} ({store.featurizer_name})")

    train = store.trainable("train")
    print(f"trainable(train) = {len(train)} "
          f"(of {len(store.splits['train'])} in split)")
    assert len(train) > 0

    rng = np.random.default_rng(7)
    col = np.asarray(store.col_idx)

    for bi in range(args.batches):
        ids = rng.choice(train, size=args.batch, replace=False)
        b = collate(store, ids)
        gn = b["global_node"]

        # 1. counts
        spans = store.sent_off[ids + 1] - store.sent_off[ids]
        assert b["num_nodes"] == int(spans.sum()), "node count mismatch"
        exp_e = sum(int(store.row_ptr[e] - store.row_ptr[s])
                    for s, e in zip(store.sent_off[ids], store.sent_off[ids + 1]))
        assert b["num_edges"] == exp_e, "edge count mismatch"

        # 2. edges map back to real corpus edges
        gsrc, gdst = gn[b["src"]], gn[b["dst"]]
        for k in rng.choice(b["num_edges"], size=min(400, b["num_edges"]),
                            replace=False):
            u, v = int(gsrc[k]), int(gdst[k])
            lo, hi = int(store.row_ptr[u]), int(store.row_ptr[u + 1])
            assert v in col[lo:hi], f"edge {u}->{v} not in corpus CSR"

        # 3. level ordering
        dl = store.topo_level[gdst]
        assert np.all(np.diff(dl) >= 0), "edges not sorted by dst level"
        lp = b["level_ptr"]
        for l in range(len(lp) - 1):
            lo, hi = int(lp[l]), int(lp[l + 1])
            if hi > lo:
                assert np.all(dl[lo:hi] == l), f"level_ptr slice {l} impure"
        # a dst always sits one level above at least one of its srcs
        assert np.all(store.topo_level[gsrc] < dl), "edge violates topo order"

        # 4. gold chains connected
        gp = b["gold_edge_ptr"]
        for i in range(len(ids)):
            ge = b["gold_edge"][gp[i]:gp[i + 1]]
            assert len(ge) >= 2, "gold chain too short"
            chain = list(zip(b["src"][ge], b["dst"][ge]))
            assert chain[0][0] == b["source"][i], "gold does not start at source"
            assert chain[-1][1] == b["sink"][i], "gold does not end at sink"
            for (_, v0), (u1, _) in zip(chain[:-1], chain[1:]):
                assert v0 == u1, "gold chain is broken"
            assert b["gold_node"][b["dst"][ge]].all(), "gold node mask disagrees"

        # 5. sweep vs brute force on the smallest sentences in the batch
        w = rng.normal(size=b["num_edges"]).astype(np.float64)
        best = np_viterbi(b, w)
        order = np.argsort(spans)[:3]
        for s in order:
            bf = brute_force(b, w, int(s))
            got = best[b["sink"][int(s)]]
            assert abs(bf - got) < 1e-9, f"viterbi {got} != brute force {bf}"

        print(f"  batch {bi}: nodes={b['num_nodes']:>6} edges={b['num_edges']:>7} "
              f"levels={len(lp) - 1:>3}  ok")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
