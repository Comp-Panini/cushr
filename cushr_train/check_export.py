#!/usr/bin/env python3

# check if export worked

import argparse
import os
import struct

import numpy as np

MAGIC = 0x43534232


def read_bin(path):
    with open(path, "rb") as f:
        magic, feat_dim, hidden, bias = struct.unpack("<iiif", f.read(16))
        assert magic == MAGIC, f"bad magic {magic:#x}"
        n = hidden * feat_dim
        src = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(hidden, feat_dim)
        dst = np.frombuffer(f.read(n * 4), dtype="<f4").reshape(hidden, feat_dim)
    return feat_dim, hidden, bias, src.copy(), dst.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_biaffine.npz")
    ap.add_argument("--bin", default="model_biaffine.bin")
    ap.add_argument("--edge-scores", default="edge_score.npy")
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    m = np.load(args.model)
    feat_dim, hidden, bias, src, dst = read_bin(args.bin)

    assert feat_dim == int(m["feat_dim"]) and hidden == int(m["hidden"])
    assert np.array_equal(src, m["src_proj"]), "src_proj differs in .bin"
    assert np.array_equal(dst, m["dst_proj"]), "dst_proj differs in .bin"
    assert bias == float(np.asarray(m["bias"]).reshape(-1)[0]), "bias differs"
    print(f"bin round-trip ok: feat_dim={feat_dim} hidden={hidden} bias={bias:.6f}")

    if not os.path.exists(args.edge_scores):
        print("no edge_score.npy; skipping edge check")
        return

    nf = np.load(os.path.join(args.cache, "node_features.npy"), mmap_mode="r")
    wl = np.load(os.path.join(args.cache, "node_word_length.npy"), mmap_mode="r")
    row_ptr = np.load(os.path.join(args.cache, "row_ptr.npy"))
    col_idx = np.load(os.path.join(args.cache, "col_idx.npy"), mmap_mode="r")
    es = np.load(args.edge_scores, mmap_mode="r")

    e_total = int(row_ptr[-1])
    assert es.shape[0] == e_total, "edge_score length != edge count"

    rng = np.random.default_rng(0)
    eids = np.sort(rng.choice(e_total, size=args.n, replace=False))
    u = np.searchsorted(row_ptr, eids, side="right") - 1
    v = np.asarray(col_idx[eids], dtype=np.int64)

    def x(nodes):
        return np.concatenate(
            [np.asarray(nf[nodes], dtype=np.float32),
             np.log1p(np.asarray(wl[nodes], dtype=np.float32))[:, None]], axis=1)

    want = np.einsum("ij,ij->i", x(u) @ src.T, x(v) @ dst.T) + bias
    got = np.asarray(es[eids])
    err = np.abs(want - got)
    print(f"edge_score check on {args.n} random edges: "
          f"max|err|={err.max():.3e}  mean|err|={err.mean():.3e}")
    assert err.max() < 1e-3, "edge_score.npy disagrees with the weights"

    print("\nreference values -- a C++ BiaffineScorer run must match these:")
    for k in range(5):
        print(f"  edge {int(eids[k]):>9}  ({int(u[k])} -> {int(v[k])})  "
              f"score {float(got[k]): .6f}")
    print("\nall export checks passed")


if __name__ == "__main__":
    main()
