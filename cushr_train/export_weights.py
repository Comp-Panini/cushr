#!/usr/bin/env python3

# python to Cuda bridge
# produces model_biaffine.bin, the CSB3 weight file read by BOTH the C++
# BiaffineScorer (cushr_cpu/src/scorer.cpp) and the K4 CUDA kernels
# (cushr_gpu/score_edges.cu).
#
# --edge-scores writes edge_score.npy, a precomputed biaffine score for every
# edge. That used to be how the GPU got its scores. As of week 10 it is NOT:
# K4 computes them on the device from the .bin, and cushr_batched never reads
# this file. It is still worth producing for check_export.py, which uses it to
# verify the exported weights reproduce the trained model's edge scores.

import argparse
import struct

import numpy as np

# 'CSB3'. See the matching constant in cushr_cpu/src/scorer.cpp: under the old
# 'CSB2' contract the C++ scorer appended log1p(word_length) as a final feature
# column itself. Featurizers now emit every column, so the file format changes
# version to make a stale weight file fail loudly at load.
MAGIC = 0x43534233


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_biaffine.npz")
    ap.add_argument("--bin", default="model_biaffine.bin")
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--edge-scores", default="")
    args = ap.parse_args()

    m = np.load(args.model)
    feat_dim = int(m["feat_dim"])
    hidden = int(m["hidden"])
    src = np.ascontiguousarray(m["src_proj"], dtype=np.float32)  
    dst = np.ascontiguousarray(m["dst_proj"], dtype=np.float32)
    bias = float(np.asarray(m["bias"]).reshape(-1)[0])
    assert src.shape == (hidden, feat_dim) and dst.shape == (hidden, feat_dim)
    if "featurizer_name" in m.files:
        print(f"model featurizer: {m['featurizer_name']}")

    with open(args.bin, "wb") as f:
        f.write(struct.pack("<iiif", MAGIC, feat_dim, hidden, bias))
        f.write(src.tobytes())
        f.write(dst.tobytes())
    print(f"wrote {args.bin}  feat_dim={feat_dim} hidden={hidden} bias={bias:.6f}")

    if not args.edge_scores:
        return

    import os
    nf = np.load(os.path.join(args.cache, "node_features.npy"), mmap_mode="r")
    row_ptr = np.load(os.path.join(args.cache, "row_ptr.npy"))
    col_idx = np.load(os.path.join(args.cache, "col_idx.npy"), mmap_mode="r")
    sent_off = np.load(os.path.join(args.cache, "sentence_offsets.npy"))
    n = row_ptr.shape[0] - 1
    e_total = int(row_ptr[-1])

    out = np.lib.format.open_memmap(args.edge_scores, mode="w+",
                                    dtype=np.float32, shape=(e_total,))
    n_sent = len(sent_off) - 1
    block_sent = 20000
    for s0 in range(0, n_sent, block_sent):
        s1 = min(s0 + block_sent, n_sent)
        nb0, nb1 = int(sent_off[s0]), int(sent_off[s1])
        # Features verbatim: the featurizer already emitted every column,
        # including the length one this used to append.
        x = np.asarray(nf[nb0:nb1], dtype=np.float32)
        S = x @ src.T
        D = x @ dst.T
        e0, e1 = int(row_ptr[nb0]), int(row_ptr[nb1])
        deg = np.diff(row_ptr[nb0:nb1 + 1]).astype(np.int64)
        u = np.repeat(np.arange(nb1 - nb0, dtype=np.int64), deg)
        v = np.asarray(col_idx[e0:e1], dtype=np.int64) - nb0
        assert v.min() >= 0 and v.max() < (nb1 - nb0), "edge crosses block"
        out[e0:e1] = np.einsum("ij,ij->i", S[u], D[v]) + bias
        print(f"  sentences {s0}-{s1}  edges {e0}-{e1}")
    out.flush()
    print(f"wrote {args.edge_scores}  edges={e_total}  "
          f"mean={out.mean():.4f} std={out.std():.4f}")


if __name__ == "__main__":
    main()
