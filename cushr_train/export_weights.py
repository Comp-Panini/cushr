#!/usr/bin/env python3

# python to Cuda bridge
# produces model_biaffine.bin that will be read by cuda scorer
# edge_score.npy precomputes biaffine score for every edge so gpu doesnt have to do it

import argparse
import struct

import numpy as np

MAGIC = 0x43534232


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

    with open(args.bin, "wb") as f:
        f.write(struct.pack("<iiif", MAGIC, feat_dim, hidden, bias))
        f.write(src.tobytes())
        f.write(dst.tobytes())
    print(f"wrote {args.bin}  feat_dim={feat_dim} hidden={hidden} bias={bias:.6f}")

    if not args.edge_scores:
        return

    import os
    nf = np.load(os.path.join(args.cache, "node_features.npy"), mmap_mode="r")
    wl = np.load(os.path.join(args.cache, "node_word_length.npy"), mmap_mode="r")
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
        x = np.concatenate(
            [np.asarray(nf[nb0:nb1], dtype=np.float32),
             np.log1p(np.asarray(wl[nb0:nb1], dtype=np.float32))[:, None]],
            axis=1)
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
