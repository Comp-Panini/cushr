#!/usr/bin/env python3
"""Repack a prepare.py cache directory into a lattice .npz the C++/CUDA side loads.

WHY THIS EXISTS
---------------
cushr_batched calls Lattice::load_npz (cushr_cpu/src/lattice.cpp:56), which wants
a single .npz. prepare.py instead explodes the same arrays into one .npy per
field under ./cacheNN/. The two formats hold identical data under identical
names and dtypes -- the cache is not a lossy or derived view -- so going from
one to the other is a repack, not a conversion.

This matters for K4 specifically. The GPU needs MATERIALIZED node features: the
headline model scores over a char-BiLSTM's output and no kernel can run that
encoder. That materialization has ALREADY HAPPENED for cache95_ctx_ex4200 --
PAPER_COMPARISON.md:693-697 shows it was trained with
    --materialize ../data/g95_ctx_ex4200.npz
and the cache built from that file with
    prepare.py --npz ../data/g95_ctx_ex4200.npz --cache ./cache95_ctx_ex4200
so its node_features are already [hybrid_tag 96 | char_bilstm 96] = 192 dense
columns. Confirmed numerically: applying model95_ctx_ex4200.npz's projections
directly to those columns reproduces the documented base decoder accuracy, and
zeroing either 96-column block collapses it.

The intermediate g95_ctx_ex4200.npz was deleted (it is ~3.3 GiB). This script
reconstitutes an equivalent one from the cache, so the model does not have to be
re-materialized just to get the file back.

DTYPES ARE LOAD-BEARING
-----------------------
load_npz reads the integer arrays with cnpy's arr.data<int>() -- a raw
reinterpret, with no width conversion. An int64 column would be silently read as
garbage int32s rather than failing. Every array is therefore asserted to the
exact dtype the loader expects before anything is written.

USAGE
    python cache_to_lattice_npz.py --cache ./cache95_ctx_ex4200 \\
        --out ../data/g95_ctx_mat.npz

The output is large and uncompressed by design (np.savez, not savez_compressed):
the C++ loader has to decompress the whole thing into RAM either way, and the
cluster has the disk.
"""
import argparse
import os

import numpy as np

# name -> (required?, expected dtype). The names are exactly the keys
# Lattice::load_npz looks up; do not rename them to match anything else.
FIELDS = {
    "row_ptr":           (True,  np.int32),
    "col_idx":           (True,  np.int32),
    "topo_level":        (True,  np.int32),
    "sentence_offsets":  (True,  np.int32),
    "node_features":     (True,  np.float32),
    "gold_path_mask":    (False, np.int8),
    "node_word_length":  (False, np.int32),
    "gold_path_nodes":   (False, np.int32),
    "gold_path_offsets": (False, np.int32),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache95_ctx_ex4200")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {}
    for name, (required, dtype) in FIELDS.items():
        path = os.path.join(args.cache, name + ".npy")
        if not os.path.exists(path):
            if required:
                raise SystemExit(f"missing required {path}")
            print(f"  (optional {name} absent, skipping)")
            continue
        a = np.load(path, mmap_mode="r")
        if a.dtype != dtype:
            raise SystemExit(
                f"{name}.npy is {a.dtype}, loader reinterprets it as {dtype().dtype}; "
                "that would be read as garbage rather than fail. Refusing.")
        out[name] = np.ascontiguousarray(a)
        print(f"  {name:20s} {str(a.dtype):9s} {a.shape}")

    nf = out["node_features"]
    if nf.ndim != 2:
        raise SystemExit(f"node_features must be 2-D, got {nf.shape}")
    n_nodes = nf.shape[0]

    # The same structural invariants Lattice::validate() enforces after load.
    # Failing here names the field; failing there is a bare exception from C++.
    if out["row_ptr"].shape[0] != n_nodes + 1:
        raise SystemExit(f"row_ptr has {out['row_ptr'].shape[0]} entries, "
                         f"expected n_nodes+1 = {n_nodes + 1}")
    if int(out["row_ptr"][-1]) != out["col_idx"].shape[0]:
        raise SystemExit(f"row_ptr[-1] = {int(out['row_ptr'][-1])} but col_idx has "
                         f"{out['col_idx'].shape[0]} entries")
    if out["topo_level"].shape[0] != n_nodes:
        raise SystemExit("topo_level length does not match n_nodes")
    if int(out["sentence_offsets"][-1]) != n_nodes:
        raise SystemExit(f"sentence_offsets[-1] = {int(out['sentence_offsets'][-1])} "
                         f"!= n_nodes {n_nodes}")

    print(f"\nnodes={n_nodes:,}  edges={out['col_idx'].shape[0]:,}  "
          f"sentences={out['sentence_offsets'].shape[0] - 1:,}  "
          f"feat_dim={nf.shape[1]}")
    print(f"writing {args.out} (~{sum(a.nbytes for a in out.values()) / 2**30:.2f} GiB) ...")
    np.savez(args.out, **out)
    print("done. feat_dim must match the --model .bin; the driver checks it.")


if __name__ == "__main__":
    main()
