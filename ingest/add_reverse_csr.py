#!/usr/bin/env python3
"""
Patch an existing cuSHR .npz to add the reverse-CSR arrays (Week-4 deliverable),
WITHOUT re-running the full ingest.

The forward CSR (row_ptr, col_idx) already lives in the archive. This script
transposes it into a reverse CSR keyed by destination node:

    in_row_ptr [N+1]  incoming-edge offsets per node
    in_col_idx [E]    source node of each incoming edge
    in_edge_id [E]    original forward edge id of each incoming edge

The ordering is identical to the C++ Lattice::build_reverse_csr_(): incoming
edges of a node are listed in ascending forward-edge-id order (a stable bucket
sort by col_idx). This keeps CPU and GPU decoders bit-identical.

All original arrays are copied through unchanged; the result is written
atomically over the source file (a .bak is kept).
"""
import os
import shutil
import sys

import numpy as np

NPZ = os.path.join(os.path.dirname(__file__),
                   "..", "data", "new_cushr_data_fixed_USE_THIS.npz")


def build_reverse_csr(row_ptr, col_idx, N):
    E = col_idx.shape[0]
    # src[e] = source node of forward edge e (row_ptr groups edges by source)
    out_deg = np.diff(row_ptr)
    src = np.repeat(np.arange(N, dtype=np.int32), out_deg)
    assert src.shape[0] == E, "row_ptr/col_idx inconsistent"

    # stable bucket sort of edges by destination -> within a bucket, edges stay
    # in ascending forward-edge-id order (exactly what the C++ loop produces).
    order = np.argsort(col_idx, kind="stable").astype(np.int32)   # in_edge_id
    in_col_idx = src[order].astype(np.int32)                      # source nodes

    # in_row_ptr from per-destination counts
    counts = np.bincount(col_idx, minlength=N).astype(np.int32)
    in_row_ptr = np.zeros(N + 1, dtype=np.int32)
    np.cumsum(counts, out=in_row_ptr[1:])
    return in_row_ptr, in_col_idx, order


def main():
    path = os.path.abspath(NPZ)
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    print(f"loading {path}")
    d = np.load(path)
    data = {k: d[k] for k in d.keys()}

    if "in_row_ptr" in data:
        print("reverse-CSR already present; nothing to do.")
        return

    row_ptr = data["row_ptr"]
    col_idx = data["col_idx"]
    N = row_ptr.shape[0] - 1
    E = col_idx.shape[0]
    print(f"nodes={N:,}  edges={E:,}  -> building reverse CSR")

    in_row_ptr, in_col_idx, in_edge_id = build_reverse_csr(row_ptr, col_idx, N)

    # ---- verify the transpose really is the inverse of the forward CSR ----
    # every reverse edge must point back to a real forward edge with matching
    # endpoints, and the row_ptr totals must agree.
    assert in_row_ptr[-1] == E
    fwd_dst = col_idx[in_edge_id]              # dst of the referenced fwd edge
    # the destination implied by in_row_ptr (which bucket each rev edge sits in)
    rev_dst = np.repeat(np.arange(N, dtype=np.int32), np.diff(in_row_ptr))
    assert np.array_equal(fwd_dst, rev_dst), "reverse dst mismatch"
    # source recorded must equal source of the referenced forward edge
    out_deg = np.diff(row_ptr)
    fwd_src = np.repeat(np.arange(N, dtype=np.int32), out_deg)
    assert np.array_equal(in_col_idx, fwd_src[in_edge_id]), "reverse src mismatch"
    print("verification passed: reverse CSR is a faithful transpose")

    data["in_row_ptr"] = in_row_ptr
    data["in_col_idx"] = in_col_idx
    data["in_edge_id"] = in_edge_id

    tmp = path + ".tmp.npz"
    print(f"writing {tmp}")
    np.savez_compressed(tmp, **data)

    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"backed up original -> {bak}")
    shutil.move(tmp, path)
    print(f"done. {path} now carries in_row_ptr / in_col_idx / in_edge_id")


if __name__ == "__main__":
    main()
