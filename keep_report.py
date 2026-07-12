#!/usr/bin/env python3
"""keep_report.py -- reproduce the CPU TopKDecoder "keep" histogram in Python.

`keep = min(K, buf.size())` in decoder.cpp, and buf.size() for node v is the sum
of the beam sizes (counts) of v's parents. That is purely structural -- it does
NOT depend on edge scores -- so we can compute the exact same distribution from
the lattice alone, no C++ build / GPU needed.

    count[v] = 1                              if v is a source (no in-edges)
    count[v] = min(K, sum_{u->v} count[u])    otherwise

Only non-source nodes with a non-empty candidate buffer are counted in the
histogram (matching the C++ instrumentation).

Usage:
    python keep_report.py data/new_cushr_data_fixed_USE_THIS.npz --K 1 --out keep_k1.csv
    python keep_report.py data/...npz --K 1,8,16,32,64      # writes keep_k<K>.csv each
"""
import argparse
import numpy as np


def compute_keep(in_row_ptr, in_col_idx, topo_level, K):
    N = topo_level.shape[0]
    count = np.zeros(N, dtype=np.int64)

    indeg = np.diff(in_row_ptr)          # incoming edges per node
    is_source = indeg == 0
    count[is_source] = 1                 # sources seed one path

    seg_starts = in_row_ptr[:-1]         # reduceat segment boundaries

    # Process nodes level by level in ascending topo order so every parent's
    # count is final before its children are evaluated.
    levels = np.unique(topo_level)
    for L in levels:
        at_level = (topo_level == L) & (~is_source)
        if not at_level.any():
            continue
        # sum of parent counts for every node, using current (final for lower
        # levels) counts. reduceat is cheap and we only read the level-L rows.
        sums = np.add.reduceat(count[in_col_idx], seg_starts)
        # reduceat returns a garbage value for empty segments (sources), but we
        # masked those out via at_level, so ignore them.
        nonempty = at_level & (sums > 0)
        count[nonempty] = np.minimum(K, sums[nonempty])

    # histogram over the merging nodes only (non-source, non-empty buffer)
    counted = (~is_source) & (count > 0)
    keep_vals = count[counted]
    hist = np.bincount(keep_vals, minlength=K + 1)[: K + 1]
    return hist, int(counted.sum())


def write_report(path, K, hist, total):
    full = int(hist[K])
    under = total - full
    fp = 100.0 * full / total if total else 0.0
    up = 100.0 * under / total if total else 0.0
    summary = (f"# K={K}  total_nodes={total}  "
               f"full(keep==K)={full} ({fp:.2f}%)  "
               f"under(keep<K)={under} ({up:.2f}%)")
    with open(path, "w") as f:
        f.write(summary + "\n")
        f.write("keep,num_nodes\n")
        for k in range(K + 1):
            f.write(f"{k},{int(hist[k])}\n")
    print(summary)
    print(f"Wrote keep report to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--K", default="1", help="comma-separated K values")
    ap.add_argument("--out", default=None, help="output CSV (single K only)")
    args = ap.parse_args()

    d = np.load(args.npz)
    in_row_ptr = d["in_row_ptr"].astype(np.int64)
    in_col_idx = d["in_col_idx"].astype(np.int64)
    topo_level = d["topo_level"].astype(np.int64)

    Ks = [int(x) for x in args.K.split(",") if x]
    for K in Ks:
        hist, total = compute_keep(in_row_ptr, in_col_idx, topo_level, K)
        out = args.out if (args.out and len(Ks) == 1) else f"keep_k{K}.csv"
        write_report(out, K, hist, total)


if __name__ == "__main__":
    main()
