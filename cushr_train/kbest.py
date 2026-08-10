#!/usr/bin/env python3
"""Exact K-best paths over the lattice, mirroring viterbi.py's DP.

`viterbi.py` is deliberately NOT modified: it is the 1-best path behind every
published number (91.52 / 85.40 / 51.95), and the whole point of this module is
to be checkable against it. `test_kbest.py` asserts K=1 reproduces it exactly.

THE ONE CHANGED LINE
--------------------
`viterbi()` keeps a single best score per node:

    best.scatter_reduce_(0, d, cand, reduce="amax")

Here each node keeps its top K, so the reduction becomes a *segmented top-K*
over all incoming candidates, grouped by destination. Everything else -- the
level-by-level relaxation order, the NEG_INF padding, the backtrack -- is the
same shape as the original.

WHY A SINGLE WRITE PER NODE IS COMPLETE
---------------------------------------
`collate` sorts edges by their destination's topological level, so ALL incoming
edges of a node fall in one level block. A node's top-K is therefore fully
determined by one level's candidates and never needs merging with a previous
partial result. This is what lets the update be a plain indexed assignment
rather than a read-modify-write.

TIE-BREAKING, AND WHY IT IS NOT AN ARBITRARY CHOICE
---------------------------------------------------
`viterbi()` breaks score ties by *lowest local edge id* (viterbi.py:36-40, a
`scatter_reduce_(amin)` over edge ids). To reproduce that exactly, candidates
are laid out edge-major / rank-minor and sorted with two STABLE sorts: score
descending first, then destination. Stability preserves the edge-major layout
among equal scores, so ties resolve to the smaller edge id and then the smaller
parent rank -- identical to `viterbi()`. Without stable sorts K=1 would match
only "up to ties", which is not a testable property.

The C++ `TopKDecoder` breaks ties differently -- by (score, parent_node,
parent_rank) at cushr_cpu/src/decoder.cpp:14-20 -- so its paths can legitimately
differ from these on exact float ties. `tie_break="cpp"` switches the minor key
for anyone who wants to compare paths rather than score multisets; nothing in
this repo uses it.
"""
import torch

from viterbi import NEG_INF

# NEG_INF is -1e30 and scores accumulate, so a padded lane can drift well away
# from the sentinel. Half of it is far below any real path score and far above
# any sum of sentinels.
_DEAD = NEG_INF / 2


@torch.no_grad()
def kbest(batch_t, w, K, tie_break="edge_id"):
    """Top-K paths per sentence.

    Returns (path_edges [B,K,L] local edge ids, path_mask [B,K,L],
             scores [B,K], valid [B,K]).

    `valid` marks ranks that correspond to a real path: a sentence with fewer
    than K distinct paths fills the remaining ranks with NEG_INF. Valid ranks
    always form a prefix, and `scores` is non-increasing across them.
    """
    src, dst = batch_t["src"], batch_t["dst"]
    level_ptr = batch_t["level_ptr"]
    nb = batch_t["num_nodes"]
    source, sink = batch_t["source"], batch_t["sink"]
    dev, n_edges = w.device, w.numel()

    best = torch.full((nb, K), NEG_INF, device=dev, dtype=w.dtype)
    best[source, 0] = 0.0
    parent_edge = torch.full((nb, K), n_edges, device=dev, dtype=torch.int64)
    parent_rank = torch.full((nb, K), -1, device=dev, dtype=torch.int64)

    ar_K = torch.arange(K, device=dev)

    for l in range(len(level_ptr) - 1):
        lo, hi = int(level_ptr[l]), int(level_ptr[l + 1])
        if hi <= lo:
            continue
        s, d = src[lo:hi], dst[lo:hi]
        E = hi - lo

        # [E, K] -> flat, edge-major / rank-minor. That layout IS the tie-break.
        cand = (best[s] + w[lo:hi, None]).reshape(-1)
        eid = torch.arange(lo, hi, device=dev).repeat_interleave(K)
        rnk = ar_K.repeat(E)
        dd = d.repeat_interleave(K)

        alive = cand > _DEAD
        if not bool(alive.any()):
            continue
        cand, eid, rnk, dd = cand[alive], eid[alive], rnk[alive], dd[alive]

        # Two stable sorts: minor key = score desc (or the C++ key), major = dst.
        if tie_break == "cpp":
            o1 = torch.argsort(src[eid] * K + rnk, stable=True)
            o1 = o1[torch.argsort(cand[o1], descending=True, stable=True)]
        else:
            o1 = torch.argsort(cand, descending=True, stable=True)
        p = o1[torch.argsort(dd[o1], stable=True)]

        g = dd[p]
        newseg = torch.ones_like(g, dtype=torch.bool)
        newseg[1:] = g[1:] != g[:-1]
        segid = newseg.cumsum(0) - 1
        starts = torch.nonzero(newseg).squeeze(1)
        r_in_seg = torch.arange(g.numel(), device=dev) - starts[segid]

        keep = r_in_seg < K
        gk, rk = g[keep], r_in_seg[keep]
        # (dst, rank) pairs are unique by construction, so this is a safe
        # scatter even though advanced-index assignment is not atomic.
        best[gk, rk] = cand[p][keep]
        parent_edge[gk, rk] = eid[p][keep]
        parent_rank[gk, rk] = rnk[p][keep]

    B = source.numel()
    scores = best[sink]                       # [B, K]
    valid = scores > _DEAD

    cur_n = sink[:, None].expand(B, K).clone()
    cur_r = ar_K[None, :].expand(B, K).clone()
    done = ~valid | (cur_n == source[:, None])

    edges, masks = [], []
    for _ in range(len(level_ptr)):
        e = parent_edge[cur_n, cur_r]
        alive = ~done & (e < n_edges)
        if not bool(alive.any()):
            break
        ec = e.clamp(max=n_edges - 1)
        edges.append(torch.where(alive, ec, torch.zeros_like(ec)))
        masks.append(alive)
        nr = parent_rank[cur_n, cur_r].clamp(min=0)
        cur_n = torch.where(alive, src[ec], cur_n)
        cur_r = torch.where(alive, nr, cur_r)
        done = done | ~alive | (cur_n == source[:, None])

    if not edges:
        z = torch.zeros((B, K, 0), dtype=torch.int64, device=dev)
        return z, z.bool(), scores, valid
    return (torch.stack(edges, dim=2), torch.stack(masks, dim=2), scores, valid)


def predicted_nodes_k(batch_t, path_edges, path_mask, valid):
    """[B][K] lists of LOCAL node ids, mirroring viterbi.predicted_nodes.

    Callers remap through batch["global_node"], exactly as with the 1-best path,
    and must sort by node_char_start -- these come back in backtracked order.
    """
    dst = batch_t["dst"].cpu()
    sink = batch_t["sink"].cpu()
    pe, pm, va = path_edges.cpu(), path_mask.cpu(), valid.cpu()
    out = []
    for i in range(pe.shape[0]):
        sk = int(sink[i])
        rows = []
        for k in range(pe.shape[1]):
            if not bool(va[i, k]):
                break
            nodes = dst[pe[i, k][pm[i, k]]].tolist()
            rows.append([n for n in nodes if n != sk])
        out.append(rows)
    return out


def keep_stats(batch_t, w, K):
    """Histogram of how full each node's K-buffer got.

    Not a correctness check -- the DP is exact regardless. It answers whether
    the LATTICE can even supply K distinct paths, which tells you when raising
    K stops buying recall. Ported in spirit from the C++ decoder's KeepStats.
    """
    src, dst = batch_t["src"], batch_t["dst"]
    level_ptr = batch_t["level_ptr"]
    nb = batch_t["num_nodes"]

    # npaths[v] = number of source->v paths, saturated at K. Saturating every
    # level keeps this in int64 range: the true path count is exponential in
    # sentence length and would overflow long before the sink.
    npaths = torch.zeros(nb, dtype=torch.int64, device=w.device)
    npaths[batch_t["source"]] = 1
    for l in range(len(level_ptr) - 1):
        lo, hi = int(level_ptr[l]), int(level_ptr[l + 1])
        if hi <= lo:
            continue
        npaths.scatter_add_(0, dst[lo:hi], npaths[src[lo:hi]])
        npaths.clamp_(max=K)
    return npaths[batch_t["sink"]]
