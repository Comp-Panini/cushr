#!/usr/bin/env python3

#purpose: find best path for sentence

import torch

NEG_INF = -1e30

# pytorch always does tracking in the background to compute gradient
# this function says dont build any breadcrumb trail
# use it during 1) decoding when training and 2) evaluating
# non differentiable stuff like argmax
@torch.no_grad()

# DP releaxation level by level
def viterbi(batch_t, w):
    src, dst = batch_t["src"], batch_t["dst"]
    level_ptr = batch_t["level_ptr"]
    nb = batch_t["num_nodes"]
    source, sink = batch_t["source"], batch_t["sink"]
    dev = w.device

    best = torch.full((nb,), NEG_INF, device=dev, dtype=w.dtype)
    best[source] = 0.0

    for l in range(len(level_ptr) - 1):
        lo, hi = int(level_ptr[l]), int(level_ptr[l + 1])
        if hi <= lo:
            continue
        s, d = src[lo:hi], dst[lo:hi]
        cand = best[s] + w[lo:hi]
        best.scatter_reduce_(0, d, cand, reduce="amax", include_self=True)

    eps = 1e-5
    slack = (best[src] + w) - best[dst]
    eid = torch.arange(w.numel(), device=dev, dtype=torch.int64)
    big = torch.full((1,), w.numel(), device=dev, dtype=torch.int64)
    keyed = torch.where(slack.abs() <= eps, eid, big.expand_as(eid))
    parent = torch.full((nb,), w.numel(), device=dev, dtype=torch.int64)
    parent.scatter_reduce_(0, dst, keyed, reduce="amin", include_self=True)

    # walk back
    max_len = len(level_ptr)
    cur = sink.clone()
    at_source = cur == source
    edges, masks = [], []
    for _ in range(max_len):
        alive = ~at_source
        if not bool(alive.any()):
            break
        pe = parent[cur].clamp(max=w.numel() - 1)
        broken = parent[cur] >= w.numel()
        alive = alive & ~broken
        edges.append(torch.where(alive, pe, torch.zeros_like(pe)))
        masks.append(alive)
        nxt = src[pe]
        cur = torch.where(alive, nxt, cur)
        at_source = at_source | broken | (cur == source)

    if not edges:
        z = torch.zeros((source.numel(), 0), dtype=torch.int64, device=dev)
        return z, z.bool(), torch.zeros(source.numel(), device=dev, dtype=w.dtype)

    path_edges = torch.stack(edges, dim=1)
    path_mask = torch.stack(masks, dim=1)
    return path_edges, path_mask, best[sink]

# sum up edge scores for path we predict
def path_score(w, path_edges, path_mask):
    return (w[path_edges] * path_mask.to(w.dtype)).sum(-1)

# sum up edge scores for gold path
def gold_score(w, gold_edge, gold_edge_ptr, num_sent):
    seg = torch.repeat_interleave(
        torch.arange(num_sent, device=w.device),
        gold_edge_ptr[1:] - gold_edge_ptr[:-1],
    )
    out = torch.zeros(num_sent, device=w.device, dtype=w.dtype)
    out.scatter_add_(0, seg, w[gold_edge])
    return out

# extract nodes to get F1
def predicted_nodes(batch_t, path_edges, path_mask):
    dst = batch_t["dst"]
    sink = batch_t["sink"]
    out = []
    pe = path_edges.cpu()
    pm = path_mask.cpu()
    d = dst.cpu()
    sk = sink.cpu()
    for i in range(pe.shape[0]):
        nodes = d[pe[i][pm[i]]].tolist()
        out.append([n for n in nodes if n != int(sk[i])])
    return out
