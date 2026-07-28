#!/usr/bin/env python3

# purpose of this file is to pack sentences into batches for training
# merge lattices into one batch

import json
import os

import numpy as np

# this class opens .npy files, caches gold path results, saves them to disk
class LatticeStore:

    def __init__(self, cache="./cache"):
        self.cache = cache
        m = lambda n: np.load(os.path.join(cache, n + ".npy"), mmap_mode="r")
        self.node_features = m("node_features")    
        self.word_len = m("node_word_length")       
        self.row_ptr = np.load(os.path.join(cache, "row_ptr.npy"))
        self.col_idx = m("col_idx")                
        self.topo_level = np.load(os.path.join(cache, "topo_level.npy"))
        self.sent_off = np.load(os.path.join(cache, "sentence_offsets.npy"))
        self.gold_nodes = np.load(os.path.join(cache, "gold_path_nodes.npy"))
        self.gold_off = np.load(os.path.join(cache, "gold_path_offsets.npy"))

        with open(os.path.join(cache, "splits.json")) as f:
            blob = json.load(f)
        self.meta = blob["meta"]
        self.splits = {k: np.asarray(v, dtype=np.int64)
                       for k, v in blob["splits"].items()}

        # The featurizer owns every column now, including the length feature
        # that this class used to append. feat_dim is whatever the data says.
        self.feat_dim = int(self.node_features.shape[1])
        self.featurizer_name = self.meta.get("featurizer", "unknown")
        declared = self.meta.get("feat_dim")
        if declared is not None and int(declared) != self.feat_dim:
            raise ValueError(
                f"cache is inconsistent: splits.json declares feat_dim="
                f"{declared} but node_features.npy has {self.feat_dim} columns. "
                "Re-run prepare.py against the featurized archive.")

        self._gold_edges: np.ndarray | None = None
        self._gold_edge_off: np.ndarray | None = None

    # cache gold path results and save to disk
    def _ensure_gold_edges(self):
        if self._gold_edges is not None:
            return
        ep = os.path.join(self.cache, "gold_edges.npy")
        op = os.path.join(self.cache, "gold_edge_offsets.npy")
        if os.path.exists(ep) and os.path.exists(op):
            self._gold_edges = np.load(ep)
            self._gold_edge_off = np.load(op)
            return

        n_sent = len(self.sent_off) - 1
        col = np.asarray(self.col_idx)
        offs = np.zeros(n_sent + 1, dtype=np.int64)
        chunks = []
        for s in range(n_sent):
            b, e = int(self.sent_off[s]), int(self.sent_off[s + 1])
            g = self.gold_nodes[self.gold_off[s]:self.gold_off[s + 1]]
            if len(g) == 0:
                offs[s + 1] = offs[s]
                continue
            path = [b] + [int(x) for x in g] + [e - 1]
            eids = []
            ok = True
            for u, v in zip(path[:-1], path[1:]):
                lo, hi = int(self.row_ptr[u]), int(self.row_ptr[u + 1])
                hit = np.nonzero(col[lo:hi] == v)[0]
                if len(hit) == 0:
                    ok = False
                    break
                eids.append(lo + int(hit[0]))
            if not ok:
                offs[s + 1] = offs[s]
                continue
            chunks.append(np.asarray(eids, dtype=np.int64))
            offs[s + 1] = offs[s] + len(eids)
        self._gold_edges = (np.concatenate(chunks) if chunks
                            else np.zeros(0, dtype=np.int64))
        self._gold_edge_off = offs
        np.save(ep, self._gold_edges)
        np.save(op, self._gold_edge_off)

    def gold_edges(self, s):
        self._ensure_gold_edges()
        edges, off = self._gold_edges, self._gold_edge_off
        assert edges is not None and off is not None
        return edges[off[s]:off[s + 1]]

    def trainable(self, split):
        self._ensure_gold_edges()
        off = self._gold_edge_off
        assert off is not None
        ids = self.splits[split]
        n = off[ids + 1] - off[ids]
        return ids[n > 0]

    def features(self, node_ids):
        # No log1p(word_length) append here any more: the featurizer already
        # emitted that column. Appending it again would silently widen the
        # vector past what the exported C++ weights expect.
        return np.asarray(self.node_features[node_ids], dtype=np.float32)


class Batch(dict):
    """Plain dict; keys documented in collate()."""

# concatenate nodes into a flat array
# remap edges to local IDs
# sort the EDGE array by dest topo level
# map gold path to local edge IDs
# build gold_node which is a boolean mask over nodes on gold path
def collate(store: LatticeStore, sent_ids):
    sent_ids = np.asarray(sent_ids, dtype=np.int64)
    begins = store.sent_off[sent_ids].astype(np.int64)
    ends = store.sent_off[sent_ids + 1].astype(np.int64)
    sizes = ends - begins

    node_off = np.zeros(len(sent_ids) + 1, dtype=np.int64)
    np.cumsum(sizes, out=node_off[1:])
    nb = int(node_off[-1])

    global_node = np.concatenate([np.arange(b, e, dtype=np.int64)
                                  for b, e in zip(begins, ends)])
    node_sent = np.repeat(np.arange(len(sent_ids), dtype=np.int64), sizes)
    shift = node_off[:-1] - begins                      

    e_lo = store.row_ptr[global_node].astype(np.int64)
    e_hi = store.row_ptr[global_node + 1].astype(np.int64)
    deg = e_hi - e_lo
    src_local = np.repeat(np.arange(nb, dtype=np.int64), deg)
    total_e = int(deg.sum())
    starts = np.zeros(nb + 1, dtype=np.int64)
    np.cumsum(deg, out=starts[1:])
    idx = np.arange(total_e, dtype=np.int64) - np.repeat(starts[:-1], deg)
    edge_global = np.repeat(e_lo, deg) + idx
    dst_global = np.asarray(store.col_idx[edge_global], dtype=np.int64)
    dst_local = dst_global + shift[node_sent[src_local]]

    # do stable sort on edges
    dst_level = store.topo_level[dst_global].astype(np.int64)
    order = np.argsort(dst_level, kind="stable")
    src_local = src_local[order]
    dst_local = dst_local[order]
    edge_global = edge_global[order]
    dst_level = dst_level[order]

    max_level = int(dst_level[-1]) if total_e else 0
    level_ptr = np.searchsorted(dst_level, np.arange(max_level + 2))

    perm = np.argsort(edge_global, kind="stable")
    eg_sorted = edge_global[perm]
    gold_chunks, gptr = [], np.zeros(len(sent_ids) + 1, dtype=np.int64)
    for i, s in enumerate(sent_ids):
        ge = store.gold_edges(int(s))
        gold_chunks.append(ge)
        gptr[i + 1] = gptr[i] + len(ge)
    if gold_chunks:
        flat = np.concatenate(gold_chunks)
        hit = np.searchsorted(eg_sorted, flat)
        assert np.all(eg_sorted[hit] == flat), "gold edge not present in batch"
        gold_edge = perm[hit]
    else:
        gold_edge = np.zeros(0, dtype=np.int64)

    gold_node = np.zeros(nb, dtype=bool)
    for i, s in enumerate(sent_ids):
        g = store.gold_nodes[store.gold_off[s]:store.gold_off[s + 1]]
        gold_node[np.asarray(g, dtype=np.int64) + shift[i]] = True
    gold_node[node_off[:-1]] = True     
    gold_node[node_off[1:] - 1] = True   

    return Batch(
        feats=store.features(global_node),
        src=src_local, dst=dst_local,
        level_ptr=level_ptr,
        source=node_off[:-1].copy(), sink=(node_off[1:] - 1).copy(),
        gold_edge=gold_edge, gold_edge_ptr=gptr,
        gold_node=gold_node, node_sent=node_sent,
        global_node=global_node, sent_ids=sent_ids,
        num_nodes=nb, num_edges=total_e,
    )


def gold_word_sets(store, sent_ids):
    """Gold word nodes (corpus ids) per sentence -- the F1 reference."""
    out = []
    for s in sent_ids:
        g = store.gold_nodes[store.gold_off[s]:store.gold_off[s + 1]]
        out.append([int(x) for x in g])
    return out
