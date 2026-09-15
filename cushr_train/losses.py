#!/usr/bin/env python3
"""Pluggable training objectives for train.py.

Two independent choices, selected by flags:

  --cost {node, surface}
      node     cost 1 for every predicted node that is not the exact gold node
               (lemma + cng included). This is the original objective and the
               default, so existing runs reproduce unchanged.
      surface  cost 1 only when the predicted node is a different SEGMENTATION
               word: a different (char_start, word_len, form) than every gold
               word. A same-segmentation node with the wrong lemma/cng costs
               --morph-cost instead (default 0).

  --gold-target {exact, surface}
      exact    the hinge compares against the exact gold node path (original).
      surface  the hinge compares against the best-scoring path that uses only
               surface-gold nodes, i.e. any path spelling the gold segmentation.
               Without this, a morph variant of the right segmentation that
               outscores the gold path still produces loss even at zero cost --
               the model is still being taught morphology.

For segmentation-only training use `--cost surface --gold-target surface`.
The exported scorer, materialised features and decoders are untouched by any of
this: only what the scorer is trained toward changes.
"""

import numpy as np
import torch

from viterbi import viterbi, path_score, gold_score

COSTS = ("node", "surface")
GOLD_TARGETS = ("exact", "surface")

# Additive penalty that keeps the constrained decode on surface-gold nodes. Edge
# scores are O(10); this dominates without overflowing float32 path sums.
_OFF_PATH = -1e4


class SurfaceTable:
    """Per-node segmentation identity: (char_start, word_len, form id)."""

    def __init__(self, store, raw_path=""):
        if not store.has_chars:
            raise SystemExit("surface objective needs node_char_start in the "
                             "cache (same requirement as --encoder).")
        if raw_path:
            # Full-vocabulary form ids: the thresholded node_ids column maps
            # rare forms to <UNK>, which would make two different rare words at
            # the same span look identical.
            fid = np.load(raw_path)["node_form_id"]
            src = f"{raw_path}:node_form_id"
        elif store.node_ids is not None:
            fid = store.node_ids[:, 0]
            src = "cache node_ids[:,0] (thresholded; pass --surface-raw for exact)"
        else:
            raise SystemExit("surface objective needs form ids: pass "
                             "--surface-raw <ingest npz> or a cache built "
                             "with --emit-ids.")
        n = store.node_features.shape[0]
        if len(fid) != n:
            raise SystemExit(f"form id array has {len(fid):,} nodes, cache has "
                             f"{n:,}; --surface-raw must be the archive the "
                             "cache was prepared from.")
        self.form = np.asarray(fid, dtype=np.int64)
        self.char_start = store.char_start
        self.word_len = store.word_len
        print(f"surface identity: form ids from {src}")

    def keys(self, global_nodes):
        g = np.asarray(global_nodes, dtype=np.int64)
        return np.stack([np.asarray(self.char_start[g], dtype=np.int64),
                         np.asarray(self.word_len[g], dtype=np.int64),
                         self.form[g]], axis=1)

    def gold_mask(self, batch):
        """Bool over batch nodes: node spells a gold word of its own sentence.
        Exact gold nodes (boundaries included) are always True."""
        return self.gold_rank(batch) >= 0

    def gold_rank(self, batch):
        """Int over batch nodes: position of the gold word this node spells in
        its sentence (source 0, words 1..n in character order, sink n+1), or -1.

        Membership alone is not enough to define "a path spelling the gold
        segmentation": the lattice has edges that jump past words, so a path of
        surface-gold nodes can silently drop gold words. Requiring consecutive
        ranks along every edge (see edge_ok) rules that out."""
        gold = np.asarray(batch["gold_node"])
        sent = np.asarray(batch["node_sent"])
        keys = self.keys(batch["global_node"])
        k = np.concatenate([sent[:, None], keys], axis=1)
        _, inv = np.unique(k, axis=0, return_inverse=True)
        inv = inv.reshape(-1)

        gi = np.nonzero(gold)[0]
        cls = np.ones(len(gi), dtype=np.int64)          # 1 = word
        cls[np.isin(gi, batch["source"])] = 0
        cls[np.isin(gi, batch["sink"])] = 2
        order = np.lexsort((keys[gi, 0], cls, sent[gi]))
        gi, gs = gi[order], sent[gi[order]]
        first = np.searchsorted(gs, gs)                  # gs is sorted
        r = np.arange(len(gi)) - first

        rank_of_key = np.full(inv.max() + 1, -1, dtype=np.int64)
        rank_of_key[inv[gi]] = r
        rank = rank_of_key[inv]
        rank[gi] = r        # boundaries share a key; pin exact gold nodes
        return rank

    @staticmethod
    def edge_ok(rank, src, dst):
        """Edge stays on a gold-segmentation path: consecutive gold words."""
        return (rank[src] >= 0) & (rank[dst] == rank[src] + 1)

    def word_set(self, global_nodes):
        """Segmentation words of a path, for surface P/R/F/PM."""
        return {tuple(r) for r in self.keys(global_nodes).tolist()
                if r[0] >= 0 and r[1] > 0}


class MarginObjective:
    """Structured hinge: relu(cost(pred) + s(pred) - s(gold))."""

    def __init__(self, margin=1.0, cost="node", gold_target="exact",
                 morph_cost=0.0, surface=None):
        assert cost in COSTS and gold_target in GOLD_TARGETS
        self.margin, self.cost, self.gold_target = margin, cost, gold_target
        self.morph_cost = morph_cost
        self.surface = surface
        if self.needs_surface and surface is None:
            raise ValueError("surface cost/gold target needs a SurfaceTable")

    @property
    def needs_surface(self):
        return self.cost == "surface" or self.gold_target == "surface"

    def describe(self):
        s = f"cost={self.cost} gold_target={self.gold_target} margin={self.margin}"
        if self.cost == "surface":
            s += f" morph_cost={self.morph_cost}"
        return s

    def __call__(self, w, t, b, n_sent):
        dst = t["dst"]
        gold_node = t["gold_node"]
        surf = ok = None
        if self.needs_surface:
            rank = torch.as_tensor(self.surface.gold_rank(b), device=w.device)
            surf = rank >= 0
            ok = SurfaceTable.edge_ok(rank, t["src"], dst)

        if self.cost == "node":
            # Original objective, kept bit-for-bit.
            cost = self.margin * (~gold_node[dst]).to(w.dtype)
        else:
            # Charged per edge that leaves the gold segmentation: a wrong word,
            # or a jump that skips a gold word (which node membership alone
            # would not charge -- recall errors would then be free).
            morph_only = surf & ~gold_node
            cost = self.margin * ((~ok).to(w.dtype)
                                  + self.morph_cost * morph_only[dst].to(w.dtype))

        pe, pmask, _ = viterbi(t, (w + cost).detach())
        s_pred = path_score(w, pe, pmask)
        hamming = (cost[pe] * pmask.to(w.dtype)).sum(-1)

        if self.gold_target == "exact":
            s_gold = gold_score(w, t["gold_edge"], t["gold_edge_ptr"], n_sent)
        else:
            # Latent gold: best path restricted to surface-gold nodes. The exact
            # gold path is one such path, so a feasible one always exists.
            fence = _OFF_PATH * (~ok).to(w.dtype)
            ge, gmask, _ = viterbi(t, (w + fence).detach())
            s_gold = path_score(w, ge, gmask)

        return torch.relu(hamming + s_pred - s_gold)
