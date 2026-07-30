#!/usr/bin/env python3
"""Node featurizers with learned embedding tables.

`featurizers.py` holds featurizers whose output is fully determined by the
corpus and can therefore be precomputed into the .npz. This module holds the
other kind: a node's vector depends on trainable parameters, so it can only be
produced while the model exists.

The important structural point is that this does **not** ripple into the C++ or
CUDA side. After training, the tables are frozen, so running the featurizer once
over all nodes yields an ordinary dense ``[num_nodes, out_dim]`` float array --
exactly the artifact the existing pipeline already consumes. Embedding tables
are a training-time construct only:

    ids + scalars --(learned featurizer)--> dense [N, D] --> .npz --> C++ decode

so `export_weights.py`, the CSB3 `.bin` format, `BiaffineScorer` and the GPU
drivers all keep working unchanged, with no gather path, no new file format,
and no vocabulary-hash invariant at decode time. See `materialize()`.

## Out-of-vocabulary discipline

The failure mode for embedding tables is not accuracy on dev -- dev shares
train's vocabulary, so it looks fine -- it is that anything outside that
vocabulary collapses. Two standard mitigations, both applied here and both
load-bearing given that 7.8% of lattice candidates in this corpus are forms
unseen in training:

* **Frequency thresholding.** Forms occurring fewer than `min_count` times in
  the training split are mapped to <UNK> *in the training data itself*, so the
  <UNK> vector is trained on a realistic mixture of rare things rather than
  being a never-updated random vector.
* **Word dropout.** During training a form id is randomly replaced by <UNK>
  with probability decreasing in its frequency. This forces the model to keep
  the morphology/length/frequency pathway predictive even when identity is
  available, which is what makes out-of-domain behaviour degrade instead of
  falling off a cliff.
"""

import numpy as np
import torch
import torch.nn as nn

# Reserved ids in every remapped vocabulary.
PAD_ID = 0      # boundary nodes; embedding row is fixed at zero
UNK_ID = 1      # below-threshold or unseen at featurization time

LEARNED = {}


def register(name):
    def wrap(cls):
        cls.name = name
        LEARNED[name] = cls
        return cls
    return wrap


def get(name, **kw):
    if name not in LEARNED:
        raise KeyError(f"unknown learned featurizer {name!r}; "
                       f"available: {sorted(LEARNED)}")
    return LEARNED[name](**kw)


def remap_ids(ids, counts, min_count, n_reserved=2):
    """Map raw vocabulary ids to compact ids, folding rare entries into <UNK>.

    Returns (new_ids, size, kept) where `kept` is the number of distinct
    entries that survived thresholding. Entries at or above `min_count` are
    renumbered densely from `n_reserved`; everything else -- including anything
    with a zero training count -- becomes UNK_ID. Boundary nodes (raw id 0)
    stay PAD_ID.

    Compacting matters as well as thresholding: the raw form vocabulary has
    89k entries but most occur once or twice, so the table shrinks a long way.
    """
    counts = np.asarray(counts)
    keep = counts >= min_count
    keep[0] = False                      # raw id 0 is the reserved empty slot
    lut = np.full(len(counts), UNK_ID, dtype=np.int64)
    lut[keep] = np.arange(int(keep.sum()), dtype=np.int64) + n_reserved
    out = lut[np.clip(ids, 0, len(counts) - 1)]
    out[ids == 0] = PAD_ID
    return out.astype(np.int32), int(keep.sum()) + n_reserved, int(keep.sum())


@register("hybrid")
class HybridFeaturizer(nn.Module):
    """Precomputed scalars concatenated with learned identity embeddings.

    This is "option 3" in the design discussion: keep the whole `ngrams80`
    vector, which degrades gracefully on unseen forms, and add embedding tables
    for surface form, lemma and preverb on top, which give lexicalised
    preferences among words the training set actually contains. A projection
    mixes the two into `out_dim` columns.

    Keeping the scalars rather than replacing them is deliberate. They are the
    pathway that still works when identity is unavailable, and word dropout
    exists precisely to stop the model from ignoring them.
    """

    def __init__(self, n_scalars, n_forms, n_lemmas, n_preverbs,
                 form_dim=32, lemma_dim=16, preverb_dim=4, out_dim=96,
                 word_dropout=0.0, sparse=True):
        super().__init__()
        self.n_scalars = n_scalars
        self.out_dim = out_dim
        self.word_dropout = word_dropout
        # padding_idx pins the boundary row at zero and keeps it there: no
        # gradient flows to it, so super-source/sink stay featureless exactly as
        # they are in the precomputed featurizers.
        self.form_emb = nn.Embedding(n_forms, form_dim, padding_idx=PAD_ID,
                                     sparse=sparse)
        self.lemma_emb = nn.Embedding(n_lemmas, lemma_dim, padding_idx=PAD_ID,
                                      sparse=sparse)
        self.preverb_emb = nn.Embedding(n_preverbs, preverb_dim,
                                        padding_idx=PAD_ID, sparse=sparse)
        self.proj = nn.Linear(n_scalars + form_dim + lemma_dim + preverb_dim,
                              out_dim)

    def embedding_parameters(self):
        """Table parameters -- these need a sparse-aware optimizer."""
        return list(self.form_emb.parameters()) + \
            list(self.lemma_emb.parameters()) + \
            list(self.preverb_emb.parameters())

    def dense_parameters(self):
        return list(self.proj.parameters())

    def forward(self, scalars, ids):
        """scalars: [n, n_scalars] float32; ids: [n, 3] int64 (form, lemma, preverb)."""
        form, lemma, preverb = ids[:, 0], ids[:, 1], ids[:, 2]
        if self.training and self.word_dropout > 0:
            form = self._drop(form)
            lemma = self._drop(lemma)
        x = torch.cat([scalars,
                       self.form_emb(form),
                       self.lemma_emb(lemma),
                       self.preverb_emb(preverb)], dim=-1)
        out = self.proj(x)
        # Boundary nodes carry no signal, matching the precomputed featurizers.
        # Their scalar row is already all-zero, so detect them by that.
        pad = (form == PAD_ID)
        return out.masked_fill(pad.unsqueeze(-1), 0.0)

    def _drop(self, ids):
        """Randomly replace ids with <UNK>, leaving PAD untouched."""
        m = (torch.rand_like(ids, dtype=torch.float) < self.word_dropout) & \
            (ids != PAD_ID)
        return torch.where(m, torch.full_like(ids, UNK_ID), ids)


class LearnedBiaffine(nn.Module):
    """A learned node featurizer feeding the ordinary biaffine edge scorer.

    Shares BiaffineEdgeScorer's call signature so the training and evaluation
    loops do not need to branch on which kind of model they hold.
    """

    def __init__(self, featurizer, scorer):
        super().__init__()
        self.featurizer = featurizer
        self.scorer = scorer

    def forward(self, feats, src, dst, ids=None):
        return self.scorer.edge_scores(self.featurizer(feats, ids), src, dst)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def materialize(featurizer, scalars, ids, device, block=1 << 18, out=None):
    """Run a trained featurizer over every node, producing a dense array.

    This is what keeps embedding tables out of the decoder. Once training has
    finished the tables no longer change, so each node has one fixed vector;
    writing those vectors to a plain [N, out_dim] float32 array reduces the
    learned featurizer to exactly the artifact the precomputed path produces.
    """
    featurizer.eval()
    n = scalars.shape[0]
    if out is None:
        out = np.zeros((n, featurizer.out_dim), dtype=np.float32)
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        s = torch.as_tensor(np.asarray(scalars[lo:hi]), dtype=torch.float32,
                            device=device)
        i = torch.as_tensor(np.asarray(ids[lo:hi]), dtype=torch.int64,
                            device=device)
        out[lo:hi] = featurizer(s, i).cpu().numpy()
    featurizer.train()
    return out
