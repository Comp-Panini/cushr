#!/usr/bin/env python3
"""Per-candidate SEGMENTATION features for the path reranker.

Why scalars and not more embeddings: candidates in one beam are nearly the same
sentence, differing at a split point or two. A sequence model mean-pooled over
the whole path mostly sees the shared words, and identity tables (form, lemma)
memorise the training split -- measured: +0.31 / +0.43 / +0.00 dev points for
form+lemma+morph / lemma+morph / morph against a 5.56-point beam headroom.

These features are about the split itself and carry no word identity, so there
is nothing to memorise:

  n_words     over- vs under-splitting
  sum_logc    sum of log(1 + count as a GOLD word in training) -- a unigram
  mean_logc   lexicon prior, the evidence TransLIST gets from its 126K-row word
  min_logc    table, as counts instead of parameters. min = the weakest word.
  n_unseen    words never seen as a gold word in training
  n_rare      words seen fewer than `rare` times
  n_short     words of <= `short` characters (spurious fragments)
  base_rel    base score minus the beam's best (scale-free per sentence)

Counts come from the correct candidates of the TRAIN candidate lists. When
featurising a training sentence its own gold words are subtracted first
(leave-one-out); otherwise every gold word on train would look "seen", and the
model would learn a rule that never holds at test time.
"""
from collections import Counter

import numpy as np

NAMES = ("n_words", "sum_logc", "mean_logc", "min_logc",
         "n_unseen", "n_rare", "n_short", "base_rel")
N_FEAT = len(NAMES)


def form_lengths(forms):
    """Character length per raw form id, from the loaded form vocabulary."""
    return np.asarray([len(f) for f in forms], dtype=np.int64)


def load_forms(path):
    return [l.split("\t", 1)[1].rstrip("\n") if "\t" in l else ""
            for l in open(path, encoding="utf-8")]


def gold_form_counts(nodes, off, sent, label, fid):
    """Form counts over ONE correct candidate per sentence (its gold split)."""
    seen, c = set(), Counter()
    for ci in np.nonzero(label)[0]:
        s = int(sent[ci])
        if s in seen:
            continue
        seen.add(s)
        c.update(int(fid[g]) for g in nodes[off[ci]:off[ci + 1]])
    fids = np.asarray(sorted(c), dtype=np.int64)
    return fids, np.asarray([c[f] for f in fids.tolist()], dtype=np.int64)


class SegFeaturizer:
    def __init__(self, fids, counts, form_len, rare=3, short=2):
        self.counts = dict(zip(np.asarray(fids).tolist(),
                               np.asarray(counts).tolist()))
        self.form_len = np.asarray(form_len)
        self.rare, self.short = rare, short

    def feats(self, seqs, bases, fid, exclude=None):
        """seqs: list of global-node sequences; bases: base scores -> [n, F]."""
        bases = np.asarray(bases, dtype=np.float64)
        top = float(bases.max()) if len(bases) else 0.0
        out = np.zeros((len(seqs), N_FEAT), dtype=np.float32)
        for j, q in enumerate(seqs):
            f = [int(fid[g]) for g in q]
            if not f:
                out[j, 7] = bases[j] - top
                continue
            c = np.asarray([self.counts.get(x, 0)
                            - (exclude.get(x, 0) if exclude else 0) for x in f],
                           dtype=np.float64)
            c = np.maximum(c, 0)
            lc = np.log1p(c)
            out[j] = (len(f), lc.sum(), lc.mean(), lc.min(),
                      (c == 0).sum(), (c < self.rare).sum(),
                      (self.form_len[f] <= self.short).sum(), bases[j] - top)
        return out
