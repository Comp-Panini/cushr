#!/usr/bin/env python3
"""Per-node morphology as role counts, straight from the archive.

WHY THIS EXISTS
---------------
To test whether path-level (second-order) scoring could help, we need to count
the syntactic roles a decoded path assigns -- how many nominatives, how many
finite verbs -- and compare model paths against gold paths.

`eval_slm.py:node_cngs` already reads morphology per node, but it reads `cng`
from the graphml with `nx.read_graphml`, once per sentence. That is the right
source for *scoring* (cng is DCS-comparable) and the wrong source here:

  - it is slow enough to dominate any sweep over the corpus, and
  - cng is an opaque integer. Mapping it to case/number/person needs a table
    that does not exist; inverting `tag_bridge.json` gives only 93% purity
    overall and 74.5% on the negative cng, which is exactly the region we care
    about.

SHR's own `morph` string ('nom. sg. n.', 'impft. [2] ac. du. 1') is already
interned into `node_features` by ingest (ingest.py:709) and already parsed into
43 typed bits by `featurizers._tag_to_vec`. So role counts are a gather through
an 848x43 table -- no graphml, no cng, no lossy inversion, and identical to what
the scorer itself sees.

THE POINT OF THAT LAST CLAUSE. The trained model's features are `ngrams80` ->
`scalars64` -> `fill_morph_presence`, so columns 0-42 of every node vector are
these same bits. `nom` and `acc` are separate columns. The model is therefore
NOT blind to case; it sees each node's own case and cannot see which case it
assigned to any other word. That is the distinction this module exists to
measure.
"""
import numpy as np

from featurizers import FEATURE_TOKENS, MORPH_DIM, _tag_to_vec, load_strings

# Column index by token name, derived rather than hardcoded -- FEATURE_TOKENS
# is the single source of truth and reordering it must not silently break this.
COL = {t: i for i, t in enumerate(FEATURE_TOKENS)}

CASES = ("nom", "acc", "dat", "abl", "g", "loc", "voc", "i")
NUMBERS = ("sg", "du", "pl")
GENDERS = ("m", "f", "n", "*")
TENSES = ("pr", "pft", "impft", "fut", "aor", "opt", "imp", "per")
PERSONS = ("1", "2", "3")
PARTICIPLES = ("ppr", "pfp", "pp", "ppa")
INDECL = ("adv", "ind", "abs", "inf", "conj", "prep", "part", "ca")


class MorphTable:
    """Role counts for arbitrary sets of global node ids.

    `raw` is the ingest archive (cushr_data_g95.npz); `vocab_path` is the
    matching morph_vocabulary.txt. Only the [n_tags, 43] lookup and the
    [n_nodes] id column are held -- the expanded [n_nodes, 43] matrix is never
    materialized, since callers only ever ask about the ~7 nodes on a path.
    """

    def __init__(self, raw, vocab_path):
        self.tags = load_strings(vocab_path)
        self.morph_ids = np.asarray(raw["node_features"]).ravel()
        n = max(len(self.tags), int(self.morph_ids.max()) + 1)
        self.lookup = np.zeros((n, MORPH_DIM), dtype=np.float32)
        for tag_id, tag in enumerate(self.tags):
            if tag:
                self.lookup[tag_id] = _tag_to_vec(tag)

    def tag(self, node):
        """SHR morph string for one global node id ('' if out of vocabulary)."""
        i = int(self.morph_ids[node])
        return self.tags[i] if i < len(self.tags) else ""

    def vecs(self, nodes):
        """[len(nodes), 43] presence bits."""
        return self.lookup[self.morph_ids[np.asarray(nodes, dtype=np.int64)]]

    # -- role predicates -----------------------------------------------------
    #
    # `finite` is the one that needs justifying. SHR writes a finite verb as
    # tense + voice + number + person ('impft. [2] ac. du. 1') and a participle
    # as a participle token plus nominal agreement ('pp. nom. sg. n.'). Both
    # carry a tense-family bit, so tense alone does not separate them. A person
    # bit is present on finite forms and absent on participles, which is the
    # discriminator used here; the participle exclusion is belt-and-braces.

    def is_finite(self, v):
        return bool(v[[COL[p] for p in PERSONS]].any()
                    and not v[[COL[p] for p in PARTICIPLES]].any())

    def is_participle(self, v):
        return bool(v[[COL[p] for p in PARTICIPLES]].any())

    def case_of(self, v):
        """The single case token set, or '' -- indeclinables and verbs have none."""
        for c in CASES:
            if v[COL[c]]:
                return c
        return ""

    def number_of(self, v):
        for n in NUMBERS:
            if v[COL[n]]:
                return n
        return ""

    def counts(self, nodes):
        """Role counts for a whole path. `nodes` are global node ids."""
        vs = self.vecs(nodes)
        out = {f"n_{c}": int(vs[:, COL[c]].sum()) for c in CASES}
        out.update({f"n_{n}": int(vs[:, COL[n]].sum()) for n in NUMBERS})
        out["n_finite"] = sum(self.is_finite(v) for v in vs)
        out["n_part"] = sum(self.is_participle(v) for v in vs)
        out["n_indecl"] = int(vs[:, [COL[t] for t in INDECL]].any(1).sum())
        out["n_iic"] = int(vs[:, COL["iic"]].sum())
        out["n_words"] = len(vs)
        return out

    def agrees(self, nodes):
        """True if some (nominative, finite verb) pair agrees in number.

        The one genuinely second-order property here: it is a predicate over a
        PAIR of words, so no per-node feature can express it however rich the
        node vector gets.
        """
        vs = self.vecs(nodes)
        noms = [self.number_of(v) for v in vs if v[COL["nom"]]]
        vrbs = [self.number_of(v) for v in vs if self.is_finite(v)]
        return any(n and n == w for n in noms for w in vrbs)
