#!/usr/bin/env python3
"""Named, swappable node featurizers.

A featurizer turns the raw per-node columns that ingest.py emits (morph tag id,
surface form id, character span, ...) into the dense ``[num_nodes, OUT_DIM]``
float matrix that the biaffine scorer consumes. Which featurizer ran is a
property of the data file, not of the training code, so comparing two
featurizations is a flag rather than a refactor.

Every featurizer emits exactly OUT_DIM = 64 columns, zero-padding if it has
less to say. Fixing the width keeps featurized archives interchangeable and
gives a future fused scoring kernel a single shape to specialise for. The cost
is some always-zero columns in the smaller featurizers, which the biaffine
simply learns zero weights for.

Two invariants every featurizer must hold:

* **Boundary nodes are zero.** Super-source and super-sink carry no surface
  realisation (ingest marks them position/chunk -1, vocabulary id 0). Their
  feature rows are all-zero and they never contribute to a corpus statistic.
* **fit() sees the training split only.** Corpus frequency is the obvious trap:
  counting over all sentences leaks test-set information into a feature and
  inflates dev numbers for free. fit() therefore takes explicit train sentence
  ids and the caller is responsible for passing the right ones.
"""

import os

import numpy as np

OUT_DIM = 64

FEATURIZERS = {}


def register(name):
    """Class decorator adding a featurizer to the registry under `name`."""
    def wrap(cls):
        if name in FEATURIZERS:
            raise ValueError(f"featurizer {name!r} already registered")
        cls.name = name
        FEATURIZERS[name] = cls
        return cls
    return wrap


def get(name, vocab_dir="../ingest"):
    """Instantiate a registered featurizer.

    `vocab_dir` is where the interned vocabularies written by ingest live. It
    must be the directory produced by the same ingest run as the archive being
    featurized -- ids are positional, so mixing a vocabulary with an archive
    from a different run silently mislabels every node.
    """
    if name not in FEATURIZERS:
        raise KeyError(
            f"unknown featurizer {name!r}; available: {sorted(FEATURIZERS)}")
    return FEATURIZERS[name](vocab_dir)


class Featurizer:
    """Base class. Subclasses implement `_transform`; padding is handled here."""

    out_dim = OUT_DIM
    name = "base"

    def __init__(self, vocab_dir="../ingest"):
        self.vocab_dir = vocab_dir

    def vocab(self, filename):
        return os.path.join(self.vocab_dir, filename)

    def fit(self, raw, train_nodes):
        """Gather corpus statistics. `train_nodes` is a boolean mask over nodes
        selecting those belonging to training-split sentences."""
        return self

    def transform(self, raw, out=None):
        """Featurize into `out` ([n, OUT_DIM] float32), allocating if needed.

        Subclasses write columns into the buffer rather than returning a matrix
        that then has to be padded and concatenated. At corpus scale that
        matters: building the vector by concatenation peaked over 3 GB (a
        772 MB morph gather, a 1.15 GB concatenate, then a 1.15 GB padding
        copy), which does not fit alongside everything else. Filling in place
        keeps the only large allocation the output itself -- and `out` may be a
        memmap, in which case even that lives on disk.
        """
        n = len(raw["node_features"])
        if out is None:
            out = np.zeros((n, OUT_DIM), dtype=np.float32)
        if out.shape != (n, OUT_DIM):
            raise ValueError(
                f"{self.name}: out has shape {out.shape}, expected {(n, OUT_DIM)}")
        out[:] = 0.0
        self._fill(raw, out)
        # Boundary nodes must not carry signal regardless of what the subclass
        # computed for them.
        out[boundary_mask(raw)] = 0.0
        return out

    def _fill(self, raw, out):
        raise NotImplementedError


# Nodes per block when gathering wide per-node tables. 1M x 43 float32 is
# ~170 MB, which is the largest temporary the fill path allocates.
BLOCK = 1 << 20


def node_blocks(n, block=BLOCK):
    for lo in range(0, n, block):
        yield lo, min(lo + block, n)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def boundary_mask(raw):
    """True for super-source / super-sink nodes."""
    return raw["node_chunk"] < 0


def fill_morph_onehots(raw, vocab_path, out):
    """Write 43 presence features per node into out[:, :43].

    Expanded from the interned morph tag, blockwise so the gathered table never
    exists whole.

    Moved verbatim from the retired cushr_cpu/fix_npz.py. One-hot over all 847
    tags would be 14 GB uncompressed and unreadable by cnpy (which cannot read
    compressed archives); parsing each SHR tag into case/number/gender/tense/
    voice/person/word-class bits keeps it small and gives the scorer weights
    with linguistic meaning rather than treating tag ids as a linear scale.
    """
    id_to_tag = {}
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tag_id, tag = line.split("\t", 1)
            id_to_tag[int(tag_id)] = tag

    morph_ids = np.asarray(raw["node_features"]).ravel()
    max_id = max(max(id_to_tag), int(morph_ids.max()))
    lookup = np.zeros((max_id + 1, MORPH_DIM), dtype=np.float32)
    for tag_id, tag in id_to_tag.items():
        lookup[tag_id] = _tag_to_vec(tag)
    for lo, hi in node_blocks(len(morph_ids)):
        out[lo:hi, :MORPH_DIM] = lookup[morph_ids[lo:hi]]


# SHR abbreviations: g. = genitive, i. = instrumental, pr. = present, etc.
FEATURE_TOKENS = [
    # Nominal case (8)
    'nom', 'acc', 'dat', 'abl', 'g', 'loc', 'voc', 'i',
    # Grammatical number (3)
    'sg', 'du', 'pl',
    # Gender (4;  '*' = gender-indeterminate form)
    'm', 'f', 'n', '*',
    # Tense / mood (8)
    'pr', 'pft', 'impft', 'fut', 'aor', 'opt', 'imp', 'per',
    # Voice (3)
    'ac', 'md', 'ps',
    # Person (3)
    '1', '2', '3',
    # Participial and verbal-noun forms (4)
    'ppr', 'pfp', 'pp', 'ppa',
    # Indeclinables and other word classes (9)
    'adv', 'ind', 'abs', 'inf', 'iic', 'conj', 'prep', 'part', 'ca',
]
MORPH_DIM = len(FEATURE_TOKENS) + 1     # +1 for the explicit UNKNOWN flag


def _tag_to_vec(tag_str):
    tokens = set(tag_str.replace('.', ' ').split())
    vec = [1.0 if t in tokens else 0.0 for t in FEATURE_TOKENS]
    vec.append(1.0 if tag_str == 'UNKNOWN' else 0.0)
    return vec


def load_strings(path):
    """Read an `id\\tstring` vocabulary into a list indexed by id."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            token_id, token = line.split("\t", 1) if "\t" in line else (line, "")
            pairs.append((int(token_id), token))
    out = [""] * (max(p[0] for p in pairs) + 1)
    for token_id, token in pairs:
        out[token_id] = token
    return out


# ---------------------------------------------------------------------------
# featurizers
# ---------------------------------------------------------------------------

@register("morph43")
class Morph43(Featurizer):
    """Morphology + length only -- the pre-registry feature vector.

    43 morph presence bits plus log1p(word length), zero-padded to 64. This is
    the regression baseline: trained under this featurizer the model should
    reproduce the dev F1 of the old fix_npz.py pipeline, which makes every
    richer featurizer a clean ablation against it rather than a separate
    pipeline whose differences are hard to attribute.
    """

    def _fill(self, raw, out):
        fill_morph_onehots(raw, self.vocab("morph_vocabulary.txt"), out)
        length = np.asarray(raw["node_word_length"], dtype=np.float32)
        out[:, MORPH_DIM] = np.log1p(length)


@register("scalars64")
class Scalars64(Featurizer):
    """morph43 plus 20 surface-derived scalar features.

    Layout (64 columns):
        0-42   morph presence bits
        43     log1p(word length)
        44-47  length extras: scaled length, short/medium/long bucket
        48-51  position: normalised start, normalised end, distance from chunk
               end, normalised chunk index
        52-55  frequency: log1p form count, log1p lemma count, hapax flag,
               frequency-rank bucket
        56-63  character class: vowel rate, cluster rate, retroflex rate,
               visarga, anusvara, ends-in-vowel, starts-with-vowel,
               final-long-vowel

    The frequency block is the important one and the one with the sharp edge:
    it is a surrogate for form identity that costs no learned parameters and
    degrades gracefully on unseen forms (they simply score as rare), but it is
    only honest if the counts come from the training split alone.
    """

    def __init__(self, vocab_dir="../ingest"):
        super().__init__(vocab_dir)
        self.form_counts = None
        self.lemma_counts = None
        self.form_rank = None

    def fit(self, raw, train_nodes):
        form_ids = np.asarray(raw["node_form_id"], dtype=np.int64)
        lemma_ids = np.asarray(raw["node_lemma_id"], dtype=np.int64)
        keep = train_nodes & ~boundary_mask(raw)

        n_forms = len(load_strings(self.vocab("form_vocabulary.txt")))
        n_lemmas = len(load_strings(self.vocab("lemma_vocabulary.txt")))
        self.form_counts = np.bincount(form_ids[keep], minlength=n_forms).astype(np.float32)
        self.lemma_counts = np.bincount(lemma_ids[keep], minlength=n_lemmas).astype(np.float32)

        # Rank forms by frequency, then bucket the rank. The raw count is
        # heavy-tailed and its scale shifts with corpus size; the rank bucket
        # gives the model a stable "how common relative to everything else"
        # signal that survives a change of training-set size.
        order = np.argsort(-self.form_counts, kind="stable")
        self.form_rank = np.empty(n_forms, dtype=np.float32)
        self.form_rank[order] = np.arange(n_forms, dtype=np.float32)
        if n_forms > 1:
            self.form_rank /= (n_forms - 1)
        return self

    def _fill(self, raw, out):
        if self.form_counts is None:
            raise RuntimeError("scalars64.fit() must run before transform()")

        n = len(raw["node_features"])
        fill_morph_onehots(raw, self.vocab("morph_vocabulary.txt"), out)
        length = np.asarray(raw["node_word_length"], dtype=np.float32)
        out[:, 43] = np.log1p(length)

        # -- length extras (44-47) --------------------------------------------
        # Word length is the single strongest surface prior in Sanskrit -- the
        # hand-tuned LengthScorer is nothing but log(1+len) -- so the learned
        # model gets the same signal in a form it can shape non-monotonically.
        out[:, 44] = np.clip(length / 20.0, 0.0, 1.0)
        out[:, 45] = length <= 3
        out[:, 46] = (length > 3) & (length <= 8)
        out[:, 47] = length > 8

        # -- position (48-51) --------------------------------------------------
        self._fill_position(raw, length, n, out)

        # -- frequency (52-55) -------------------------------------------------
        form_ids = np.asarray(raw["node_form_id"], dtype=np.int64)
        lemma_ids = np.asarray(raw["node_lemma_id"], dtype=np.int64)
        fc = self.form_counts[np.clip(form_ids, 0, len(self.form_counts) - 1)]
        lc = self.lemma_counts[np.clip(lemma_ids, 0, len(self.lemma_counts) - 1)]
        out[:, 52] = np.log1p(fc) / 10.0
        out[:, 53] = np.log1p(lc) / 10.0
        out[:, 54] = fc <= 1               # hapax / unseen-in-train
        out[:, 55] = self.form_rank[np.clip(form_ids, 0, len(self.form_rank) - 1)]
        del fc, lc, lemma_ids

        # -- character class (56-63) -------------------------------------------
        self._fill_charclass(raw, form_ids, out)

    def _fill_position(self, raw, length, n, out):
        """Where the candidate sits, in its chunk and in the sentence.

        Some phenomena are genuinely position-sensitive (verse-final verbs,
        particle placement), and chunk-relative position also encodes whether a
        candidate is compound-initial or compound-final.
        """
        position = np.asarray(raw["node_position"], dtype=np.float32)
        char_start = np.asarray(raw["node_char_start"], dtype=np.float32)
        chunk = np.asarray(raw["node_chunk"], dtype=np.int64)
        sent_off = np.asarray(raw["sentence_offsets"], dtype=np.int64)
        text_off = np.asarray(raw["surface_text_offsets"], dtype=np.int64)

        # Sentence length in characters, broadcast to each of its nodes.
        sent_id = np.searchsorted(sent_off, np.arange(n), side="right") - 1
        sent_id = np.clip(sent_id, 0, len(text_off) - 2)
        sent_len = (text_off[sent_id + 1] - text_off[sent_id]).astype(np.float32)
        sent_len = np.maximum(sent_len, 1.0)

        # Chunk width: the furthest any candidate in that chunk reaches.
        chunk_key = sent_id * (chunk.max() + 2) + np.maximum(chunk, 0)
        reach = position + length
        chunk_width = np.zeros(int(chunk_key.max()) + 1, dtype=np.float32)
        np.maximum.at(chunk_width, chunk_key, reach)
        width = np.maximum(chunk_width[chunk_key], 1.0)

        start = np.where(char_start >= 0, char_start, 0.0)
        out[:, 48] = np.clip(start / sent_len, 0.0, 1.0)
        out[:, 49] = np.clip((start + length) / sent_len, 0.0, 1.0)
        out[:, 50] = np.clip((width - reach) / width, 0.0, 1.0)
        out[:, 51] = np.clip(np.maximum(chunk, 0) / 20.0, 0.0, 1.0)

    def _fill_charclass(self, raw, form_ids, out):
        """Phonotactic summary of the surface form, into out[:, 56:64].

        Sandhi is conditioned on the phonemes at a word junction, so what a
        form ends in (vowel vs consonant, long vs short, visarga, anusvara)
        carries real information about which junctions are plausible. Computed
        per vocabulary entry, then gathered -- forms repeat heavily, so this is
        far cheaper than working per node.
        """
        forms = load_strings(self.vocab("form_vocabulary.txt"))
        table = np.zeros((len(forms), 8), dtype=np.float32)
        for i, w in enumerate(forms):
            if w:
                table[i] = _charclass(w)
        ids = np.clip(form_ids, 0, len(forms) - 1)
        for lo, hi in node_blocks(len(ids)):
            out[lo:hi, 56:64] = table[ids[lo:hi]]


# SLP1 inventories.
VOWELS = set("aAiIuUfFxXeEoO")
LONG_VOWELS = set("AIUFXeEoO")
RETROFLEX = set("wWqQRzfF")


def _charclass(w):
    n = len(w)
    letters = [c for c in w]
    n_vowel = sum(1 for c in letters if c in VOWELS)
    clusters = sum(1 for a, b in zip(letters, letters[1:])
                   if a not in VOWELS and b not in VOWELS)
    last = w[-1]
    first = w[0]
    return [
        n_vowel / n,
        clusters / max(n - 1, 1),
        sum(1 for c in letters if c in RETROFLEX) / n,
        1.0 if "H" in w else 0.0,          # visarga
        1.0 if "M" in w else 0.0,          # anusvara
        1.0 if last in VOWELS else 0.0,
        1.0 if first in VOWELS else 0.0,
        1.0 if last in LONG_VOWELS else 0.0,
    ]


@register("ngram_split")
class NgramSplit(Featurizer):
    """Hellwig & Nehrdich (2018) n-gram split probabilities. NOT IMPLEMENTED.

    The paper computes, for each *character position*, how often a split occurs
    after each left n-gram and before each right n-gram for n in [2, 7],
    normalised by the most frequent n-gram of that length -- 12 values per
    position. Our lattice is node-factored rather than character-factored, so
    the port is to evaluate those 12 values at each node's start boundary and
    again at its end boundary, giving 24 columns: "is this candidate's left
    edge a likely split point, and its right edge".

    Everything this needs is now in the archive: `surface_text` /
    `surface_text_offsets` give the raw string, `node_char_start` and
    `node_length` give each node's boundaries, and the gold path identifies
    which boundaries are true splits for counting. Counting must run over the
    training split only, in fit(), for the same reason the frequency block does.

    Unimplemented deliberately -- registered so the shape of the work is
    visible and so `--featurizer ngram_split` fails loudly rather than
    silently falling back to something else.
    """

    def fit(self, raw, train_nodes):
        raise NotImplementedError(
            "ngram_split is a design placeholder; see the class docstring for "
            "the intended construction.")

    def _transform(self, raw):
        raise NotImplementedError
