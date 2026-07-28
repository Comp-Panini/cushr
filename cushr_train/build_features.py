#!/usr/bin/env python3
"""Turn the raw ingest archive into a featurized one, using a named featurizer.

Replaces the old cushr_cpu/fix_npz.py, which hardcoded a single featurization
(morph one-hots) and a single output name. Here the featurization is selected
by `--featurizer`, and both the name and the resulting width travel with the
data so training, export, and the C++ decoder can read them instead of
agreeing on a constant.

    python build_features.py --featurizer scalars64 \
        --raw ../data/new_cushr_data_full_with_gold.npz \
        --out ../data/features_scalars64.npz

Pipeline position:

    ingest.py  ->  raw .npz  ->  build_features.py  ->  featurized .npz
                                                     ->  prepare.py  -> cache/
                                                     ->  train.py

Corpus statistics are fitted on the training split only. The split is a
deterministic md5 bucketing of the sentence id (prepare.bucket), so this script
reproduces exactly the split prepare.py will later write to splits.json without
needing it to have run first -- which resolves the ordering problem, since
prepare.py consumes the file this script produces.
"""

import argparse
import os

import numpy as np

import featurizers
from prepare import bucket

# raw ingest name -> the name the rest of the pipeline uses
RENAME = {
    "node_length": "node_word_length",
    "rowptr": "row_ptr",
    "colidx": "col_idx",
    "topolevel": "topo_level",
    "goldpathmask": "gold_path_mask",
}

# Raw columns the featurizers read. Passed through to the output too, so a
# later featurizer can be re-run from the featurized archive alone.
RAW_PASSTHROUGH = [
    "node_position", "node_chunk", "node_form_id", "node_lemma_id",
    "node_preverb_id", "node_char_start",
    "surface_text", "surface_text_offsets",
]


def load_raw(path):
    """Read the ingest archive into a dict under post-rename names."""
    z = np.load(path)
    raw = {}
    for key in z.files:
        raw[RENAME.get(key, key)] = z[key]

    # ingest writes sentenceoffsets without the trailing total; the rest of the
    # pipeline expects a closed CSR-style offset array.
    if "sentence_offsets" not in raw:
        n_nodes = len(raw["node_features"])
        raw["sentence_offsets"] = np.append(
            raw["sentenceoffsets"], n_nodes).astype(np.int32)
    missing = [k for k in ("node_position", "node_chunk", "node_form_id")
               if k not in raw]
    if missing:
        raise SystemExit(
            f"{path} predates the raw-field ingest (missing {missing}).\n"
            "Re-run ingest/ingest.py to regenerate it.")
    return raw


def train_node_mask(raw, dev_pct, test_pct):
    """Boolean mask over nodes selecting training-split sentences.

    Sentences with no resolved gold path belong to no split and are excluded
    outright -- they are never trained on, so letting them contribute to
    frequency counts would be counting text the model never sees supervised.
    """
    sent_off = np.asarray(raw["sentence_offsets"], dtype=np.int64)
    gold_off = np.asarray(raw["gold_path_offsets"], dtype=np.int64)
    n_sent = len(sent_off) - 1
    has_gold = np.diff(gold_off) > 0

    mask = np.zeros(len(raw["node_features"]), dtype=bool)
    n_train = 0
    for s in range(n_sent):
        if not has_gold[s]:
            continue
        if bucket(s) < dev_pct + test_pct:      # dev or test
            continue
        mask[sent_off[s]:sent_off[s + 1]] = True
        n_train += 1
    return mask, n_train


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--featurizer", default="scalars64",
                    choices=sorted(featurizers.FEATURIZERS))
    ap.add_argument("--raw", default="../data/new_cushr_data_full_with_gold.npz")
    ap.add_argument("--out", default=None,
                    help="default: ../data/features_<featurizer>.npz")
    ap.add_argument("--vocab-dir", default=None,
                    help="directory holding the vocabularies ingest wrote; "
                         "defaults to the directory of --raw")
    ap.add_argument("--dev-pct", type=int, default=5)
    ap.add_argument("--test-pct", type=int, default=5)
    args = ap.parse_args()

    # Vocabularies are written beside their archive, so that is the right
    # default: it makes a mismatched vocabulary/archive pairing impossible
    # unless someone opts into it explicitly.
    vocab_dir = args.vocab_dir or os.path.dirname(os.path.abspath(args.raw))

    out = args.out or f"../data/features_{args.featurizer}.npz"

    print(f"loading {args.raw} ...")
    raw = load_raw(args.raw)
    n_nodes = len(raw["node_features"])
    print(f"  {n_nodes:,} nodes")

    mask, n_train = train_node_mask(raw, args.dev_pct, args.test_pct)
    print(f"  train split: {n_train:,} sentences, {int(mask.sum()):,} nodes")

    fz = featurizers.get(args.featurizer, vocab_dir)
    print(f"fitting featurizer {args.featurizer!r} on the training split ...")
    fz.fit(raw, mask)

    # The feature matrix is ~1.1 GB at corpus scale, so build it in a memmap
    # rather than in RAM. np.savez streams it into the archive in chunks, so it
    # never has to be resident even at write time.
    scratch = out + ".features.tmp.npy"
    print(f"transforming {n_nodes:,} nodes -> ({n_nodes:,}, {fz.out_dim}) "
          f"via {scratch} ...")
    feats = np.lib.format.open_memmap(
        scratch, mode="w+", dtype=np.float32, shape=(n_nodes, fz.out_dim))
    fz.transform(raw, out=feats)
    feats.flush()

    nonzero = np.zeros(fz.out_dim, dtype=bool)
    for lo, hi in featurizers.node_blocks(n_nodes):
        nonzero |= np.abs(feats[lo:hi]).sum(axis=0) > 0
    nonzero_cols = int(nonzero.sum())
    print(f"  {nonzero_cols}/{fz.out_dim} columns carry signal "
          f"({fz.out_dim - nonzero_cols} zero-padding)")

    save = dict(
        node_features=feats,
        node_word_length=np.asarray(raw["node_word_length"], dtype=np.int32),
        row_ptr=np.asarray(raw["row_ptr"], dtype=np.int32),
        col_idx=np.asarray(raw["col_idx"], dtype=np.int32),
        topo_level=np.asarray(raw["topo_level"], dtype=np.int32),
        sentence_offsets=np.asarray(raw["sentence_offsets"], dtype=np.int32),
        gold_path_mask=np.asarray(raw["gold_path_mask"], dtype=np.int8),
        gold_path_nodes=np.asarray(raw["gold_path_nodes"], dtype=np.int32),
        gold_path_offsets=np.asarray(raw["gold_path_offsets"], dtype=np.int32),
        # provenance: which featurizer built this and how wide it is
        feat_dim=np.int32(fz.out_dim),
        featurizer_name=np.array(args.featurizer),
    )
    for key in RAW_PASSTHROUGH:
        if key in raw:
            save[key] = raw[key]

    print(f"saving {out} ...")
    # Uncompressed: cnpy, the C++ .npz reader, cannot read compressed archives.
    np.savez(out, **save)
    del feats, save
    os.remove(scratch)
    print(f"done  feat_dim={fz.out_dim}  "
          f"file size: {os.path.getsize(out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
