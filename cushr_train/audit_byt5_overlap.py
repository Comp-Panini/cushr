#!/usr/bin/env python3
"""Step 0: audits that gate the ByT5 comparison. No model, no GPU.

Three questions, answered before any inference time is spent:

  1. Is `sighum_test_4200.tsv` the same 4,200 sentences as the published
     `chronbmm/sanskrit-sandhi-split-sighum` test split? If not, the
     head-to-head framing in PAPER_COMPARISON.md §1 is wrong and has to be
     fixed before anything else.
  2. How many of those 4,200 appear in ByT5's SIGHUM FINE-TUNING split (99,900
     rows)? This is the only half of ByT5's contamination that is measurable.
  3. Restate cuSHR's own contamination status from the split file actually used
     for training, so both systems sit in one table.

WHAT THIS CANNOT MEASURE
------------------------
ByT5-Sanskrit was *pretrained* on the entire DCS, which contains the SIGHUM
sentences as raw text. Every test sentence has therefore been seen by the model
during pretraining regardless of what split 2 reports. That exposure is
unquantifiable from here and unfixable, it applies equally to their published
93.83, and it is asymmetric in their favour -- cuSHR's 4,200 are excluded from
training outright. It belongs next to the headline table, not in a footnote.
"""
import argparse
import csv
import json
import os
import sys
import unicodedata

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ingest"))
import ingest as ig  # noqa: E402

REPO = "chronbmm/sanskrit-sandhi-split-sighum"
FILES = {
    "train": "data/train-00000-of-00001-85322ccce695c64a.parquet",
    "validation": "data/validation-00000-of-00001-be4a254053c6597e.parquet",
    "test": "data/test-00000-of-00001-bffdf59021fc4727.parquet",
}


def norm(s, to_slp1=False):
    """Whitespace/unicode-normalised, optionally transliterated to SLP1.

    The two sources use different romanisations -- the HF release is IAST
    ("etac cānyac ca kauravya"), our TSV is SLP1 ("cikzepa me suto rAjan").
    Matching them raw gives 19/4,200, which looks like disjoint corpora and is
    purely an encoding artefact. `ingest.normalize_lemma` is the project's
    existing IAST->SLP1 converter; reused here rather than duplicated.

    Compared on the SANDHIED input, not the segmented output: the input is what
    the model actually consumes and is unaffected by segmentation-convention
    differences between releases.
    """
    s = " ".join(unicodedata.normalize("NFC", str(s)).split())
    if to_slp1:
        s = ig.normalize_lemma(s)
    # `normalize_lemma` drops avagraha ("'"), which is harmless for lemmas but
    # asymmetric here: our SLP1 keeps it ("so 'mutra") while the converted IAST
    # loses it ("so mutra"). Left unhandled this alone accounts for 455 spurious
    # mismatches, so strip it from BOTH sides.
    return s.replace("'", "").replace("’", "")


def load_split(name):
    path = hf_hub_download(REPO, FILES[name], repo_type="dataset")
    t = pq.read_table(path)
    cols = t.column_names
    return t.to_pylist(), cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="sighum_test_4200.tsv")
    ap.add_argument("--splits", default="splits95_ex4200.json",
                    help="the split file cuSHR actually trained with")
    ap.add_argument("--index", default="../data/sentence_index_repaired.json")
    args = ap.parse_args()

    ours = list(csv.DictReader(open(args.tsv, encoding="utf-8-sig"),
                               delimiter="\t"))
    our_in = {norm(r["input"]) for r in ours}
    our_out = {norm(r["output"]) for r in ours}
    print(f"our TSV: {len(ours):,} rows, {len(our_in):,} distinct inputs")

    # ---- 1. split identity -------------------------------------------------
    test, cols = load_split("test")
    print(f"HF test split: {len(test):,} rows, columns {cols}")
    hf_in = {norm(r["sentence"], to_slp1=True) for r in test}
    hf_out = {norm(r["unsandhied"], to_slp1=True) for r in test}

    both_in = our_in & hf_in
    print("\n=== 1. is our test set their test set? ===")
    print(f"  inputs matching   : {len(both_in):,} / {len(ours):,} "
          f"({100 * len(both_in) / len(ours):.2f}%)")
    print(f"  outputs matching  : {len(our_out & hf_out):,} / {len(ours):,}")
    print(f"  in ours, not theirs: {len(our_in - hf_in):,}")
    print(f"  in theirs, not ours: {len(hf_in - our_in):,}")
    verdict = "SAME TEST SET" if len(both_in) == len(our_in) == len(hf_in) \
        else "*** DIFFERENT -- §1 head-to-head framing is invalid ***"
    print(f"  verdict: {verdict}")

    # ---- 2. ByT5 fine-tuning contamination ---------------------------------
    train, _ = load_split("train")
    tr_in = {norm(r["sentence"], to_slp1=True) for r in train}
    leak = our_in & tr_in
    print("\n=== 2. ByT5 fine-tuning contamination ===")
    print(f"  their train rows        : {len(train):,}")
    print(f"  our 4,200 found in train: {len(leak):,} "
          f"({100 * len(leak) / len(our_in):.2f}%)")
    print("  NOTE: pretraining exposure is separate, total, and unmeasurable "
          "from here -- ByT5 was pretrained on all of DCS.")

    # ---- 3. cuSHR contamination, from the split file actually used ---------
    print("\n=== 3. cuSHR contamination ===")
    if not os.path.exists(args.splits):
        print(f"  {args.splits} not found -- skipping")
        return
    sp = json.load(open(args.splits))
    index = json.load(open(args.index))
    pos = {s: i for i, s in enumerate(index)}
    bench = {pos[str(r["DCS-ID"]).strip()] for r in ours
             if str(r["DCS-ID"]).strip() in pos}
    for k in ("train", "dev", "test"):
        if k in sp:
            n = len(bench & set(sp[k]))
            flag = "  <-- MUST BE 0" if k == "train" else ""
            print(f"  benchmark sentences in cuSHR {k:<5}: {n:,}{flag}")
    print(f"  benchmark sentences in no split      : "
          f"{len(bench - set(sp.get('train', [])) - set(sp.get('dev', [])) - set(sp.get('test', []))):,}")


if __name__ == "__main__":
    main()
