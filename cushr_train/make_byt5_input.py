#!/usr/bin/env python3
"""Write the SIGHUM test sentences as IAST .txt for ByT5-Sanskrit inference.

`run_inf_batch.py` takes an --input-folder of .txt files, one sentence per line,
and writes a sibling .tsv whose rows are (segmentnr, original, analyzed). It
carries no sentence ids, so **line order is the only link back to a DCS-ID** --
this script pins that order and writes it out as a manifest, rather than trusting
that the TSV comes back in the order it went out.

Input is the sandhied `input` column, transliterated SLP1 -> IAST to match the
model's pretraining. The transliteration is asserted lossless first: a silent
bug there is indistinguishable from a model difference.
"""
import argparse
import csv
import json
import os
import random

from translit import assert_roundtrip, slp1_to_iast


def _rows_from_split(args):
    """Training sentences as ({DCS-ID}, sandhied_text) pairs, SLP1.

    The sandhied text is reconstructed from the archive's `surface_text` rather
    than the TSV, because the TSV only covers the 4,200 test sentences. Only
    sentences with a resolved gold path are emitted -- a sentence with no gold
    contributes nothing to the bridge, since there is no cng to align against.
    """
    import numpy as np

    index = json.load(open(args.index))
    splits = json.load(open(args.from_split))
    ids = sorted(set(splits[args.split]))
    forbidden = set()
    for k in ("dev", "test"):
        if k != args.split:
            forbidden |= set(splits.get(k, []))

    z = np.load(args.raw)
    txt = z["surface_text"]
    off = z["surface_text_offsets"].astype(np.int64)
    go = z["gold_path_offsets"].astype(np.int64)

    ids = [s for s in ids if go[s + 1] - go[s] > 0]
    random.seed(args.seed)
    ids = random.sample(ids, min(args.sample, len(ids)))

    leaked = [s for s in ids if s in forbidden]
    if leaked:
        raise SystemExit(f"{len(leaked)} sampled ids are in a held-out split: "
                         f"{leaked[:5]}")

    # Drop sentences that do not survive SLP1 -> IAST -> SLP1.
    #
    # IAST `th` is ambiguous: it spells both aspirated t (SLP1 `T`) and t
    # followed by h (SLP1 `th`), so `abravItha` comes back as `abravITa`. The
    # affected sentences also tend to carry `?` replacement characters from the
    # source encoding. 1.18% of training sentences, and 0% of the 4,200 test
    # sentences -- so the already-collected test predictions are unaffected and
    # this only trims the bridge's training sample.
    from translit import iast_to_slp1

    rows, src, dropped = [], [], 0
    for s in ids:
        sent = bytes(txt[off[s]:off[s + 1]]).decode("utf-8", "replace")
        if not sent.strip():
            continue
        if iast_to_slp1(slp1_to_iast(sent)) != sent:
            dropped += 1
            continue
        rows.append({"DCS-ID": index[s]})
        src.append(sent)
    print(f"--from-split {args.from_split} [{args.split}]: "
          f"{len(rows):,} sentences, none from dev/test")
    print(f"  dropped {dropped} ({100 * dropped / max(len(ids), 1):.2f}%) "
          f"that do not round-trip through IAST")
    return rows, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="sighum_test_4200.tsv")
    ap.add_argument("--out-dir", default="byt5_in",
                    help="folder handed to run_inf_batch.py --input-folder")
    ap.add_argument("--stem", default="sighum_test",
                    help="basename; output lands at <stem>.txt / <stem>.tsv")
    ap.add_argument("--limit", type=int, default=0,
                    help="self-test: only the first N sentences")
    # --- training-sentence mode, for building the cng <-> tag bridge ---------
    ap.add_argument("--from-split", default="",
                    help="splits json; emit sentences from --split instead of "
                         "the test TSV. Used to learn ByT5's tag convention "
                         "from TRAINING data, so the bridge touches no test "
                         "sentence.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", default="../data/sentence_index_repaired.json")
    ap.add_argument("--raw", default="../data/cushr_data_g95.npz")
    args = ap.parse_args()

    if args.from_split:
        rows, src = _rows_from_split(args)
    else:
        rows = list(csv.DictReader(open(args.tsv, encoding="utf-8-sig"),
                                   delimiter="\t"))
        if args.limit:
            rows = rows[:args.limit]
        src = [r["input"] for r in rows]
    assert_roundtrip(src, "sighum input column")

    os.makedirs(args.out_dir, exist_ok=True)
    txt = os.path.join(args.out_dir, args.stem + ".txt")
    with open(txt, "w", encoding="utf-8") as f:
        for s in src:
            # One sentence per line and no blank lines: a dropped or added line
            # shifts every id after it, and the manifest would not detect it.
            line = slp1_to_iast(s).replace("\n", " ").strip()
            if not line:
                raise SystemExit("empty input line would desynchronise the manifest")
            f.write(line + "\n")

    manifest = os.path.join(args.out_dir, args.stem + ".manifest.json")
    json.dump({"stem": args.stem,
               "ids": [str(r["DCS-ID"]).strip() for r in rows],
               "n": len(rows)},
              open(manifest, "w"))

    print(f"wrote {txt}  ({len(src):,} sentences, IAST)")
    print(f"wrote {manifest}")
    print(f"  first: {slp1_to_iast(src[0])}")


if __name__ == "__main__":
    main()
