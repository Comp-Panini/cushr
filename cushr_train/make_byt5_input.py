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

from translit import assert_roundtrip, slp1_to_iast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="sighum_test_4200.tsv")
    ap.add_argument("--out-dir", default="byt5_in",
                    help="folder handed to run_inf_batch.py --input-folder")
    ap.add_argument("--stem", default="sighum_test",
                    help="basename; output lands at <stem>.txt / <stem>.tsv")
    ap.add_argument("--limit", type=int, default=0,
                    help="self-test: only the first N sentences")
    args = ap.parse_args()

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
