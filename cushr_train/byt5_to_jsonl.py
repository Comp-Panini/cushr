#!/usr/bin/env python3
"""Convert run_inf_batch.py's TSV into the scorer's JSONL, re-attaching DCS ids.

Their output is (segmentnr, original, analyzed) and carries no sentence id, so
the only link back to a DCS-ID is position. This asserts the row count against
the manifest written by `make_byt5_input.py` rather than assuming the order
survived -- an off-by-one here would shift every prediction against its
reference and show up as a uniformly terrible model rather than as a bug.

Output is one record per sentence:
    {"id": "<DCS-ID>", "mode": "...", "words": [...], "lemmas": [...],
     "tags": [...], "raw": "<verbatim analyzed field>"}

`raw` is kept so the comparison can be re-parsed without re-running the model.

Tag handling: the model emits lemma_TAG with TAG a compressed IAST code
(`SNM`), which `sanskrit_tags.tsv` expands to a feature bundle
(`Case=Nom|Gender=Masc|Number=Sing`). That is NOT the same space as our DCS
`cng` integers, so tags are carried through verbatim and left for the scorer to
bridge -- converting here would bake in an unvalidated mapping.
"""
import argparse
import csv
import json
import os

from translit import iast_to_slp1


def parse_analyzed(text, mode):
    """-> (words, lemmas, tags), each a list, whatever the mode omits empty.

    Serialization per the paper (Fig. 1-2): words are space separated, and a
    morphosyntactic tag is suffixed to its lemma with '_'. Only the LAST
    underscore is treated as the separator, because Sanskrit lemmas can contain
    one (compound joins) while the tag codes cannot.
    """
    toks = text.split()
    words, lemmas, tags = [], [], []
    for t in toks:
        if "morphosyntax" in mode and "_" in t:
            stem, tag = t.rsplit("_", 1)
        else:
            stem, tag = t, ""
        if mode == "segmentation":
            words.append(stem)
        elif mode.startswith("segmentation-lemma"):
            words.append(stem)
            lemmas.append(stem)
        else:
            lemmas.append(stem)
        tags.append(tag)
    return words, lemmas, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-iast", action="store_true",
                    help="do not transliterate back to SLP1 (debugging)")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    ids = man["ids"]

    rows = list(csv.DictReader(open(args.tsv, encoding="utf-8"), delimiter="\t"))
    if len(rows) != len(ids):
        raise SystemExit(
            f"row count mismatch: {args.tsv} has {len(rows)} rows but the "
            f"manifest lists {len(ids)} sentences. Position is the only link "
            f"between them, so this cannot be reconciled -- re-run inference.")

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for did, r in zip(ids, rows):
            raw = (r.get("analyzed") or "").strip()
            w, l, t = parse_analyzed(raw, args.mode)
            if not args.keep_iast:
                w = [iast_to_slp1(x) for x in w]
                l = [iast_to_slp1(x) for x in l]
            f.write(json.dumps({"id": did, "mode": args.mode, "words": w,
                                "lemmas": l, "tags": t, "raw": raw},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {args.out}  ({n:,} sentences, mode={args.mode})")


if __name__ == "__main__":
    main()
