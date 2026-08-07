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
    """-> (words, lemmas, tags), parsed from the model's real output format.

    OBSERVED FORMAT (job 3347274, mode segmentation-lemma-morphosyntax), which
    is NOT what the paper's Figure 1-2 suggested:

        cikṣepa_kṣip_Tense=Past|Mood=Ind|Person=3|Number=Sing
        atha_atha_                       <- indeclinable: empty feature field
        saubha-_saubha_Case=Cpd          <- compound: trailing hyphen on form

    Three underscore-separated fields, `form_lemma_features` -- not
    `lemma_TAG`. An earlier version of this function did `rsplit("_", 1)` and
    would have silently turned `cikṣepa_kṣip_Tense=...` into
    form=`cikṣepa_kṣip`, tag=`Tense=...`, mangling every token.

    Two further points the real output settled:

    * Features arrive as FULL UD bundles (`Case=Nom|Gender=Masc|Number=Sing`),
      not the compressed IAST codes (`SNM`) of the paper's figures -- so
      `data/sanskrit_tags.tsv` is not needed to read this output at all.
    * `split("_", 2)` (maxsplit) is correct rather than `rsplit`: the feature
      bundle uses `|` and `=` internally and never `_`, while splitting from
      the right would break the moment a lemma contained one.

    The trailing `-` on a compound member is stripped, because the reference
    (`sighum_test_4200.tsv`) writes `sOBa`, not `sOBa-`. That is a deliberate
    normalisation, not incidental cleanup.
    """
    words, lemmas, tags = [], [], []
    for t in text.split():
        parts = t.split("_", 2)
        if len(parts) == 3:
            form, lemma, feat = parts
        elif len(parts) == 2:
            form, lemma, feat = parts[0], parts[1], ""
        else:
            # `segmentation` mode emits bare word forms with no underscores.
            form, lemma, feat = parts[0], "", ""
        form = form.rstrip("-")
        if form:
            words.append(form)
        lemmas.append(lemma)
        tags.append(feat)
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
