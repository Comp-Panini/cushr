#!/usr/bin/env python3
"""Is our benchmark the same test set TransLIST reports on?

For a long time PAPER_COMPARISON.md, RESULTS_MATRIX.md and
WEEK11_EVALUATION_MATRIX.md all said the two overlap by 97.02% (4,075 of
4,200) "after transliterating to a common scheme", and every TransLIST
comparison carried a caveat because of it. No code computed that number.

Measured here against the file TransLIST actually loads, the overlap is
4,200 / 4,200 with identical input AND output strings. The 97.02% figure was
wrong -- most likely a string match against a differently transliterated or
differently sourced copy, where 125 sentences failed to match as strings even
though they are the same sentences.

Which file TransLIST loads, from its own source:

  fastnlp-copy/core/dataset.py:798   setting 'sighum-ngram' / 'sighum-shr'
                                     -> ../LREC-Data/new_LREC_data_complete.csv
  constrained_inference.py:36        dataset 'sighum'
                                     -> LREC-Data/new_LREC_data_complete.csv

So cuSHR and TransLIST are scored on the same 4,200 sentences, and the
difference between their numbers is a difference between the systems.

Usage:
    python check_testset_overlap.py [--translist ../../translist]
"""
import argparse
import csv


def norm(s):
    """TransLIST joins words with '_', our TSV with spaces."""
    return s.replace("_", " ").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="sighum_test_4200.tsv")
    ap.add_argument("--translist-csv",
                    default="../../translist/LREC-Data/new_LREC_data_complete.csv",
                    help="the file TransLIST's sighum settings read")
    args = ap.parse_args()

    theirs = {r["DCS-ID"].strip(): r
              for r in csv.DictReader(open(args.translist_csv, encoding="utf-8"))
              if r["split"] == "test"}
    ours = {r["DCS-ID"].strip(): r
            for r in csv.DictReader(open(args.tsv, encoding="utf-8-sig"),
                                    delimiter="\t")}

    shared = set(ours) & set(theirs)
    print(f"ours   : {len(ours):,} sentences  ({args.tsv})")
    print(f"theirs : {len(theirs):,} sentences  ({args.translist_csv}, split=test)")
    print(f"shared DCS-IDs : {len(shared):,}  "
          f"({100 * len(shared) / len(ours):.2f}% of ours)")
    print(f"  ours not in theirs   : {len(set(ours) - set(theirs)):,}")
    print(f"  theirs not in ours   : {len(set(theirs) - set(ours)):,}")

    si = sum(norm(ours[i]["input"]) == norm(theirs[i]["input"]) for i in shared)
    so = sum(norm(ours[i]["output"]) == norm(theirs[i]["output"]) for i in shared)
    print(f"\nof the shared ids, identical after normalising the word separator:")
    print(f"  input  (sandhied sentence) : {si:,} / {len(shared):,}")
    print(f"  output (gold segmentation) : {so:,} / {len(shared):,}")

    if len(shared) == len(ours) == len(theirs) and si == so == len(shared):
        print("\n=> IDENTICAL test sets. TransLIST comparisons need no "
              "overlap caveat.")
    else:
        print("\n=> NOT identical -- any TransLIST comparison is across "
              "different data.")


if __name__ == "__main__":
    main()
