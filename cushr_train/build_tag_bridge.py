#!/usr/bin/env python3
"""Learn ByT5's UD bundle -> DCS `cng` from its own output on TRAINING data.

The M level cannot be compared across systems as-is: the reference carries DCS
`cng` integers (`29`, `-153`) while ByT5 emits UD bundles
(`Case=Nom|Gender=Masc|Number=Sing`).

DIRECTION MATTERS, AND IT WAS MEASURED
--------------------------------------
Both directions were built and compared on the same exact-word-aligned data:

    cng -> bundle   93.18% pure over 120 cng      <- loses information
    bundle -> cng   96.69% pure over 168 bundles  <- kept

`cng -> bundle` fails because DCS's negative `cng` are *underspecified* root
analyses: one value such as -190 ("past passive participle") legitimately spans
many case/gender/number bundles. Split by sign, `cng -> bundle` is 96.80% pure
on positive `cng` but only 74.53% on negative ones. Going the other way is
many-to-one, which is well defined.

Mapping into `cng` space is also the fairer choice: the reference and cuSHR
both stay in their native representation and their published numbers remain
valid, and only ByT5 is translated. Mapping everything into bundle space would
instead have collapsed distinct `cng` together and made M easier for both.

ALIGNMENT
---------
Positional, and only where ByT5's words EXACTLY equal the gold surface forms --
not merely where the counts agree. Equal counts with different splits pair the
wrong tag with the wrong `cng`; requiring exact words lifted purity from 87.52%
to 93.18% and is what makes the correspondence measurable at all.

No test sentence is involved: the split file is read directly and dev/test
membership is asserted absent rather than assumed.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ingest"))
import ingest as ig  # noqa: E402


def dcs_cng(p_dir, stem):
    with open(os.path.join(p_dir, stem + ".p"), "rb") as f:
        d = ig._DCSUnpickler(f, encoding="utf-8").load()
    return [str(c) for cl in getattr(d, "cng", []) for c in cl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-jsonl", required=True,
                    help="ByT5 output on TRAINING sentences (TRAINMODE=1)")
    ap.add_argument("--splits", default="splits95_ex4200.json")
    ap.add_argument("--index", default="../data/sentence_index_repaired.json")
    ap.add_argument("--raw", default="../data/cushr_data_g95.npz")
    ap.add_argument("--form-vocab", default="../data/form_vocabulary.txt")
    ap.add_argument("--p-dir", default="../../SIGHUM_database_gold_path/DCS_pick")
    ap.add_argument("--out", default="tag_bridge.json")
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--purity-gate", type=float, default=95.0)
    args = ap.parse_args()

    index = json.load(open(args.index))
    pos = {s: i for i, s in enumerate(index)}
    splits = json.load(open(args.splits))
    held_out = {index[s] for s in
                set(splits.get("dev", [])) | set(splits.get("test", []))}

    z = np.load(args.raw)
    fid, cs = z["node_form_id"], z["node_char_start"]
    go = z["gold_path_offsets"].astype(np.int64)
    gn = z["gold_path_nodes"].astype(np.int64)
    fv = [l.split("\t", 1)[1].rstrip("\n") if "\t" in l else ""
          for l in open(args.form_vocab, encoding="utf-8")]

    preds = [json.loads(l) for l in open(args.pred_jsonl, encoding="utf-8")
             if l.strip()]
    leaked = [r["id"] for r in preds if r["id"] in held_out]
    if leaked:
        raise SystemExit(f"{len(leaked)} predictions are for dev/test: "
                         f"{leaked[:5]}")
    print(f"{len(preds):,} training predictions, none from dev/test")

    tally = defaultdict(Counter)
    aligned = skip_len = skip_word = 0
    for r in preds:
        s = pos.get(r["id"])
        if s is None or go[s + 1] - go[s] == 0:
            continue
        try:
            g = dcs_cng(args.p_dir, r["id"])
        except Exception:
            continue
        if len(r["tags"]) != len(g):
            skip_len += 1
            continue
        nodes = sorted(gn[go[s]:go[s + 1]].tolist(), key=lambda x: int(cs[x]))
        if [fv[fid[x]] for x in nodes] != r["words"]:
            skip_word += 1
            continue
        aligned += 1
        for cng, bundle in zip(g, r["tags"]):
            tally[bundle][cng] += 1

    print(f"  aligned {aligned:,} sentences "
          f"(skipped {skip_len:,} on length, {skip_word:,} on word mismatch)")

    bridge, rows = {}, []
    tot = pure = 0
    for bundle, c in tally.items():
        top, cnt = c.most_common(1)[0]
        sup = sum(c.values())
        tot += sup
        pure += cnt
        bridge[bundle] = top
        rows.append((bundle, sup, 100.0 * cnt / sup, top,
                     c.most_common(2)[1][0] if len(c) > 1 else ""))

    wpurity = 100.0 * pure / tot if tot else 0.0
    solid = [r for r in rows if r[1] >= args.min_support]
    low = [r for r in rows if r[1] < args.min_support]
    wp_solid = (sum(r[1] * r[2] for r in solid) / sum(r[1] for r in solid)
                if solid else 0.0)
    loose_tokens = sum(r[1] for r in rows if r[2] < 90.0)

    print()
    print(f"bundles covered           : {len(bridge):,}")
    print(f"support-weighted purity   : {wpurity:.2f}%  "
          f"(gate {args.purity_gate}%)")
    print(f"  ... over bundles with >= {args.min_support} support: "
          f"{wp_solid:.2f}% ({len(solid):,} bundles)")
    print(f"  low-support bundles (< {args.min_support}) : {len(low):,}")
    print(f"tokens in <90%-pure bundles: {loose_tokens:,} / {tot:,} "
          f"({100 * loose_tokens / max(tot, 1):.1f}%)")

    rows.sort(key=lambda r: -r[1])
    print()
    print("highest-traffic ambiguous bundles:")
    print(f"  {'sup':>5} {'purity':>7}  bundle -> modal cng | runner-up")
    for b, sup, pur, top, second in [r for r in rows if r[4]][:8]:
        print(f"  {sup:>5} {pur:6.1f}%  {(b or '(empty)')[:40]:<40} "
              f"-> {top:>5} | {second}")

    verdict = wp_solid >= args.purity_gate
    json.dump({"_provenance": {
                   "direction": "UD bundle -> DCS cng",
                   "built_from": "ByT5 output on TRAINING sentences only",
                   "pred_jsonl": os.path.basename(args.pred_jsonl),
                   "alignment": "exact word match against the gold path",
                   "sentences_aligned": aligned,
                   "support_weighted_purity": round(wpurity, 2),
                   "purity_over_supported": round(wp_solid, 2),
                   "bundles_covered": len(bridge),
                   "tokens_in_loose_bundles_pct":
                       round(100 * loose_tokens / max(tot, 1), 2),
                   "usable": verdict},
               "map": bridge},
              open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, sort_keys=True)
    print()
    print(f"wrote {args.out}")
    print("VERDICT: " + ("usable -- ByT5's tags can be scored in cng space"
                         if verdict else
                         "NOT usable -- keep M as n/a and report why"))


if __name__ == "__main__":
    main()
