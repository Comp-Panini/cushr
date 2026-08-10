#!/usr/bin/env python3
"""Learn DCS `cng` -> UD feature bundle from ByT5's own output on TRAINING data.

The M level cannot be compared across systems as-is: the reference carries DCS
`cng` integers (`29`, `-153`) while ByT5 emits UD bundles
(`Case=Nom|Gender=Masc|Number=Sing`, 252 distinct on the test set).

WHY NOT A HAND-WRITTEN RULE TABLE
---------------------------------
SHR morph strings are structured (`pr. [4] md. pl. 3` = present, class 4,
middle, plural, 3rd) and could be parsed into bundles. But writing
`pfp. -> VerbForm=Gdv` or `iic. -> Case=Cpd` means inventing *their* spelling of
each feature, and a systematic mismatch would make ByT5's M read artificially
low with no way to detect it short of using test labels. Learning the
correspondence from ByT5's own output captures the convention it actually uses.

METHOD
------
Run ByT5 over a sample of TRAINING sentences (`byt5_infer.slurm TRAINMODE=1`),
align its output position-wise against the DCS `cng` for the same sentences,
and take the modal bundle per `cng`. No test sentence is involved; the split
file is read directly and test/dev membership is asserted absent.

Alignment is positional and only used where ByT5's word count matches the
reference, since a length mismatch means the positions do not correspond.

WHAT THE DIAGNOSTICS DECIDE
---------------------------
* purity (support-weighted) -- does a `cng` determine a bundle at all? Below
  ~95% the two tagsets do not encode the same distinctions and M is
  incommensurable, which is a finding rather than a failure.
* low-support tags -- a `cng` seen 3 times can hit 100% purity by luck.
* collisions -- distinct `cng` collapsing onto one bundle. Bundle-space scoring
  is *easier* than cng-space by exactly this much, for BOTH systems, so it has
  to be reported alongside any M number.
* failure clustering -- if the impure `cng` concentrate in one grammatical
  category, that is systematic, not noise.
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

    preds = [json.loads(l) for l in open(args.pred_jsonl, encoding="utf-8")
             if l.strip()]
    leaked = [r["id"] for r in preds if r["id"] in held_out]
    if leaked:
        raise SystemExit(f"{len(leaked)} predictions are for dev/test "
                         f"sentences: {leaked[:5]}")
    print(f"{len(preds):,} training predictions, none from dev/test")

    tally = defaultdict(Counter)
    aligned = skipped = 0
    for r in preds:
        if r["id"] not in pos:
            continue
        try:
            g = dcs_cng(args.p_dir, r["id"])
        except Exception:
            continue
        t = r.get("tags") or []
        # Positional alignment only where the lengths agree; otherwise the
        # positions do not correspond and the pairs would be noise.
        if len(t) != len(g):
            skipped += 1
            continue
        aligned += 1
        for cng, bundle in zip(g, t):
            tally[cng][bundle] += 1

    print(f"  aligned {aligned:,} sentences, skipped {skipped:,} on length")

    bridge, rows = {}, []
    tot = pure = 0
    for cng, c in tally.items():
        top, cnt = c.most_common(1)[0]
        sup = sum(c.values())
        tot += sup
        pure += cnt
        bridge[cng] = top
        rows.append((cng, sup, 100.0 * cnt / sup, top,
                     c.most_common(2)[1][0] if len(c) > 1 else ""))

    wpurity = 100.0 * pure / tot if tot else 0.0
    low = [r for r in rows if r[1] < args.min_support]
    solid = [r for r in rows if r[1] >= args.min_support]
    wp_solid = (100.0 * sum(r[1] * r[2] / 100 for r in solid)
                / sum(r[1] for r in solid)) if solid else 0.0

    # collisions: distinct cng sharing one bundle -- how much easier
    # bundle-space scoring is than cng-space, for BOTH systems
    inv = Counter(bridge.values())
    collided = sum(v for v in inv.values() if v > 1)

    print()
    print(f"cng values covered        : {len(bridge):,}")
    print(f"support-weighted purity   : {wpurity:.2f}%  "
          f"(gate {args.purity_gate}%)")
    print(f"  ... over cng with >= {args.min_support} support: {wp_solid:.2f}% "
          f"({len(solid):,} tags)")
    print(f"  low-support cng (< {args.min_support}) : {len(low):,} "
          f"-- purity there is unreliable")
    print(f"distinct bundles          : {len(inv):,}")
    print(f"cng sharing a bundle      : {collided:,} of {len(bridge):,} "
          f"({100 * collided / max(len(bridge), 1):.1f}%) "
          f"-- bundle-space M is EASIER than cng-space by this much")

    rows.sort(key=lambda r: r[2])
    print()
    print("least pure cng (worst 12):")
    print(f"  {'cng':>6} {'sup':>6} {'purity':>7}  modal -> runner-up")
    for cng, sup, pur, top, second in rows[:12]:
        print(f"  {cng:>6} {sup:>6} {pur:6.1f}%  {top[:34]:<34} | {second[:24]}")

    # failure clustering: are the impure cng concentrated in one category?
    def cat(bundle):
        for k in ("VerbForm", "Tense", "Case"):
            for part in bundle.split("|"):
                if part.startswith(k + "="):
                    return part
        return bundle or "(empty)"
    bad = Counter(cat(r[3]) for r in rows if r[2] < args.purity_gate
                  and r[1] >= args.min_support)
    allc = Counter(cat(r[3]) for r in rows if r[1] >= args.min_support)
    print()
    print("impure categories vs all (share of well-supported cng):")
    for k, v in bad.most_common(8):
        print(f"  {k:<30} impure {v:3d} / {allc[k]:3d}")

    verdict = wp_solid >= args.purity_gate
    json.dump({"_provenance": {
                   "built_from": "ByT5 output on TRAINING sentences only",
                   "pred_jsonl": os.path.basename(args.pred_jsonl),
                   "sentences_aligned": aligned,
                   "support_weighted_purity": round(wpurity, 2),
                   "purity_over_supported": round(wp_solid, 2),
                   "cng_covered": len(bridge),
                   "distinct_bundles": len(inv),
                   "cng_sharing_a_bundle": collided,
                   "usable": verdict},
               "map": bridge},
              open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, sort_keys=True)
    print()
    print(f"wrote {args.out}")
    print("VERDICT: " + ("usable -- M is reportable in bundle space"
                         if verdict else
                         "NOT usable -- cng and UD bundles do not determine "
                         "each other; keep M as n/a and report why"))


if __name__ == "__main__":
    main()
