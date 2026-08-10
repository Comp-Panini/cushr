#!/usr/bin/env python3
"""Falsifiers F1/F2/F2b: could a clause-level constraint fix the case errors?

Before building a path-level reranker we need evidence that the errors it would
target are *of the kind a path-level model can see*. This script produces that
evidence, and is designed so that a negative result is as reportable as a
positive one.

WHAT IS COMPARED, AND WHY IT IS THE HONEST COMPARISON
-----------------------------------------------------
The model's decoded path against OUR OWN GOLD PATH -- never against DCS. The
gold path is exactly what the model was trained to reproduce, so every
divergence is the scorer's fault. Scoring against DCS instead would mix in the
SHR/DCS convention gap, which the convention tables already showed is a separate
and independent deficit.

Morphology comes from SHR's `morph` string via `morph_table`, not from `cng`.
Those are different label spaces (288 morph strings collapse onto 168 cng), so
the error count here need not equal the 494 counted in the earlier cng-based
analysis. The script prints both its own totals and that comparison rather than
assuming they agree.

THE THREE TESTS
---------------
F1  Direction of the nom/acc confusion. A ~95/5 split means the model is
    emitting a fixed prior on homographs, which any context-sensitive scorer
    can improve on. A ~50/50 split means it is already conditioning on
    something and these are residual ambiguity.

F2  Constraint separability -- THE ACTUAL GATE. For each case error, is the
    gold path LEGAL and the predicted path ILLEGAL under a clause-level
    constraint? If both are legal, no reranker over role counts can prefer
    gold, however well trained. Below ~40% separable, the mechanism story is
    wrong and the plan stops.

F2b Verbless sentences. Sanskrit nominal sentences have no finite verb to
    compete for a subject role, so "one nominative per finite verb" has no
    force there. These are permanently unreachable by this mechanism and come
    off the headroom estimate up front.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ingest"))

from dataset import LatticeStore, collate      # noqa: E402
from model import BiaffineEdgeScorer           # noqa: E402
from morph_table import COL, MorphTable        # noqa: E402
from train import batches, to_torch            # noqa: E402
from viterbi import predicted_nodes, viterbi   # noqa: E402


# --- the candidate constraints ------------------------------------------- #
#
# Each returns True when the path is LEGAL. They are deliberately crude: the
# question is not whether they are good syntax, it is whether role counts carry
# enough signal to separate gold from prediction at all. A constraint that
# cannot separate them even in principle cannot be learned by a reranker.

def c_one_nom_per_verb(c):
    """At most one nominative per finite verb (verbless => at most one)."""
    return c["n_nom"] <= max(1, c["n_finite"])


def c_acc_needs_verb(c):
    """An accusative needs something to govern it: a finite verb or participle."""
    return c["n_acc"] == 0 or (c["n_finite"] + c["n_part"]) > 0


def c_nom_le_finite_plus_one(c):
    return c["n_nom"] <= c["n_finite"] + 1


CONSTRAINTS = [
    ("<=1 nominative per finite verb", c_one_nom_per_verb),
    ("accusative implies a governor", c_acc_needs_verb),
    ("n_nom <= n_finite + 1", c_nom_le_finite_plus_one),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache95_ctx_ex4200")
    ap.add_argument("--model", default="model95_ctx_ex4200.npz")
    ap.add_argument("--raw", default="../data/cushr_data_g95.npz")
    ap.add_argument("--morph-vocab", default="../data/morph_vocabulary.txt")
    ap.add_argument("--form-vocab", default="../data/form_vocabulary.txt")
    ap.add_argument("--lemma-vocab", default="../data/lemma_vocabulary.txt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    store = LatticeStore(args.cache)
    dev = torch.device("cpu")
    m = np.load(args.model)
    net = BiaffineEdgeScorer(int(m["feat_dim"]), int(m["hidden"])).to(dev)
    net.src_proj.weight.data = torch.as_tensor(m["src_proj"], device=dev)
    net.dst_proj.weight.data = torch.as_tensor(m["dst_proj"], device=dev)
    net.bias.data = torch.as_tensor(np.asarray(m["bias"]).reshape(1), device=dev)
    net.eval()

    raw = np.load(args.raw)
    fid, lid, cs = raw["node_form_id"], raw["node_lemma_id"], raw["node_char_start"]
    mt = MorphTable(raw, args.morph_vocab)

    def strings(path):
        return [l.split("\t", 1)[1].rstrip("\n") if "\t" in l else ""
                for l in open(path, encoding="utf-8")]

    fv, lv = strings(args.form_vocab), strings(args.lemma_vocab)

    ids = store.trainable(args.split)
    cat = Counter()
    confusion = Counter()
    cases = []          # one record per same-form/same-lemma morph error
    n_words = 0

    with torch.no_grad():
        for chunk in batches(ids, args.batch_size, shuffle=False):
            b = collate(store, chunk)
            t = to_torch(b, dev)
            pe, pmask, _ = viterbi(t, net(t["feats"], t["src"], t["dst"],
                                          t.get("ids"), t))
            pred_local = predicted_nodes(t, pe, pmask)
            gn_ = np.asarray(b["global_node"])
            for i, s in enumerate(chunk):
                s = int(s)
                # predicted_nodes returns Viterbi-BACKTRACKED order, i.e.
                # reversed. Sorting by char start is what fixes it -- the same
                # bug that once reported PM 0.0081 alongside word F 98.10.
                pred = sorted((int(gn_[x]) for x in pred_local[i]),
                              key=lambda g: int(cs[g]))
                pred = [g for g in pred if fv[fid[g]]]
                gold = sorted(
                    store.gold_nodes[store.gold_off[s]:store.gold_off[s + 1]].tolist(),
                    key=lambda g: int(cs[g]))
                gold = [g for g in gold if fv[fid[g]]]
                if len(pred) != len(gold):
                    cat["sentence: different word count"] += 1
                    continue
                pc, gc = mt.counts(pred), mt.counts(gold)
                for p, g in zip(pred, gold):
                    n_words += 1
                    if p == g:
                        cat["node correct"] += 1
                        continue
                    if fv[fid[p]] != fv[fid[g]]:
                        cat["WRONG NODE: different surface form"] += 1
                        continue
                    if lv[lid[p]] != lv[lid[g]]:
                        cat["WRONG NODE: same form, different lemma"] += 1
                        continue
                    tp, tg = mt.tag(p), mt.tag(g)
                    if tp == tg:
                        cat["WRONG NODE: same form+lemma+morph"] += 1
                        continue
                    cat["WRONG NODE: same form+lemma, different morph"] += 1
                    confusion[(tp, tg)] += 1
                    cases.append({"sent": s, "pred_tag": tp, "gold_tag": tg,
                                  "pred_counts": pc, "gold_counts": gc})

    tot_wrong = sum(v for k, v in cat.items() if k.startswith("WRONG"))
    print(f"word positions compared : {n_words:,}")
    print(f"sentences skipped (len) : {cat['sentence: different word count']:,}")
    print()
    for k, v in cat.most_common():
        if k.startswith("sentence"):
            continue
        extra = f"  ({100 * v / tot_wrong:5.1f}% of errors)" if k.startswith("WRONG") else ""
        print(f"  {k:<46} {v:6,}  {100 * v / n_words:5.2f}% of words{extra}")

    print("\nmost common morph confusions (model picked -> gold had):")
    for (a, b_), c in confusion.most_common(12):
        print(f"    {a:<22} -> {b_:<22} {c:5,}")

    # ---------------------------------------------------------------- F1 --
    print("\n" + "=" * 72)
    print("F1  direction of the nominative/accusative confusion")
    print("=" * 72)
    na = [c for c in cases
          if {c["pred_tag"], c["gold_tag"]} == {"nom. sg. n.", "acc. sg. n."}]
    p2a = sum(1 for c in na if c["pred_tag"] == "acc. sg. n.")
    print(f"  nom.sg.n. <-> acc.sg.n. errors : {len(na):,}")
    if na:
        print(f"    gold nom, model said acc     : {p2a:5,}  ({100*p2a/len(na):5.1f}%)")
        print(f"    gold acc, model said nom     : {len(na)-p2a:5,}  "
              f"({100*(len(na)-p2a)/len(na):5.1f}%)")
        skew = max(p2a, len(na) - p2a) / len(na)
        print(f"  -> {'DIRECTIONAL PRIOR' if skew >= 0.8 else 'NOT strongly directional'}"
              f" (skew {100*skew:.1f}%)")

    # widen to all case-only confusions, which is the population a reranker
    # would actually target
    case_only = [c for c in cases
                 if c["pred_tag"].split(".", 1)[1:] == c["gold_tag"].split(".", 1)[1:]
                 and c["pred_tag"].split(".")[0] in ("nom", "acc", "dat", "abl",
                                                     "g", "loc", "voc", "i")]
    print(f"  all case-only confusions       : {len(case_only):,} "
          f"({100*len(case_only)/max(1,len(cases)):.1f}% of morph errors)")

    # ---------------------------------------------------------------- F2b -
    print("\n" + "=" * 72)
    print("F2b  how many sit in sentences with no finite verb?")
    print("=" * 72)
    for label, pop in (("nom/acc errors", na), ("all morph errors", cases)):
        if not pop:
            continue
        nv = sum(1 for c in pop if c["gold_counts"]["n_finite"] == 0)
        print(f"  {label:<18} {nv:5,} / {len(pop):5,} = {100*nv/len(pop):5.1f}% verbless")
    print("  (verbless sentences have no verb to compete for a subject role,")
    print("   so a role-competition constraint has no force there)")

    # ----------------------------------------------------------------- F2 -
    print("\n" + "=" * 72)
    print("F2  constraint separability  -- THE GATE")
    print("=" * 72)
    print("  % of errors where GOLD is legal and PREDICTION is illegal.")
    print("  Both legal => no count-based reranker can prefer gold.\n")
    print(f"  {'constraint':<34} {'nom/acc':>16} {'all morph':>16}")
    best = 0.0
    for name, fn in CONSTRAINTS:
        row = []
        for pop in (na, cases):
            if not pop:
                row.append("       n/a")
                continue
            sep = sum(1 for c in pop
                      if fn(c["gold_counts"]) and not fn(c["pred_counts"]))
            frac = 100 * sep / len(pop)
            if pop is na:
                best = max(best, frac)
            row.append(f"{sep:6,} {frac:5.1f}%")
        print(f"  {name:<34} {row[0]:>16} {row[1]:>16}")

    # any-constraint union: the most generous reading available
    if na:
        anysep = sum(1 for c in na
                     if any(fn(c["gold_counts"]) and not fn(c["pred_counts"])
                            for _, fn in CONSTRAINTS))
        frac = 100 * anysep / len(na)
        best = max(best, frac)
        print(f"  {'ANY of the above (union)':<34} {anysep:6,} {frac:5.1f}%")

        # how often are the two paths even DISTINGUISHABLE by role counts?
        ident = sum(1 for c in na if c["pred_counts"] == c["gold_counts"])
        print(f"\n  gold and predicted role counts IDENTICAL: {ident:,} / {len(na):,}"
              f" = {100*ident/len(na):.1f}%")
        print("  (these are unreachable by ANY reranker over role counts,")
        print("   trained or not -- the feature vectors are equal)")

    print("\n" + "=" * 72)
    verdict = "PASS -- proceed to Phase 2" if best >= 40 else "FAIL -- stop, report negative result"
    print(f"  GATE (best separability >= 40%): {best:.1f}%  =>  {verdict}")
    print("=" * 72)

    if args.out:
        json.dump(cases, open(args.out, "w"), ensure_ascii=False)
        print(f"\nwrote {args.out}  ({len(cases):,} error records)")


if __name__ == "__main__":
    main()
