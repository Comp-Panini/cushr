#!/usr/bin/env python3
"""Paired significance of the seg+attn ensemble against each single model.

Reads three --dump files from eval_union.py, all on the same 4,200 SIGHUM
sentences. The single-model dumps come from running eval_union.py with a model
paired against a copy of itself, which reproduces that model alone exactly
(union_selftest.json: seg + segcopy = seg = 93.43).

Reports: exact McNemar on the sentences where the two systems disagree, a
paired bootstrap CI on the PM difference, an unpaired bootstrap CI on the
ensemble PM, and the agreement table that explains where the gain comes from.

    python union_significance.py
"""
import argparse
import json
import random
from math import comb


def load(path):
    return {d["id"]: bool(d["exact"]) for d in json.load(open(path))}


def mcnemar(x, y, ids):
    b = sum(x[i] and not y[i] for i in ids)
    c = sum(y[i] and not x[i] for i in ids)
    m, k = b + c, min(b, c)
    p = min(1.0, 2 * sum(comb(m, j) for j in range(k + 1)) / 2 ** m)
    return b, c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens", default="eval_union_sighum_k16_preds.json")
    ap.add_argument("--seg", default="eval_union_seg_alone_preds.json")
    ap.add_argument("--attn", default="eval_union_attn_alone_preds.json")
    ap.add_argument("--ref-pm", type=float, default=93.97, help="TransLIST")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    E, A, B = load(args.ens), load(args.seg), load(args.attn)
    assert E.keys() == A.keys() == B.keys()
    ids = sorted(E)
    n = len(ids)
    pm = lambda d: 100 * sum(d[i] for i in ids) / n
    print(f"n={n}  PM ensemble {pm(E):.2f}  seg {pm(A):.2f}  attn {pm(B):.2f}")

    for name, other in (("seg", A), ("attn", B)):
        b, c, p = mcnemar(E, other, ids)
        print(f"ensemble vs {name}: only ensemble right {b}, only {name} "
              f"right {c}, net +{b - c}, exact McNemar p={p:.2e}")

    both = sum(A[i] and B[i] for i in ids)
    only_a = sum(A[i] and not B[i] for i in ids)
    only_b = sum(B[i] and not A[i] for i in ids)
    neither = n - both - only_a - only_b
    print(f"seg/attn: both right {both}, only seg {only_a}, only attn "
          f"{only_b}, neither {neither}")
    print(f"  ensemble right on {sum(E[i] for i in ids if A[i] != B[i])} of "
          f"{only_a + only_b} disagreements, "
          f"{sum(E[i] for i in ids if not A[i] and not B[i])} of {neither} "
          f"both-wrong, broke {sum(not E[i] for i in ids if A[i] and B[i])} "
          f"of {both} both-right")

    random.seed(0)
    e = [E[i] for i in ids]
    a = [A[i] for i in ids]
    pms, dif = [], []
    for _ in range(args.boot):
        s = [random.randrange(n) for _ in range(n)]
        se, sa = sum(e[j] for j in s), sum(a[j] for j in s)
        pms.append(100 * se / n)
        dif.append(100 * (se - sa) / n)
    pms.sort()
    dif.sort()
    lo, hi = int(0.025 * args.boot), int(0.975 * args.boot)
    print(f"ensemble PM 95% CI [{pms[lo]:.2f}, {pms[hi]:.2f}]")
    print(f"ensemble - seg PM 95% CI (paired) [{dif[lo]:.2f}, {dif[hi]:.2f}]")
    print(f"resamples with ensemble PM <= {args.ref_pm}: "
          f"{sum(x <= args.ref_pm for x in pms) / args.boot:.3f}")


if __name__ == "__main__":
    main()
