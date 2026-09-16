#!/usr/bin/env python3
"""Invariance tests for the lattice-attention encoder (context.py).

These are the regression suite for `lattice_attn` / `lattice_attn_char`, because
every failure mode here is SILENT: the model still trains, the loss still falls,
and the score is merely worse. In particular:

  padding leakage   attending to padded slots averages in zeros
  cross-sentence    attending past a sentence boundary invents context
  chunk dependence  training (batch 64) and materialisation (chunk 256) would
                    then disagree, so the exported node vectors would not be the
                    ones the model was trained with
  NaN               an all--inf softmax row poisons the batch through the residual

The chunk-invariance test is the one that licenses `materialize_contextual` to
stay unchanged: if the pair budget cannot change the output, neither can the
batch size.

Usage:
    python test_lattice_attn.py --cache ./cache95_ngrams80
"""
import argparse

import numpy as np
import torch

import context as CTX
from dataset import LatticeStore, collate

BASE_DIM = 96


def batch_tensors(store, sent_ids, seed=0):
    """Real spans/chars for `sent_ids`, with a random stand-in for `base`."""
    b = collate(store, np.asarray(sent_ids, dtype=np.int64))
    g = torch.Generator().manual_seed(seed)
    t = {
        "chars": torch.as_tensor(b["chars"], dtype=torch.int64),
        "char_len": torch.as_tensor(b["char_len"], dtype=torch.int64),
        "node_sent": torch.as_tensor(b["node_sent"], dtype=torch.int64),
        "span_start": torch.as_tensor(b["span_start"], dtype=torch.int64),
        "span_end": torch.as_tensor(b["span_end"], dtype=torch.int64),
        "char_ok": torch.as_tensor(np.asarray(b["char_ok"]), dtype=torch.bool),
    }
    base = torch.randn(b["num_nodes"], BASE_DIM, generator=g)
    return b, t, base


def run(enc, t, base, sent_slice=None):
    with torch.no_grad():
        return enc(t["chars"], t["char_len"], t["node_sent"], t["span_start"],
                   t["span_end"], t["char_ok"], base)


def make(name="lattice_attn_char", **kw):
    opts = dict(base_dim=BASE_DIM, hidden=64, layers=2, out_dim=96, heads=8,
                max_rel=64, char_dim=32, dropout=0.0)
    opts.update(kw)
    enc = CTX.get(name, **opts)
    enc.eval()
    torch.manual_seed(0)
    # Zero-initialised span tables would make the bias tests vacuous.
    if enc.span_tables is not None:
        for tab in enc.span_tables:
            torch.nn.init.normal_(tab.weight, std=0.5)
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache95_ngrams80")
    ap.add_argument("--sentences", type=int, default=48)
    ap.add_argument("--tol", type=float, default=0.0,
                    help="0 = pick from dtype (5e-6 float32, 1e-12 float64)")
    ap.add_argument("--double", action="store_true",
                    help="run every check in float64. The invariances below are "
                         "EXACT mathematically but not bitwise in float32: "
                         "padding width and chunking change the reduction order "
                         "inside attention, which moves the last ~1e-6. If a "
                         "delta shrinks to ~1e-13 here it was rounding; if it "
                         "survives, it is a real bug.")
    args = ap.parse_args()
    if args.double:
        torch.set_default_dtype(torch.float64)
    if not args.tol:
        args.tol = 1e-12 if args.double else 5e-6

    store = LatticeStore(args.cache)
    if not store.has_chars:
        raise SystemExit(f"{args.cache} has no surface_text; use a cache built "
                         "with RAW_PASSTHROUGH (e.g. ./cache95_ngrams80)")
    # Sentences of mixed length: padding bugs hide when everything is the same size.
    sizes = np.diff(store.sent_off)
    order = np.argsort(sizes[:20000])
    ids = np.concatenate([order[:args.sentences // 2],
                          order[-args.sentences // 2:]])
    ids = np.sort(ids)
    n_fail = 0

    def check(name, ok, detail=""):
        nonlocal n_fail
        n_fail += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")

    enc = make()
    b, t, base = batch_tensors(store, ids)
    counts = np.bincount(np.asarray(b["node_sent"]), minlength=len(ids))
    print(f"{len(ids)} sentences, {b['num_nodes']:,} nodes, "
          f"max {counts.max()} nodes/sentence, {len(store.sent_off)-1:,} in corpus\n")

    ctx = run(enc, t, base)

    # 1. finite, and boundary rows exactly zero
    check("finite outputs", bool(torch.isfinite(ctx).all()))
    bad = ctx[~t["char_ok"]]
    check("boundary rows are exactly zero", bool((bad == 0).all()),
          f"max |ctx[~char_ok]| = {float(bad.abs().max()) if bad.numel() else 0.0}")

    # 2. padding invariance: one sentence alone must equal that sentence batched
    #    with everything else. This is the classic silent failure.
    worst, worst_sent = 0.0, -1
    for si in (0, len(ids) // 2, len(ids) - 1):
        _, t1, _ = batch_tensors(store, [ids[si]])
        sel = torch.as_tensor(np.asarray(b["node_sent"]) == si)
        solo = run(enc, t1, base[sel])
        d = float((solo - ctx[sel]).abs().max())
        if d > worst:
            worst, worst_sent = d, si
    check("padding invariance (alone vs batched)", worst < args.tol,
          f"max |delta| = {worst:.2e} (sentence {worst_sent}, tol {args.tol:g})")

    # 3. chunk invariance: the pair budget must not change the result. This is
    #    what makes training at batch 64 and materialising at chunk 256 agree.
    small = make(pair_budget=4096)
    small.load_state_dict(enc.state_dict())
    small.eval()
    d = float((run(small, t, base) - ctx).abs().max())
    check("chunk invariance (budget 1e6 vs 4096)", d < args.tol,
          f"max |delta| = {d:.2e}")

    # 4. no cross-sentence leakage: perturbing one sentence must leave every
    #    other sentence's vectors untouched.
    node_sent = torch.as_tensor(np.asarray(b["node_sent"]))
    victim = int(node_sent[len(node_sent) // 2])
    base2 = base.clone()
    base2[node_sent == victim] += 7.0
    ctx2 = run(enc, t, base2)
    other = node_sent != victim
    d_other = float((ctx2[other] - ctx[other]).abs().max())
    d_self = float((ctx2[~other] - ctx[~other]).abs().max())
    check("no cross-sentence leakage", d_other < args.tol,
          f"other sentences moved {d_other:.2e}")
    check("attention is not inert", d_self > 1e-4,
          f"perturbed sentence moved {d_self:.3f}")

    # 5. permutation equivariance: node order within a sentence is a storage
    #    artefact, so permuting it must permute the outputs and nothing else.
    perm = np.arange(b["num_nodes"])
    ns = np.asarray(b["node_sent"])
    for si in range(len(ids)):
        idx = np.nonzero(ns == si)[0]
        perm[idx] = np.random.default_rng(si).permutation(idx)
    p = torch.as_tensor(perm)
    tp = dict(t)
    for k in ("span_start", "span_end", "char_ok"):
        tp[k] = t[k][p]
    ctx_p = run(enc, tp, base[p])
    d = float((ctx_p - ctx[p]).abs().max())
    check("permutation equivariance", d < args.tol, f"max |delta| = {d:.2e}")

    # 6. the span bias must actually be doing something
    plain = make(span_bias=False)
    sd = {k: v for k, v in enc.state_dict().items()
          if not k.startswith("span_tables")}
    plain.load_state_dict(sd, strict=False)
    plain.eval()
    d = float((run(plain, t, base) - ctx).abs().max())
    check("span bias changes the output", d > 1e-4, f"max |delta| = {d:.3f}")
    check("span-bias-free encoder is finite",
          bool(torch.isfinite(run(plain, t, base)).all()))

    # 7. char_bilstm is untouched by the signature change
    cb = CTX.get("char_bilstm", char_dim=32, hidden=64, layers=2, out_dim=96,
                 base_dim=BASE_DIM, heads=8)       # extra kwargs must be dropped
    cb.eval()
    with torch.no_grad():
        six = cb(t["chars"], t["char_len"], t["node_sent"], t["span_start"],
                 t["span_end"], t["char_ok"])
        seven = cb(t["chars"], t["char_len"], t["node_sent"], t["span_start"],
                   t["span_end"], t["char_ok"], base)
    check("char_bilstm ignores base (bit-identical)", bool((six == seven).all()))

    # 8. materialize_contextual must reproduce the training path. It rebuilds its
    #    own 6-key tensor dict (context.py) and walks sentences in chunks of its
    #    own choosing, so this is where a silent train/export divergence would
    #    live. Run it over a truncated store, with a chunk size deliberately
    #    different from the batch used for the direct call.
    from model import BiaffineEdgeScorer

    if args.double:
        # materialize_contextual builds its feats tensor as float32 by hand, so
        # this one check only runs in the default dtype.
        print("  SKIP  materialize path == training path (float32 only)")
        print(f"\n{'ALL CHECKS PASSED' if not n_fail else str(n_fail) + ' CHECK(S) FAILED'}")
        raise SystemExit(1 if n_fail else 0)

    n_keep = 32
    sub = LatticeStore(args.cache)
    sub.sent_off = store.sent_off[:n_keep + 1]
    enc80 = make(base_dim=int(store.feat_dim))
    net = CTX.ContextualBiaffine(
        None, enc80, BiaffineEdgeScorer(int(store.feat_dim) + enc80.out_dim, 32))
    net.eval()

    dim = int(store.feat_dim) + enc80.out_dim
    out = np.zeros((int(sub.sent_off[-1]), dim), dtype=np.float32)
    CTX.materialize_contextual(net, sub, torch.device("cpu"), out, sent_chunk=7)

    bb = collate(store, np.arange(n_keep, dtype=np.int64))
    tt = {k: torch.as_tensor(np.asarray(bb[k]),
                             dtype=torch.bool if k == "char_ok" else torch.int64)
          for k in ("chars", "char_len", "node_sent", "span_start", "span_end",
                    "char_ok")}
    net.eval()
    with torch.no_grad():
        direct = net.node_vectors(
            torch.as_tensor(bb["feats"], dtype=torch.float32), None, tt).numpy()
    d = float(np.abs(direct - out[bb["global_node"]]).max())
    check("materialize path == training path", d < args.tol,
          f"max |delta| = {d:.2e} (chunk 7 vs batch {n_keep})")

    print(f"\n{'ALL CHECKS PASSED' if not n_fail else str(n_fail) + ' CHECK(S) FAILED'}")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
