#!/usr/bin/env python3
"""Sum-rule ensemble of two base models' beams, scored as a word sequence.

WHY. cuSHR-seg (char BiLSTM) and cuSHR-attn (lattice self-attention) score the
same to within 0.05 PM, but their top-16 beams barely overlap: on the g95 test
split, 13,751 candidates appear in both, 29,620 only in seg's beam and 34,329
only in attn's. Different errors at equal accuracy is the textbook case for an
ensemble, and on g95 this was worth +1.02 over the better single model. This
script measures the same thing on the published SIGHUM 4,200, which is the only
number comparable to TransLIST.

METHOD. The two models' base Viterbi scores are NOT on a common scale, so they
cannot be compared or averaged directly. Instead, per model:
  1. decode its own top-K distinct segmentations,
  2. score them with its own rerankers (several seeds), softmax WITHIN that
     beam, average the seeds -> a probability over that model's candidates.
A candidate missing from a model's beam gets probability 0 from that model.
The ensemble score is the mean of the two probabilities: the standard sum rule
over disjoint candidate sets.

A candidate's identity is its word sequence -- the same thing the metric scores,
so two models proposing the same segmentation by different node ids merge, which
is correct here.

Scoring is copied from eval_surface.py so the numbers are directly comparable:
PM plus macro and micro word P/R/F, macro being TransLIST's protocol.

RECIPE IS FIXED. K, the reranker set and the equal weights were all chosen on
g95 before this was run. Changing any of them after seeing a SIGHUM number is
tuning on the test set -- do not.
"""
import argparse
import csv
import json
from collections import Counter

import numpy as np
import torch

from dataset import LatticeStore, collate
from model import BiaffineEdgeScorer
from train import batches, to_torch
from kbest import kbest, predicted_nodes_k
import surface_cands as SC


def load_forms(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        out.append(line.split("\t", 1)[1] if "\t" in line else "")
    return out


def load_ranker(path, forms, raw_fid):
    from rerank import PathRanker, form_index
    ck = torch.load(path, map_location="cpu", weights_only=False)
    r = PathRanker(ck["n_morph"], ck["n_lemma"], hidden=ck["hidden"],
                   use_lemma=ck.get("use_lemma", True),
                   n_form=ck.get("n_form", 0),
                   form_emb=ck.get("form_emb", 32),
                   use_seq=ck.get("use_seq", True),
                   n_feat=ck.get("n_feat", 0),
                   feat_hidden=ck.get("feat_hidden", 16))
    r.load_state_dict(ck["state"])
    r.eval()
    form_ix = form_index(ck["form_vocab"], raw_fid) if ck.get("n_form", 0) else None
    seg = None
    if ck.get("n_feat", 0):
        import seg_features as SF
        seg = SF.SegFeaturizer(ck["seg_fids"], ck["seg_counts"],
                               SF.form_lengths(forms))
    return r, form_ix, seg


@torch.no_grad()
def model_probs(cache, model, ranker_paths, sids, k, raw, forms, dev):
    """-> {sentence id: {word tuple: probability}} for one base model."""
    raw_fid = raw["node_form_id"]
    char_start = raw["node_char_start"]
    r_morph = np.asarray(raw["node_features"]).ravel().astype(np.int64)
    r_lemma = np.asarray(raw["node_lemma_id"]).ravel().astype(np.int64)

    rankers = [load_ranker(p, forms, raw_fid) for p in ranker_paths]

    m = np.load(model)
    net = BiaffineEdgeScorer(int(m["feat_dim"]), int(m["hidden"])).to(dev)
    net.src_proj.weight.data = torch.as_tensor(m["src_proj"], device=dev)
    net.dst_proj.weight.data = torch.as_tensor(m["dst_proj"], device=dev)
    net.bias.data = torch.as_tensor(np.asarray(m["bias"]).reshape(1), device=dev)
    net.eval()
    store = LatticeStore(cache)

    out = {}
    for chunk in batches(sids, 128, shuffle=False):
        b = collate(store, chunk)
        t = to_torch(b, dev)
        w_edge = net(t["feats"], t["src"], t["dst"], t.get("ids"), t)
        ke, km, ksc, kv = kbest(t, w_edge, k)
        kb = predicted_nodes_k(t, ke, km, kv)
        gn = np.asarray(b["global_node"])
        for i, s in enumerate(chunk):
            seqs, scs = SC.sentence_candidates(
                kb[i], ksc[i], gn, char_start, raw_fid, forms, k, True)
            if not seqs:
                out[int(s)] = {}
                continue
            # Each reranker scores this beam; softmax within the beam, then
            # average over seeds. Base scores never leave their own beam.
            T = max(1, max(len(q) for q in seqs))
            mo = np.zeros((len(seqs), T), dtype=np.int64)
            le = np.zeros_like(mo)
            ln = np.ones(len(seqs), dtype=np.int64)
            for j, q in enumerate(seqs):
                if q:
                    mo[j, :len(q)] = r_morph[q]
                    le[j, :len(q)] = r_lemma[q]
                    ln[j] = len(q)
            base = torch.as_tensor(np.asarray(scs, np.float32))
            acc = None
            for r, form_ix, seg in rankers:
                fo = None
                if form_ix is not None:
                    fo = np.zeros_like(mo)
                    for j, q in enumerate(seqs):
                        if q:
                            fo[j, :len(q)] = form_ix[q]
                sc = r(torch.as_tensor(mo), torch.as_tensor(le),
                       torch.as_tensor(ln), base,
                       form=torch.as_tensor(fo) if fo is not None else None,
                       feats=(torch.as_tensor(seg.feats(seqs, scs, raw_fid))
                              if seg is not None else None))
                p = torch.softmax(sc.reshape(-1), dim=0)
                acc = p if acc is None else acc + p
            p = (acc / len(rankers)).numpy()
            d = {}
            for q, pp in zip(seqs, p):
                key = tuple(w for w in (forms[raw_fid[x]] for x in q) if w)
                d[key] = max(d.get(key, 0.0), float(pp))
            out[int(s)] = d
    return out


def score(name, picks, ref, has_gold, index, dump_path=""):
    pm = n = tp = fp = fn = 0
    mp, mr, mf = [], [], []
    by_group = {"gold-resolved": [0, 0], "no-gold-path": [0, 0]}
    dump = []
    for s, words in picks.items():
        gold = ref[s]
        words = [w for w in words if w]
        pm += int(words == gold)
        n += 1
        cp, cg = Counter(words), Counter(gold)
        inter = sum((cp & cg).values())
        tp += inter
        fp += sum(cp.values()) - inter
        fn += sum(cg.values()) - inter
        p_i = inter / len(words) if words else 0.0
        r_i = inter / len(gold) if gold else 0.0
        mp.append(p_i)
        mr.append(r_i)
        mf.append(2 * p_i * r_i / (p_i + r_i) if p_i + r_i else 0.0)
        bucket = "gold-resolved" if has_gold[s] else "no-gold-path"
        by_group[bucket][0] += int(words == gold)
        by_group[bucket][1] += 1
        if dump_path:
            dump.append({"id": index[s], "pred": words, "gold": gold,
                         "exact": words == gold, "group": bucket})
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    res = {"n": n, "pm": 100 * pm / n,
           "p_macro": 100 * float(np.mean(mp)),
           "r_macro": 100 * float(np.mean(mr)),
           "f1_macro": 100 * float(np.mean(mf)),
           "p_micro": 100 * prec, "r_micro": 100 * rec, "f1_micro": 100 * f1,
           "by_group": {k: {"pm": 100 * h / t, "n": t}
                        for k, (h, t) in by_group.items() if t}}
    print(f"\n{name}  (n={n:,})")
    print(f"  perfect match     : {res['pm']:.2f}")
    print(f"  word P/R/F MACRO  : {res['p_macro']:.2f} / {res['r_macro']:.2f} /"
          f" {res['f1_macro']:.2f}   <- TransLIST's protocol")
    print(f"  word P/R/F micro  : {res['p_micro']:.2f} / {res['r_micro']:.2f} /"
          f" {res['f1_micro']:.2f}")
    for k, v in res["by_group"].items():
        print(f"    {k:<16} PM {v['pm']:6.2f}   n={v['n']:,}")
    if dump_path:
        json.dump(dump, open(dump_path, "w"), ensure_ascii=False)
        print(f"  wrote {dump_path}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-a", required=True)
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--rerank-a", nargs="+", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--cache-b", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--rerank-b", nargs="+", required=True)
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--tsv", default="sighum_test_4200.tsv")
    ap.add_argument("--index", default="../data/sentence_index_g95.json")
    ap.add_argument("--raw", default="../data/cushr_data_g95.npz")
    ap.add_argument("--vocab", default="../data/form_vocabulary.txt")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.tsv, encoding="utf-8-sig"),
                               delimiter="\t"))
    index = json.load(open(args.index))
    pos = {s: i for i, s in enumerate(index)}
    forms = load_forms(args.vocab)
    raw = np.load(args.raw)

    pairs = [(pos[r["DCS-ID"].strip()], r["output"].split())
             for r in rows if r["DCS-ID"].strip() in pos]
    ref = dict(pairs)
    sids = np.asarray([s for s, _ in pairs], dtype=np.int64)
    print(f"{args.tsv}: {len(rows):,} rows, {len(sids):,} located in our corpus")
    print(f"K={args.k} distinct segmentations per model; "
          f"{len(args.rerank_a)}+{len(args.rerank_b)} rerankers, equal weight")

    store = LatticeStore(args.cache_a)
    has_gold = (store.gold_off[1:] - store.gold_off[:-1]) > 0
    del store

    dev = torch.device("cpu")
    # One model at a time: each cache is multi-GB and nothing needs both open.
    A = model_probs(args.cache_a, args.model_a, args.rerank_a, sids, args.k,
                    raw, forms, dev)
    print(f"decoded {args.name_a}")
    B = model_probs(args.cache_b, args.model_b, args.rerank_b, sids, args.k,
                    raw, forms, dev)
    print(f"decoded {args.name_b}")

    pick_a, pick_b, pick_e = {}, {}, {}
    both = only_a = only_b = 0
    ceil = n = 0
    for s in sorted(ref):
        a, b = A.get(s, {}), B.get(s, {})
        keys = list(dict.fromkeys(list(a) + list(b)))
        if not keys:
            pick_a[s] = pick_b[s] = pick_e[s] = []
            continue
        for key in keys:
            if key in a and key in b:
                both += 1
            elif key in a:
                only_a += 1
            else:
                only_b += 1
        n += 1
        ceil += int(tuple(ref[s]) in set(keys))
        pick_a[s] = list(max(a, key=a.get)) if a else []
        pick_b[s] = list(max(b, key=b.get)) if b else []
        comb = {key: (a.get(key, 0.0) + b.get(key, 0.0)) / 2 for key in keys}
        pick_e[s] = list(max(comb, key=comb.get))

    print(f"\nunion candidates: in both {both:,}  only {args.name_a} "
          f"{only_a:,}  only {args.name_b} {only_b:,}")
    print(f"union oracle (gold word sequence present): {100 * ceil / n:.2f}")

    ra = score(f"{args.name_a} alone, {len(args.rerank_a)}-seed prob-mean",
               pick_a, ref, has_gold, index)
    rb = score(f"{args.name_b} alone, {len(args.rerank_b)}-seed prob-mean",
               pick_b, ref, has_gold, index)
    re_ = score("SUM-RULE ENSEMBLE over the union", pick_e, ref, has_gold,
                index, args.dump)

    if args.json_out:
        json.dump({"k": args.k, "tsv": args.tsv,
                   "models": {args.name_a: {"cache": args.cache_a,
                                            "model": args.model_a,
                                            "rerank": args.rerank_a},
                              args.name_b: {"cache": args.cache_b,
                                            "model": args.model_b,
                                            "rerank": args.rerank_b}},
                   "union": {"in_both": both, f"only_{args.name_a}": only_a,
                             f"only_{args.name_b}": only_b,
                             "oracle_pm": 100 * ceil / n},
                   args.name_a: ra, args.name_b: rb, "ensemble": re_},
                  open(args.json_out, "w"), indent=1)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
