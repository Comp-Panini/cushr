#!/usr/bin/env python3
"""Score at ByT5-Sanskrit's task levels: S, L, S+M, L+M, S+L+M.

ByT5-Sanskrit (Nehrdich et al. 2024, Table 7) reports SENTENCE-LEVEL PERFECT
MATCH for segmentation (S), lemmatisation (L), and their combinations with
morphosyntactic tagging (M). A cuSHR lattice node is (form, lemma, cng), so
choosing a path commits to all three at once and the whole ladder falls out of
how much of each node has to match.

WHAT CHANGED, AND WHY
---------------------
An earlier version of this script scored against OUR OWN reconstructed gold
path, restricted to sentences whose gold path ingest could resolve. That was
wrong twice over: the reference was the pipeline's own output (so it could not
detect a systematically mis-resolved gold), and dropping unresolved sentences
kept only the easy ones -- it reported S at 92.98 where the unbiased number was
82.60.

Both references here are EXTERNAL and exist for every test sentence:

  form            <- sighum_test_4200.tsv `output` column  (published segmentation)
  lemma, cng      <- SIGHUM_database_gold_path/DCS_pick/<id>.p  (published DCS)

They are positionally aligned: the pickle's flattened (lemmas, cng) sequence has
exactly the same length as the TSV's output words for 4,200/4,200 sentences.
That is asserted below, not assumed.

Every sentence is scored whether or not ingest resolved a gold path for it, so
S here is directly comparable to eval_surface.py's perfect match. If the two
disagree, something in this file's reference wiring is broken.

COMPARABILITY CAVEATS -- read before quoting these numbers
----------------------------------------------------------
1. ByT5's Table 7 is measured on a DCS April-2024 split (8,398 test sentences),
   NOT on SIGHUM. Their only SIGHUM figure is the 93.83 segmentation PM. The
   ladder therefore compares our-SIGHUM against their-DCS: same metric
   definition, different corpus. Compare against their `Sen` column, never
   `Par` -- `Par` feeds pseudo-paragraphs of up to 512 chars, a different input
   granularity from our per-sentence decode.
2. The M level assumes SHR's `cng` and DCS's `cng` are the same tag scheme. They
   are: ingest's gold resolution joins on `(chunk, lemma, cng)` with a literal
   string comparison of the two, and that join succeeds for ~94% of the corpus,
   which it could not if the schemes differed. NOTE this is `cng`, not the
   `morph` string in node_features -- those are NOT interchangeable (288 morph
   strings collapse onto 168 cng values), which is why cng is read from the
   graphml here rather than taken from the archive.
3. The L level is reported STRICT and TOLERANT. DCS and SHR use different lemma
   spelling conventions (`chada`/`Cada`, `mard`/`mfd`), so a literal comparison
   understates. Strict = exact match after ingest.normalize_lemma. Tolerant =
   the predicted lemma is anywhere in ingest.lemma_candidates(gold). Tolerant is
   an UPPER BOUND and is sensitive to how wide the normaliser happens to be --
   widening it inflates this number, so strict is the primary figure.
4. **The L level is capped well below 100 by a convention difference, not by
   model quality.** Scoring the GOLD PATH itself gives L = 70.19% at sentence
   level (94.84% at word level): DCS lemmatises participles to the verbal root
   (`kf`, `dA`, `pratipad`) where SHR gives the participial stem (`kfta`,
   `deya`, `pratipanna`). Our own orphan repair makes this worse, since it
   deliberately redirects gold onto the inflected wired twin
   (`INGEST_METHODOLOGY.md` §5). Any L / L+M / S+L+M number here is therefore
   measuring "does the model reproduce SHR's lemma convention", NOT "can it
   lemmatise" -- and is not commensurable with ByT5's L, which is trained to
   emit DCS lemmas directly. The ORACLE row below reports that ceiling for every
   level so no number is read without it.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

import networkx as nx
import numpy as np
import torch

from dataset import LatticeStore, collate
from model import BiaffineEdgeScorer
from train import batches, to_torch
from viterbi import viterbi, predicted_nodes

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ingest"))
import ingest as ig  # noqa: E402


def load_vocab(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        out.append(line.split("\t", 1)[1] if "\t" in line else "")
    return out


def dcs_reference(p_dir, stem):
    """(lemmas, cngs) flattened in reading order, or None."""
    path = os.path.join(p_dir, f"{stem}.p")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        d = ig._DCSUnpickler(f, encoding="utf-8").load()
    lem = [str(w) for cl in getattr(d, "lemmas", []) for w in cl]
    cng = [str(c) for cl in getattr(d, "cng", []) for c in cl]
    return lem, cng


def node_cngs(graphml_dir, stem, base, n_nodes):
    """{global node id -> cng string} for one sentence.

    Reproduces ingest's layout: node_list is sorted by numeric node id, and the
    emitted block is [super-source, words..., super-sink], so global id
    base + 1 + i is the i-th word node. Verified against node_form_id by the
    caller rather than trusted.
    """
    G = nx.read_graphml(os.path.join(graphml_dir, f"{stem}.graphml"))
    try:
        nl = sorted(G.nodes(), key=ig._node_id)
    except ValueError:
        nl = list(G.nodes())
    out = {}
    for i, nid in enumerate(nl):
        g = base + 1 + i
        if g < base + n_nodes - 1:
            out[g] = str(G.nodes[nid].get("cng", ""))
    return out, [str(G.nodes[nid].get("word", "")) for nid in nl]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tsv", default="sighum_test_4200.tsv")
    ap.add_argument("--index", default="../data/sentence_index_repaired.json")
    ap.add_argument("--raw", default="../data/cushr_data_repaired.npz")
    ap.add_argument("--form-vocab", default="../data/form_vocabulary.txt")
    ap.add_argument("--lemma-vocab", default="../data/lemma_vocabulary.txt")
    ap.add_argument("--graphml-dir", default="../../SIGHUM_database/After_graphml")
    ap.add_argument("--p-dir", default="../../SIGHUM_database_gold_path/DCS_pick")
    ap.add_argument("--dump", default="")
    ap.add_argument("--pred-jsonl", default="",
                    help="score an EXTERNAL system's predictions (byt5_to_jsonl.py "
                         "output) through this same reference and the same "
                         "score() function, so the only difference between it "
                         "and cuSHR is where the predictions came from")
    ap.add_argument("--pred-name", default="ByT5",
                    help="column label for --pred-jsonl")
    ap.add_argument("--restrict-agreeing-seg", action="store_true",
                    help="score BOTH systems only on sentences where "
                         "--pred-jsonl's segmentation already equals the "
                         "reference. Removes the compound-convention penalty "
                         "from L and M. Biased-easy subset, but identical for "
                         "both systems.")
    ap.add_argument("--tag-bridge", default="",
                    help="tag_bridge.json from build_tag_bridge.py. Translates "
                         "--pred-jsonl's UD bundles into DCS cng so M can be "
                         "scored. Applied to the external system only; the "
                         "reference and cuSHR stay in cng.")
    ap.add_argument("--lemma-map", default="",
                    help="lemma_map.json from build_lemma_map.py. Rewrites "
                         "cuSHR's SHR-convention lemmas into DCS's before "
                         "scoring. Applied to cuSHR and its ORACLE only, never "
                         "to --pred-jsonl.")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.tsv, encoding="utf-8-sig"),
                               delimiter="\t"))
    index = json.load(open(args.index))
    pos = {s: i for i, s in enumerate(index)}
    forms = load_vocab(args.form_vocab)
    lemmas_v = load_vocab(args.lemma_vocab)

    z = np.load(args.raw)
    fid, lid = z["node_form_id"], z["node_lemma_id"]
    cstart = z["node_char_start"]
    soff = z["sentenceoffsets"]
    n_sent_nodes = np.append(soff[1:], len(fid)) - soff

    # ---- build the external reference, and assert it lines up -------------
    ref = {}
    missing = 0
    for r in rows:
        stem = str(r["DCS-ID"]).strip()
        if stem not in pos:
            missing += 1
            continue
        got = dcs_reference(args.p_dir, stem)
        if got is None:
            missing += 1
            continue
        lem, cng = got
        words = r["output"].split()
        if not (len(words) == len(lem) == len(cng)):
            raise SystemExit(
                f"reference length mismatch for {stem}: "
                f"{len(words)} tsv words / {len(lem)} lemmas / {len(cng)} cngs. "
                "The TSV and the DCS pickle no longer align; this script's "
                "whole premise is that they do.")
        ref[pos[stem]] = (words, lem, cng)
    print(f"{args.tsv}: {len(rows):,} rows, {len(ref):,} with a complete "
          f"external reference ({missing:,} unusable)")

    sids = np.asarray(sorted(ref), dtype=np.int64)

    # Optionally restrict to sentences where the external system's segmentation
    # already equals the reference.
    #
    # Motivation: the released chronbmm/sanskrit5-multitask learned DCS compound
    # conventions while this reference uses SIGHUM's, so most of its S failures
    # are convention (droRaputraH vs droRa putraH), not mis-segmentation. That
    # penalty propagates into L and M -- a different split makes the word
    # sequences non-comparable and the sentence is lost outright. No
    # SIGHUM-trained checkpoint was ever released, so this subset is the only
    # way to remove the confound without fine-tuning one.
    #
    # BOTH systems are scored on the SAME restricted set, so the comparison
    # between them stays fair even though the subset is biased easy in absolute
    # terms. Report it beside the full-set numbers, never instead of them.
    if args.restrict_agreeing_seg:
        if not args.pred_jsonl:
            raise SystemExit("--restrict-agreeing-seg needs --pred-jsonl")
        _p = {}
        with open(args.pred_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r_ = json.loads(line)
                    _p[str(r_["id"])] = r_
        keep = [s for s in sids
                if _p.get(str(index[int(s)]), {}).get("words") == ref[int(s)][0]]
        print(f"--restrict-agreeing-seg: {len(keep):,} of {len(sids):,} "
              f"sentences ({100 * len(keep) / len(sids):.1f}%) where "
              f"{args.pred_name}'s segmentation equals the reference")
        sids = np.asarray(keep, dtype=np.int64)

    # ---- cng per node, straight from the graphml --------------------------
    print("reading cng from graphml for the test sentences ...", flush=True)
    cng_of = {}
    bad_align = 0
    for s in sids:
        stem = index[int(s)]
        base = int(soff[s])
        m, words = node_cngs(args.graphml_dir, stem, base, int(n_sent_nodes[s]))
        # verify the id layout against the archive rather than trusting it
        for k, (g, _) in enumerate(zip(sorted(m), words)):
            if forms[fid[g]] != words[k]:
                bad_align += 1
                break
        cng_of.update(m)
    if bad_align:
        raise SystemExit(f"node id layout disagrees with the archive on "
                         f"{bad_align} sentences -- cng would be misattributed")
    print("  layout verified against node_form_id on all test sentences")

    store = LatticeStore(args.cache)
    dev = torch.device("cpu")
    m = np.load(args.model)
    net = BiaffineEdgeScorer(int(m["feat_dim"]), int(m["hidden"])).to(dev)
    net.src_proj.weight.data = torch.as_tensor(m["src_proj"], device=dev)
    net.dst_proj.weight.data = torch.as_tensor(m["dst_proj"], device=dev)
    net.bias.data = torch.as_tensor(np.asarray(m["bias"]).reshape(1), device=dev)
    net.eval()

    has_gold = (store.gold_off[1:] - store.gold_off[:-1]) > 0
    LEVELS = ["S", "L", "L(tol)", "S+M", "L+M", "L+M(tol)", "S+L+M", "S+L+M(tol)"]
    hit = Counter()
    n = 0
    dump = []

    def score(p_form, p_lem, p_cng, g_form, g_lem, g_cng, into):
        """Tally one predicted analysis against the external reference."""
        g_lem_n = [ig.normalize_lemma(x) for x in g_lem]
        s_ok = p_form == g_form
        m_ok = p_cng == g_cng
        l_strict = p_lem == g_lem_n
        l_tol = (len(p_lem) == len(g_lem) and
                 all(pl in ig.lemma_candidates(gl, "extended")
                     for pl, gl in zip(p_lem, g_lem)))
        into["S"] += s_ok
        into["L"] += l_strict
        into["L(tol)"] += l_tol
        into["S+M"] += s_ok and m_ok
        into["L+M"] += l_strict and m_ok
        into["L+M(tol)"] += l_tol and m_ok
        into["S+L+M"] += s_ok and l_strict and m_ok
        into["S+L+M(tol)"] += s_ok and l_tol and m_ok
        return s_ok, l_strict, m_ok

    # SHR -> DCS lemma rewrite, applied to cuSHR's output only.
    #
    # cuSHR reports the lemma on the node it selected, in SHR's convention;
    # the reference is DCS's. For participles they disagree systematically
    # (ukta/vac, gata/gam, kfta/kf), and that one pattern costs most of the L
    # gap. The table is built from the TRAINING split by build_lemma_map.py.
    #
    # NOT applied to --pred-jsonl: ByT5 already emits DCS lemmas, so the table
    # (keyed on SHR strings) could only fire on coincidental collisions and
    # would corrupt correct output.
    lemma_map = {}
    if args.lemma_map:
        blob = json.load(open(args.lemma_map, encoding="utf-8"))
        lemma_map = blob["map"] if "map" in blob else blob
        prov = blob.get("_provenance", {})
        print(f"lemma map: {len(lemma_map):,} rules from {args.lemma_map}"
              + (f" (built on {prov.get('train_sentences_used', '?')} train "
                 f"sentences)" if prov else ""))

    def analysis(nodes):
        return ([forms[fid[g]] for g in nodes],
                [lemma_map.get(lemmas_v[lid[g]], lemmas_v[lid[g]])
                 for g in nodes],
                [cng_of.get(g, "") for g in nodes])

    # ---- oracle ceiling: score the GOLD PATH itself at every level --------
    # Nothing about the model enters this. It is how well our own gold agrees
    # with DCS's published analysis, and it bounds every model number below.
    # Only defined where ingest resolved a path, so it is reported over that
    # subset with the count shown.
    oracle = Counter()
    n_oracle = 0
    for s in sids:
        s = int(s)
        if not has_gold[s]:
            continue
        nodes = sorted(store.gold_nodes[store.gold_off[s]:store.gold_off[s + 1]]
                       .tolist(), key=lambda g: int(cstart[g]))
        nodes = [g for g in nodes if forms[fid[g]]]
        g_form, g_lem, g_cng = ref[s]
        if len(nodes) != len(g_form):
            continue
        score(*analysis(nodes), g_form, g_lem, g_cng, oracle)
        n_oracle += 1
    for chunk in batches(sids, 128, shuffle=False):
        b = collate(store, chunk)
        t = to_torch(b, dev)
        pe, pmask, _ = viterbi(t, net(t["feats"], t["src"], t["dst"],
                                      t.get("ids"), t))
        pred_local = predicted_nodes(t, pe, pmask)
        gn = np.asarray(b["global_node"])
        for i, s in enumerate(chunk):
            nodes = sorted((int(gn[x]) for x in pred_local[i]),
                           key=lambda g: int(cstart[g]))
            nodes = [g for g in nodes if forms[fid[g]]]
            p_form, p_lem, p_cng = analysis(nodes)
            g_form, g_lem, g_cng = ref[int(s)]
            s_ok, l_strict, m_ok = score(p_form, p_lem, p_cng,
                                         g_form, g_lem, g_cng, hit)
            n += 1
            if args.dump:
                dump.append({"id": index[int(s)], "S": s_ok, "L": l_strict,
                             "M": m_ok, "pred_form": p_form, "gold_form": g_form,
                             "pred_lemma": p_lem, "gold_lemma": g_lem,
                             "pred_cng": p_cng, "gold_cng": g_cng})

    # ---- external system, scored through the identical reference + score() ---
    ext = Counter()
    n_ext = 0
    tag_bridge = {}
    if args.tag_bridge:
        tb = json.load(open(args.tag_bridge, encoding="utf-8"))
        tag_bridge = tb["map"] if "map" in tb else tb
        prov = tb.get("_provenance", {})
        print(f"tag bridge: {len(tag_bridge):,} bundles -> cng from "
              f"{args.tag_bridge}"
              + (f" (purity {prov.get('purity_over_supported')}%, "
                 f"{prov.get('sentences_aligned')} train sentences)"
                 if prov else ""))
        if prov and not prov.get("usable", True):
            print("  WARNING: this bridge did not pass its own purity gate")
    ext_tags_usable = bool(tag_bridge)
    if args.pred_jsonl:
        preds = {}
        with open(args.pred_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    preds[str(r["id"])] = r
        missing = 0
        for s in sids:
            s = int(s)
            rec = preds.get(str(index[s]))
            if rec is None:
                missing += 1
                continue
            g_form, g_lem, g_cng = ref[s]
            p_form = rec.get("words") or []
            p_lem = [ig.normalize_lemma(x) for x in (rec.get("lemmas") or [])]
            # Their tags are UD bundles; the reference is DCS cng. With a
            # validated bridge the bundles are translated INTO cng, so all
            # three (reference, cuSHR, external) are compared in one space and
            # the reference is never weakened. Without one, a sentinel that can
            # never equal g_cng keeps the M columns honestly unscored rather
            # than silently plausible.
            if tag_bridge:
                p_cng = [tag_bridge.get(t, "<unmapped:" + t + ">")
                         for t in (rec.get("tags") or [])]
            else:
                p_cng = ["<unmapped>"] * len(p_form)
            score(p_form, p_lem, p_cng, g_form, g_lem, g_cng, ext)
            n_ext += 1
        if missing:
            print(f"warning: {missing:,} sentences absent from {args.pred_jsonl}")

    print(f"\nsentence-level perfect match over {n:,} sentences "
          f"(all scored, gold path or not)")
    byt5 = {"S": 84.61, "L": 79.88, "S+M": 63.86, "L+M": 62.00, "S+L+M": 61.27}
    M_LEVELS = {"S+M", "L+M", "L+M(tol)", "S+L+M", "S+L+M(tol)"}
    if n_ext:
        print(f"{'level':<12}{'cuSHR':>8}{'ORACLE':>9}{args.pred_name:>10}"
              f"      ByT5 paper (Sen, DCS 2024)")
    else:
        print(f"{'level':<12}{'cuSHR':>8}{'ORACLE':>9}      ByT5-Sanskrit (Sen, DCS 2024)")
    for k in LEVELS:
        b_ = byt5.get(k)
        o = f"{100 * oracle[k] / n_oracle:8.2f}" if n_oracle else "       --"
        line = f"  {k:<10}{100 * hit[k] / n:7.2f}{o}"
        if n_ext:
            # Any level involving M is not scoreable for the external system
            # until the tag bridge exists; print n/a rather than a real-looking 0.
            line += ("       n/a" if (k in M_LEVELS and not ext_tags_usable)
                     else f"{100 * ext[k] / n_ext:10.2f}")
        line += (f"      {b_:>6.2f}" if b_ is not None else "         --")
        print(line)
    if n_ext:
        print(f"\n  {args.pred_name}: {n_ext:,} sentences from "
              f"{os.path.basename(args.pred_jsonl)}, scored against the SAME "
              f"external reference\n  and the same score() as cuSHR -- this "
              f"column IS a head-to-head.")
        if not ext_tags_usable:
            print(f"  M levels are n/a: {args.pred_name} emits full UD feature "
                  f"bundles\n  (Case=Nom|Gender=Masc|Number=Sing) while the "
                  f"reference is DCS cng integers\n  (29, -153). Scoring them "
                  f"directly would read as total tagging failure rather\n  than "
                  f"an unmapped vocabulary. NOTE the observed output is NOT the "
                  f"compressed\n  SNM codes of the paper's figures, so "
                  f"sanskrit_tags.tsv is not needed to bridge it.")
    print(f"\n  ORACLE = our own gold path scored against the same external "
          f"reference, over the {n_oracle:,} sentences that have one.")
    print("  It is a CEILING, and it is not 100: DCS lemmatises participles to "
          "the verbal\n  root where SHR gives the participial stem, so L / L+M / "
          "S+L+M measure\n  convention agreement as much as model quality. Read "
          "every cuSHR number\n  against its oracle, not against ByT5.")
    print("  (tol) = lemma matched through ingest.lemma_candidates; upper bound.")
    print("  'ByT5 paper' is a DIFFERENT CORPUS (DCS April-2024, 8,398 test), "
          "shown for\n  category alignment only. It is NOT the --pred-jsonl "
          "column, which is measured\n  here on these 4,200 sentences.")
    if n_ext:
        print(f"\n  CAVEAT on the {args.pred_name} S column: the released "
              f"chronbmm/sanskrit5-multitask\n  is trained on DCS compound "
              f"conventions, while this reference uses SIGHUM's.\n  Most of its "
              f"segmentation errors are compound-boundary disagreements\n"
              f"  (droRaputraH vs droRa putraH, alOlyena vs a lOlyena), not "
              f"failures to\n  segment. Their published 93.83 on SIGHUM comes "
              f"from a model fine-tuned ON\n  SIGHUM; this is the off-the-shelf "
              f"multitask checkpoint, so the S column\n  understates it. L is "
              f"unaffected -- both sides use DCS lemma conventions.")

    res = [int(s) for s in sids if has_gold[int(s)]]
    print(f"\n  ingest resolved a gold path for {len(res):,} / {len(sids):,} "
          f"of these sentences")

    if args.dump:
        json.dump(dump, open(args.dump, "w"), ensure_ascii=False)
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
