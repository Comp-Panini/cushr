#!/usr/bin/env python3
"""K-best candidates as SEGMENTATIONS, shared by make_rerank_data.py and
eval_surface.py.

A lattice node is (form, lemma, cng), so the raw K-best list is dominated by
morphological variants of one segmentation: the same words with a different
lemma or tag. For a segmentation reranker those are the same answer, and they
crowd distinct splits out of the beam. `dedup` keeps the best-scoring node path
per distinct surface sequence, so K candidates means K different segmentations.

A candidate's surface identity is its sequence of (char_start, form id): the
same form at the same start spans the same characters.
"""


def clean(loc, global_node, cstart, fid, forms):
    """Local node ids of one path -> global ids in reading order, boundary and
    empty-form nodes dropped. Same convention as eval_slm / make_rerank_data."""
    seq = sorted((int(global_node[x]) for x in loc), key=lambda g: int(cstart[g]))
    return [g for g in seq if forms[fid[g]]]


def surface_key(seq, cstart, fid):
    return tuple((int(cstart[g]), int(fid[g])) for g in seq)


def dedup(seqs, scores, cstart, fid, k):
    """First (= best-scoring, since k-best scores are non-increasing) path per
    distinct segmentation, at most k of them."""
    out_s, out_c, seen = [], [], set()
    for seq, sc in zip(seqs, scores):
        key = surface_key(seq, cstart, fid)
        if key in seen:
            continue
        seen.add(key)
        out_s.append(seq)
        out_c.append(sc)
        if len(out_s) == k:
            break
    return out_s, out_c


def sentence_candidates(cands_i, scores_i, global_node, cstart, fid, forms,
                        k, dedup_surface):
    """One sentence's k-best list -> (seqs, scores), deduplicated if asked."""
    seqs = [clean(loc, global_node, cstart, fid, forms) for loc in cands_i]
    scores = [float(scores_i[j]) for j in range(len(seqs))]
    if dedup_surface:
        return dedup(seqs, scores, cstart, fid, k)
    return seqs[:k], scores[:k]
