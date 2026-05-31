"""
diagnose_gold_match.py

Runs gold-path matching on a sample of sentences and categorizes every miss
so we can figure out *why* matches are failing. Three buckets:

  chunk_missing : .p file references a chunk_no that no graphml node has.

  lemma_mismatch: chunk exists but no node in that chunk has any of the
                  candidate (lemma, cng) pairs we generate from this gold
                  entry.

  cng_mismatch  : chunk + at least one candidate lemma form both exist
                  but the cng disagrees. Often a real DCS-vs-SHR encoding
                  disagreement (cng=1 compound stem vs decomposed nom)
                  rather than something fixable with normalization.

This script uses the SAME matching function (lemma_cng_candidates) as
ingest.py, so its reported miss rate matches ingest's actual behavior.
When you add a new alias in ingest.py, the diagnostic immediately sees it.

Run from cushr-bootcamp/dataset/:
    python diagnose_gold_match.py
"""

import os
import sys
import glob
import pickle
import json
import random
from collections import Counter, defaultdict

import networkx as nx

# Pull the candidate-generation logic from ingest.py so the two scripts
# can never drift out of sync.
from ingest import (
    DCS,
    normalize_lemma,
    lemma_candidates,
    lemma_cng_candidates,
)  # noqa: F401

# Ensure pickle can resolve __main__.DCS even though we imported it.
sys.modules['__main__'].DCS = DCS


def diagnose(graphml_dir, p_dir, sample_size=500,
             out_json='gold_diagnosis.json'):
    random.seed(0)
    paths = sorted(glob.glob(os.path.join(graphml_dir, '*.graphml')))
    if not paths:
        print(f"error: no graphmls in {graphml_dir}")
        return
    sample = random.sample(paths, min(sample_size, len(paths)))
    print(f"sampling {len(sample)} sentences from {len(paths)} total\n")

    failure_counts = Counter()
    examples = defaultdict(list)

    # bonus: track patterns
    unmatched_lemma_counts = Counter()
    cng_swap_patterns = Counter()

    total_triples = 0
    total_matched = 0
    sentences_with_p = 0

    for gml_path in sample:
        stem = os.path.splitext(os.path.basename(gml_path))[0]
        p_path = os.path.join(p_dir, f"{stem}.p")
        if not os.path.exists(p_path):
            continue

        try:
            with open(p_path, 'rb') as f:
                dcs = pickle.load(f, encoding='utf-8')
        except Exception as e:
            print(f"  failed to load {p_path}: {e}")
            continue

        try:
            G = nx.read_graphml(gml_path)
        except Exception:
            continue

        sentences_with_p += 1

        # ----- build per-sentence indices over the graphml -----
        # triple_idx: (chunk, lemma, cng) -> [nodes]    (for primary match)
        # pair_idx_by_chunk_lemma: (chunk, lemma) -> [cng, ...]  (for cng miss inspection)
        # chunk_idx: chunk -> [(lemma, cng, node)]      (for lemma miss inspection)
        triple_idx = defaultdict(list)
        pair_idx_by_chunk_lemma = defaultdict(list)
        chunk_idx = defaultdict(list)
        all_chunks = set()

        for n in G.nodes():
            d = G.nodes[n]
            chunk = str(d.get('chunk_no', ''))
            lemma = str(d.get('lemma', ''))
            cng = str(d.get('cng', ''))
            triple_idx[(chunk, lemma, cng)].append(n)
            pair_idx_by_chunk_lemma[(chunk, lemma)].append(cng)
            chunk_idx[chunk].append((lemma, cng, n))
            all_chunks.add(chunk)

        # ----- iterate over gold triples from .p -----
        lemmas = getattr(dcs, 'lemmas', None)
        cngs = getattr(dcs, 'cng', None)
        if lemmas is None or cngs is None:
            continue

        for chunk_no_p, (lem_list, cng_list) in enumerate(zip(lemmas, cngs),
                                                          start=1):
            for lem_raw, cng_raw in zip(lem_list, cng_list):
                total_triples += 1
                chunk_str = str(chunk_no_p)
                cng_str = str(cng_raw)

                # Use the exact same pair-based matching as ingest.py
                cand_pairs = lemma_cng_candidates(lem_raw, cng_str)

                matched = False
                for (l_cand, c_cand) in cand_pairs:
                    if (chunk_str, l_cand, c_cand) in triple_idx:
                        matched = True
                        break
                if matched:
                    total_matched += 1
                    continue

                # categorize the miss
                lem_norm = normalize_lemma(lem_raw)
                cands_lemma_only = lemma_candidates(lem_raw)

                if chunk_str not in all_chunks:
                    cat = 'chunk_missing'
                    detail = {
                        'sent_id': stem,
                        'expected_chunk': chunk_str,
                        'available_chunks': sorted(
                            all_chunks,
                            key=lambda x: int(x) if x.isdigit() else -1),
                        'lemma': lem_raw,
                        'lemma_norm': lem_norm,
                        'candidates': sorted(cands_lemma_only),
                        'cng': cng_str,
                    }
                else:
                    # Did any candidate lemma form appear in this chunk
                    # (under any cng)? If so it's a cng_mismatch.
                    matching_lemma_entries = []
                    for cand in cands_lemma_only:
                        for c_avail in pair_idx_by_chunk_lemma.get(
                                (chunk_str, cand), ()):
                            matching_lemma_entries.append((cand, c_avail))

                    if matching_lemma_entries:
                        cat = 'cng_mismatch'
                        available_cngs = [c for _, c in matching_lemma_entries]
                        detail = {
                            'sent_id': stem,
                            'chunk': chunk_str,
                            'lemma': lem_raw,
                            'lemma_norm': lem_norm,
                            'matched_candidate_forms': sorted(
                                {l for l, _ in matching_lemma_entries}),
                            'expected_cng': cng_str,
                            'available_cngs': available_cngs,
                        }
                        cng_swap_patterns[(cng_str,
                                           tuple(sorted(set(available_cngs))))] += 1
                    else:
                        cat = 'lemma_mismatch'
                        detail = {
                            'sent_id': stem,
                            'chunk': chunk_str,
                            'expected_lemma_raw': lem_raw,
                            'expected_lemma_norm': lem_norm,
                            'candidates_tried': sorted(cands_lemma_only),
                            'expected_cng': cng_str,
                            'available_in_chunk': [
                                {'lemma': l, 'cng': c}
                                for l, c, _ in chunk_idx[chunk_str]
                            ],
                            'available_lemmas_unique': sorted(
                                {l for l, _, _ in chunk_idx[chunk_str]}),
                        }
                        unmatched_lemma_counts[(lem_raw, lem_norm)] += 1

                failure_counts[cat] += 1
                if len(examples[cat]) < 30:
                    examples[cat].append(detail)

    # ----- report -----
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"sentences sampled:        {len(sample)}")
    print(f"sentences with .p file:   {sentences_with_p}")
    print(f"total gold triples seen:  {total_triples}")
    print(f"matched:                  {total_matched} "
          f"({100 * total_matched / max(total_triples, 1):.2f}%)")
    print()
    print("failure breakdown:")
    for cat, n in failure_counts.most_common():
        pct = 100 * n / max(total_triples, 1)
        print(f"  {cat:18s}  {n:6d}  ({pct:.2f}%)")
    print()

    print("top 20 unmatched lemmas (raw -> normalized):")
    for (raw, norm), n in unmatched_lemma_counts.most_common(20):
        print(f"  {n:4d}  {raw!r}  ->  {norm!r}")
    print()

    print("top 10 cng-mismatch patterns (expected -> available):")
    for (exp, avail), n in cng_swap_patterns.most_common(10):
        print(f"  {n:4d}  expected {exp:>6s}  available {list(avail)}")
    print()

    # save examples
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(
            {k: v for k, v in examples.items()},
            f, indent=2, ensure_ascii=False
        )
    print(f"detailed examples written to {out_json}")
    print("recommended: open it and skim ~5 examples per bucket. patterns will be obvious.")
    print("to fix new patterns: add to LEMMA_ALIASES / LEMMA_CNG_ALIASES /")
    print("variant helpers in ingest.py.")


if __name__ == '__main__':
    GRAPHML_DIR = "../../SIGHUM_database/After_graphml"
    P_DIR = "../../SIGHUM_database_gold_path/DCS_pick"
    diagnose(GRAPHML_DIR, P_DIR, sample_size=500)