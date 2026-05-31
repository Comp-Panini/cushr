import networkx as nx
import numpy as np
import os
import glob
import sys
import pickle
from collections import defaultdict


# ---------------------------------------------------------------------------
# DCS pickle support
# ---------------------------------------------------------------------------
# The .p files are Python 2 pickles of a class named "DCS" that lives in
# __main__ in the original DCS tooling. We stub it here so pickle.load can
# reconstruct the object. We then read its attributes directly.
class DCS(object):
    pass


sys.modules['__main__'].DCS = DCS


# ---------------------------------------------------------------------------
# Lemma normalization
# ---------------------------------------------------------------------------
# IAST multi-char sequences -> SLP1 single chars.
# Must run BEFORE the single-char translate.
IAST_DIGRAPHS = [
    ('kh', 'K'), ('gh', 'G'), ('ch', 'C'), ('jh', 'J'),
    ('ṭh', 'W'), ('ḍh', 'Q'),
    ('th', 'T'), ('dh', 'D'),
    ('ph', 'P'), ('bh', 'B'),
    ('ai', 'E'), ('au', 'O'),
]

IAST_TO_SLP = str.maketrans({
    'ā': 'A', 'ī': 'I', 'ū': 'U',
    'ṛ': 'f', 'ṝ': 'F', 'ḷ': 'x', 'ḹ': 'X',
    'ṅ': 'N', 'ñ': 'Y', 'ṭ': 'w', 'ḍ': 'q', 'ṇ': 'R',
    'ś': 'S', 'ṣ': 'z', 'ḥ': 'H', 'ṃ': 'M',
    "'": '',
})


def normalize_lemma(s):
    if s is None:
        return ''
    for src, dst in IAST_DIGRAPHS:
        s = s.replace(src, dst)
    return s.translate(IAST_TO_SLP)


# ---------------------------------------------------------------------------
# Lemma candidate generation
# ---------------------------------------------------------------------------
# The gold (.p) files and the graphml files come from two different
# lemmatization pipelines (DCS vs SHR). They agree most of the time but
# diverge in predictable ways. Instead of trying to canonicalize to one
# single form, we generate a small set of plausible graphml-side strings
# for each gold lemma and try each one.
#
# IMPORTANT: alias keys are matched AFTER normalize_lemma, so they must
# be written in SLP1 (e.g. 'uB' not 'ubh').

# Compound-prefix / stem aliases where ONLY the lemma differs.
# (For cases where the cng also differs, use LEMMA_CNG_ALIASES below.)
LEMMA_ALIASES = {
    'mahA':   ['mahant'],   # mahā- compound prefix vs canonical mahant stem
    'tva':    ['tvad'],     # 2nd-person pronoun stem with/without final d
    'Bavant': ['Bavat'],    # present participle: -ant vs -at
    'uB':     ['uBa'],      # 'both' stem (was 'ubh' as raw; SLP1 is 'uB')
    'nAma':   ['nAman'],    # n-stem noun with/without final n
    'ityAdi': ['iti'],      # "and so on" compound vs base particle
}


# (gold_lemma_norm, gold_cng_str) -> [(graphml_lemma, graphml_cng_str), ...]
# Use ONLY when both the lemma AND the cng differ between conventions for
# what is unambiguously the same surface token. The case here represents
# DCS treating a form as a bound compound stem (cng=2) while SHR treats
# the same surface token as a decomposed nominative (cng=3) — same node
# in the graph, different morphological label by convention.
#
# Be conservative adding to this. cng=1 compound-encoding disagreements
# (sentences like 6991, 16766) are NOT safe to add — those are real
# decompositional disagreements where the gold path implies a different
# segmentation than the graphml represents.
LEMMA_CNG_ALIASES = {
    ('mahA', '2'): [('mahant', '3')],
}


def _retroflex_variants(s):
    """R↔n, z↔s, M↔m, and the anusvara-before-dental case (M+d → nd).

    Covers gold-side 'praRam' vs graphml-side 'pranam', 'aBizic' vs
    'aBisic', 'ariMdama' vs 'arindama', etc.
    """
    out = {s}
    out.add(s.replace('R', 'n').replace('z', 's').replace('M', 'm'))
    out.add(s.replace('Md', 'nd').replace('Mt', 'nt').replace('MD', 'nD'))
    out.add(s.replace('Mb', 'mb').replace('Mp', 'mp'))
    return out


def _vowel_variants(s):
    """Long vowels stored as digraphs in some graphml files (A vs aa)."""
    out = {s}
    out.add(s.replace('A', 'aa'))
    out.add(s.replace('I', 'ii'))
    out.add(s.replace('U', 'uu'))
    return out


def _guna_variants(s):
    """Sanskrit vowel-grade alternation between zero-grade and guna.

    Roots alternate: bhū↔bho, śuṣ↔śoṣ, vid↔ved, etc. In SLP1 this is
    u↔o, i↔e (and au↔ū / ai↔ī represented as O↔U / E↔I).

    Combined with _suffix_variants (which strips causative -ay), this
    catches causative gold forms like 'Sozay' (śoṣay) -> 'Suz' (śuṣ),
    'niveday' -> 'nivid', etc.
    """
    out = {s}
    # zero-grade -> guna
    out.add(s.replace('u', 'o').replace('U', 'O'))
    out.add(s.replace('i', 'e').replace('I', 'E'))
    # guna -> zero-grade
    out.add(s.replace('o', 'u').replace('O', 'U'))
    out.add(s.replace('e', 'i').replace('E', 'I'))
    return out


def _suffix_variants(s):
    """Strip causative -ay (with optional epenthetic consonant) and a few
    pronoun/participle final consonants.

    Causative formant patterns:
      - root + ay:           vAray <- vAr,   prakIrtay <- prakIrt
      - root + p/v + ay:     dApay <- dA,    BAvay <- BA
    """
    out = {s}
    if s.endswith('ay') and len(s) > 3:
        out.add(s[:-2])                       # vAray -> vAr
        # epenthetic-consonant causative: dA + p + ay, BA + v + ay
        if len(s) > 4 and s[-3] in 'pv':
            out.add(s[:-3])                   # dApay -> dA, BAvay -> BA
    if s.endswith('ant') and len(s) > 4:
        out.add(s[:-1])                       # Bavant -> Bavat
    if s.endswith('d') and len(s) > 2:
        out.add(s[:-1])                       # tvad -> tva
    if s.endswith('n') and len(s) > 3:
        out.add(s[:-1])                       # rAjan -> rAja
    return out


def lemma_candidates(raw):
    """Return the set of plausible graphml-side lemma forms for one gold lemma.

    Generation is layered: base SLP1 -> retroflex/vowel/suffix/guna
    variants -> one round of cross-application -> alias substitution.
    Cross-apply catches things like 'Sozay' which needs BOTH the -ay
    strip AND o->u guna reduction to land on 'Suz'.
    """
    base = normalize_lemma(raw)
    cands = {base}
    cands |= _retroflex_variants(base)
    cands |= _vowel_variants(base)
    cands |= _suffix_variants(base)
    cands |= _guna_variants(base)

    # one round of cross-application (small set, cheap)
    expanded = set()
    for c in list(cands):
        expanded |= _retroflex_variants(c)
        expanded |= _vowel_variants(c)
        expanded |= _suffix_variants(c)
        expanded |= _guna_variants(c)
    cands |= expanded

    # lemma-only aliases — applied last so they pick up suffix-stripped forms
    for c in list(cands):
        for alias in LEMMA_ALIASES.get(c, []):
            cands.add(alias)

    return cands


def lemma_cng_candidates(raw_lemma, cng_str):
    """Return set of (graphml_lemma, graphml_cng) pairs to match this gold
    (lemma, cng) entry against.

    Combines the lemma-only candidate set (paired with the gold cng) with
    any (lemma, cng) alias entries that translate both fields together.
    """
    pairs = {(l, cng_str) for l in lemma_candidates(raw_lemma)}
    base = normalize_lemma(raw_lemma)
    for alt_lemma, alt_cng in LEMMA_CNG_ALIASES.get((base, cng_str), ()):
        pairs.add((alt_lemma, alt_cng))
    return pairs


# ---------------------------------------------------------------------------
# Gold-path loading
# ---------------------------------------------------------------------------
def load_gold_path(p_filepath):
    """
    Load a DCS .p file and return a list of (chunk_str, candidate_pairs)
    tuples, where candidate_pairs is a frozenset of (lemma_in_graphml,
    cng_in_graphml) tuples that should be considered a gold match.

    Returns None if the file is missing or unparseable.
    """
    if not os.path.exists(p_filepath):
        return None
    try:
        with open(p_filepath, 'rb') as f:
            dcs = pickle.load(f, encoding='utf-8')
    except Exception as e:
        print(f"  warning: failed to load {p_filepath}: {e}")
        return None

    lemmas = getattr(dcs, 'lemmas', None)
    cngs = getattr(dcs, 'cng', None)
    if lemmas is None or cngs is None:
        return None

    triples = []
    for chunk_idx, (lem_list, cng_list) in enumerate(zip(lemmas, cngs), start=1):
        # The chunk_no in the graphml is 1-indexed by DCS convention.
        for lem, cng in zip(lem_list, cng_list):
            triples.append((
                str(chunk_idx),
                frozenset(lemma_cng_candidates(lem, str(cng))),
            ))
    return triples


# ---------------------------------------------------------------------------
# Main ingest
# ---------------------------------------------------------------------------
def process_corpus(graphml_dir, p_dir, output_filename="cushr_data_full_with_gold.npz"):
    print(f"scanning graphml directory: {graphml_dir}")
    print(f"scanning gold-path directory: {p_dir}")
    filepaths = sorted(glob.glob(os.path.join(graphml_dir, "*.graphml")))

    if not filepaths:
        print("error: no .graphml files found, check directory path")
        sys.exit(1)

    print(f"found {len(filepaths)} sentences, beginning ingest process...")

    # global arrays
    global_node_features = []
    global_rowptr = [0]
    global_colidx = []
    global_topolevel = []
    global_goldpathmask = []
    sentence_offsets = [0]
    current_global_node_index = 0

    # dynamic vocabulary for morphological tags
    morph_vocab = defaultdict(lambda: len(morph_vocab))
    morph_vocab["UNKNOWN"] = 0

    # gold-path bookkeeping
    n_with_pfile = 0
    n_without_pfile = 0
    total_gold_expected = 0
    total_gold_matched = 0
    fully_matched_sentences = 0

    for idx, filepath in enumerate(filepaths):
        if idx % 1000 == 0 and idx > 0:
            print(f"processed {idx} / {len(filepaths)} sentences  "
                  f"(gold match rate so far: "
                  f"{total_gold_matched}/{total_gold_expected})")

        # ----- locate matching .p file by sent_id (filename stem) -----
        stem = os.path.splitext(os.path.basename(filepath))[0]
        p_filepath = os.path.join(p_dir, f"{stem}.p")
        gold_triples = load_gold_path(p_filepath)
        if gold_triples is None:
            n_without_pfile += 1
        else:
            n_with_pfile += 1

        # ----- parse XML graph -----
        try:
            G = nx.read_graphml(filepath)
        except Exception as e:
            print(f"skipping corrupted file {filepath}: {e}")
            continue

        # ----- force DAG: drop conflict edges and any non-forward edges -----
        edges_to_remove = []
        for u, v, data in G.edges(data=True):
            u_id = int(u[1:]) if str(u).startswith('n') else int(u)
            v_id = int(v[1:]) if str(v).startswith('n') else int(v)
            if str(data.get('key', '0')) != '1' or u_id >= v_id:
                edges_to_remove.append((u, v))
        G.remove_edges_from(edges_to_remove)

        # ----- consistent node ordering -----
        try:
            node_list = sorted(
                G.nodes(),
                key=lambda x: int(x[1:]) if x.startswith('n') else int(x)
            )
        except ValueError:
            node_list = list(G.nodes())

        num_nodes = len(node_list)
        if num_nodes == 0:
            continue

        local_id_map = {node_id: i for i, node_id in enumerate(node_list)}

        # ----- cache per-node attribute dicts once (avoids double-lookup later) -----
        node_attrs = [G.nodes[n] for n in node_list]

        # ----- topological levels -----
        topo_levels = {n: 0 for n in node_list}
        try:
            topo_sorted = list(nx.topological_sort(G))
            for n in topo_sorted:
                for child in G.successors(n):
                    topo_levels[child] = max(topo_levels[child],
                                             topo_levels[n] + 1)
        except nx.NetworkXUnfeasible:
            print(f"warning: cycle detected in {filepath} after edge filter, skipping.")
            continue

        # ----- build per-node gold mask -----
        # Index nodes by full (chunk, lemma, cng) triple. The candidate set
        # for each gold entry is a small set of (lemma, cng) pairs; for
        # each pair we do a single dict lookup.
        local_gold_mask = [0] * num_nodes
        if gold_triples is not None:
            triple_idx = defaultdict(list)
            for i, attrs in enumerate(node_attrs):
                key = (str(attrs.get('chunk_no', '')),
                       str(attrs.get('lemma', '')),
                       str(attrs.get('cng', '')))
                triple_idx[key].append(i)

            sentence_matched = 0
            for chunk_str, cand_pairs in gold_triples:
                hit_any = False
                for (l_cand, c_cand) in cand_pairs:
                    for local_id in triple_idx.get((chunk_str, l_cand, c_cand), ()):
                        local_gold_mask[local_id] = 1
                        hit_any = True
                if hit_any:
                    sentence_matched += 1

            total_gold_expected += len(gold_triples)
            total_gold_matched += sentence_matched
            if sentence_matched == len(gold_triples):
                fully_matched_sentences += 1

        # ----- extract node features, topo, gold -----
        for i, n in enumerate(node_list):
            raw_morph = node_attrs[i].get('morph', 'UNKNOWN')
            morph_id = morph_vocab[raw_morph]
            global_node_features.append(morph_id)
            global_goldpathmask.append(local_gold_mask[i])
            global_topolevel.append(topo_levels[n])

        # ----- CSR edges with global index shifting -----
        local_rowptr = [0] * (num_nodes + 1)
        local_colidx = []
        edge_count = 0
        for i, n in enumerate(node_list):
            for target in G.successors(n):
                target_local_id = local_id_map[target]
                target_global_id = target_local_id + current_global_node_index
                local_colidx.append(target_global_id)
                edge_count += 1
            local_rowptr[i + 1] = edge_count

        global_colidx.extend(local_colidx)

        current_total_edges = global_rowptr[-1]
        for val in local_rowptr[1:]:
            global_rowptr.append(val + current_total_edges)

        # ----- update sentence offsets -----
        current_global_node_index += num_nodes
        sentence_offsets.append(current_global_node_index)

    # -----------------------------------------------------------------------
    # write archive
    # -----------------------------------------------------------------------
    print("\nextraction complete! packaging into NumPy archive...")
    np.savez_compressed(
        output_filename,
        node_features=np.array(global_node_features, dtype=np.int32),
        rowptr=np.array(global_rowptr, dtype=np.int32),
        colidx=np.array(global_colidx, dtype=np.int32),
        topolevel=np.array(global_topolevel, dtype=np.int32),
        sentenceoffsets=np.array(sentence_offsets[:-1], dtype=np.int32),
        goldpathmask=np.array(global_goldpathmask, dtype=np.int8),
    )

    with open("morph_vocabulary.txt", "w", encoding="utf-8") as f:
        for tag, tag_id in sorted(morph_vocab.items(), key=lambda x: x[1]):
            f.write(f"{tag_id}\t{tag}\n")

    # -----------------------------------------------------------------------
    # summary
    # -----------------------------------------------------------------------
    print(f"\nsuccess! artifact saved as: {output_filename}")
    print(f"total nodes:            {len(global_node_features):,}")
    print(f"total edges:            {len(global_colidx):,}")
    print(f"total sentences:        {len(sentence_offsets)-1:,}")
    print(f"unique morph tags:      {len(morph_vocab):,}")
    print()
    print("--- gold-path coverage ---")
    print(f"sentences with .p file:    {n_with_pfile:,}")
    print(f"sentences without .p file: {n_without_pfile:,}")
    if total_gold_expected > 0:
        rate = 100.0 * total_gold_matched / total_gold_expected
        print(f"gold (lemma, cng) entries matched: "
              f"{total_gold_matched:,} / {total_gold_expected:,}  ({rate:.2f}%)")
        print(f"sentences with every entry matched: {fully_matched_sentences:,}")
    print(f"total gold nodes flagged: {int(np.sum(global_goldpathmask)):,}")


if __name__ == "__main__":
    GRAPHML_DIR = "../../SIGHUM_database/After_graphml"
    P_DIR = "../../SIGHUM_database_gold_path/DCS_pick"

    process_corpus(
        GRAPHML_DIR,
        P_DIR,
        output_filename="cushr_data_full_with_gold.npz",
    )