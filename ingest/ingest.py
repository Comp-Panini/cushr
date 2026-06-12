import networkx as nx
import numpy as np
import os
import glob
import sys
import json
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
# what is unambiguously the same surface token.
LEMMA_CNG_ALIASES = {
    ('mahA', '2'): [('mahant', '3')],
}


def _retroflex_variants(s):
    """R↔n, z↔s, M↔m, and the anusvara-before-dental case (M+d → nd)."""
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
    """Sanskrit vowel-grade alternation between zero-grade and guna."""
    out = {s}
    # zero-grade -> guna
    out.add(s.replace('u', 'o').replace('U', 'O'))
    out.add(s.replace('i', 'e').replace('I', 'E'))
    # guna -> zero-grade
    out.add(s.replace('o', 'u').replace('O', 'U'))
    out.add(s.replace('e', 'i').replace('E', 'I'))
    return out


def _suffix_variants(s):
    """Strip causative -ay and a few pronoun/participle final consonants."""
    out = {s}
    if s.endswith('ay') and len(s) > 3:
        out.add(s[:-2])                       # vAray -> vAr
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
    """Return the set of plausible graphml-side lemma forms for one gold lemma."""
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
    """Return set of (graphml_lemma, graphml_cng) pairs for this gold entry."""
    pairs = {(l, cng_str) for l in lemma_candidates(raw_lemma)}
    base = normalize_lemma(raw_lemma)
    for alt_lemma, alt_cng in LEMMA_CNG_ALIASES.get((base, cng_str), ()):
        pairs.add((alt_lemma, alt_cng))
    return pairs


# ---------------------------------------------------------------------------
# Gold-path loading
# ---------------------------------------------------------------------------
def load_gold_entries(p_filepath):
    """Load a DCS .p file and return the ordered list of gold words.

    Each gold word is a tuple (chunk_str, candidate_pairs) where
    candidate_pairs is a frozenset of (lemma_in_graphml, cng_in_graphml)
    tuples that should be considered a match. The list is FLAT and ORDERED:
    a chunk that decomposes into several words (e.g. śāstra+kovida) yields
    several consecutive entries that share the same chunk_str. This ordered
    sequence is exactly the gold path through the lattice.

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

    entries = []
    for chunk_idx, (lem_list, cng_list) in enumerate(zip(lemmas, cngs), start=1):
        # chunk_no in the graphml is 1-indexed by DCS convention.
        for lem, cng in zip(lem_list, cng_list):
            entries.append((
                str(chunk_idx),
                frozenset(lemma_cng_candidates(lem, str(cng))),
            ))
    return entries


def reconstruct_gold_path(gold_entries, triple_idx, succ, sources, sinks):
    """Resolve the ordered gold-word sequence to a single connected path.

    The gold annotation is a *path*: source-word, ..., sink-word, with each
    consecutive pair joined by a forward edge in the lattice. A per-node mask
    cannot express this — the same (lemma, cng) can land on several parallel
    nodes, which is why ~98% of sentences looked "ambiguous" before. Here we
    pick exactly one node per gold word so that:
      - word 0 is an original source (reachable from the super-source),
      - each word i is a forward-successor of word i-1,
      - the last word is an original sink (reaches the super-sink).

    Implemented as a small forward DP over gold positions with backpointers.

    Args:
        gold_entries : ordered [(chunk_str, candidate_pairs), ...]
        triple_idx   : {(chunk_str, lemma, cng): [local_word_id, ...]}
        succ         : {local_word_id: set(local_word_id successors)}
        sources      : set of local_word_ids with in-degree 0
        sinks        : set of local_word_ids with out-degree 0

    Returns the ordered list of local word ids, or [] if unresolvable.
    """
    if not gold_entries:
        return []

    # Candidate node list per gold position (deduped, sorted for determinism).
    cand = []
    for chunk_str, pairs in gold_entries:
        nodes = set()
        for (lem, cng) in pairs:
            nodes.update(triple_idx.get((chunk_str, lem, cng), ()))
        if not nodes:
            return []  # this gold word matched nothing -> path is broken
        cand.append(sorted(nodes))

    m = len(cand)
    # reach[i] = local id -> back-pointer (local id at i-1, or -1 for i==0)
    reach = [dict() for _ in range(m)]
    for n in cand[0]:
        if n in sources:
            reach[0][n] = -1
    if not reach[0]:
        return []

    for i in range(1, m):
        prev = reach[i - 1]
        for n in cand[i]:
            for p in prev:
                if n in succ.get(p, ()):  # forward edge p -> n
                    reach[i][n] = p
                    break
        if not reach[i]:
            return []

    # Final node must be an original sink (connects to super-sink).
    ends = [n for n in reach[m - 1] if n in sinks]
    if not ends:
        return []
    end = min(ends)  # deterministic

    # Backtrack.
    path = []
    i = m - 1
    n = end
    while i >= 0:
        path.append(n)
        n = reach[i][n]
        i -= 1
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Main ingest
# ---------------------------------------------------------------------------
def process_corpus(graphml_dir, p_dir, output_filename="cushr_data_full_with_gold.npz",
                   index_filename="sentence_index.json"):
    print(f"scanning graphml directory: {graphml_dir}")
    print(f"scanning gold-path directory: {p_dir}")
    filepaths = sorted(glob.glob(os.path.join(graphml_dir, "*.graphml")))

    if not filepaths:
        print("error: no .graphml files found, check directory path")
        sys.exit(1)

    print(f"found {len(filepaths)} sentences, beginning ingest process...")

    # global arrays
    global_node_features = []        # morph tag id per node
    global_node_length = []          # surface char length per node (0 for boundary)
    global_rowptr = [0]
    global_colidx = []
    global_topolevel = []
    global_goldpathmask = []
    sentence_offsets = [0]
    current_global_node_index = 0

    # explicit gold path (flat node ids + per-sentence CSR offsets)
    gold_path_flat = []
    gold_path_offsets = [0]

    # npz sentence index -> graphml filename stem (sorted glob order)
    sentence_index = []

    # dynamic vocabulary for morphological tags
    morph_vocab = defaultdict(lambda: len(morph_vocab))
    morph_vocab["UNKNOWN"] = 0
    BOUNDARY_ID = morph_vocab["BOUNDARY"]   # feature-less super source/sink tag

    # gold-path bookkeeping
    n_with_pfile = 0
    n_without_pfile = 0
    resolved_sentences = 0

    for idx, filepath in enumerate(filepaths):
        if idx % 1000 == 0 and idx > 0:
            print(f"processed {idx} / {len(filepaths)} sentences  "
                  f"(gold paths resolved so far: {resolved_sentences})")

        stem = os.path.splitext(os.path.basename(filepath))[0]
        sentence_index.append(stem)

        # ----- locate matching .p file by sent_id (filename stem) -----
        p_filepath = os.path.join(p_dir, f"{stem}.p")
        gold_entries = load_gold_entries(p_filepath)
        if gold_entries is None:
            n_without_pfile += 1
        else:
            n_with_pfile += 1

        # ----- parse XML graph -----
        try:
            G = nx.read_graphml(filepath)
        except Exception as e:
            print(f"skipping corrupted file {filepath}: {e}")
            sentence_index.pop()
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

        num_word = len(node_list)
        if num_word == 0:
            sentence_index.pop()
            continue

        local_id_map = {node_id: i for i, node_id in enumerate(node_list)}
        node_attrs = [G.nodes[n] for n in node_list]

        # ----- topological levels (over word nodes) -----
        word_topo = {n: 0 for n in node_list}
        try:
            for n in nx.topological_sort(G):
                for child in G.successors(n):
                    word_topo[child] = max(word_topo[child], word_topo[n] + 1)
        except nx.NetworkXUnfeasible:
            print(f"warning: cycle detected in {filepath} after edge filter, skipping.")
            sentence_index.pop()
            continue

        # ----- word-level adjacency / sources / sinks (in local word space) -----
        succ = {i: set() for i in range(num_word)}
        for i, n in enumerate(node_list):
            for tgt in G.successors(n):
                succ[i].add(local_id_map[tgt])
        has_in = set()
        for i in succ:
            has_in.update(succ[i])
        sources = {i for i in range(num_word) if i not in has_in}
        sinks = {i for i in range(num_word) if not succ[i]}

        # ----- resolve the gold path (list of local word ids) -----
        gold_local = []
        if gold_entries is not None:
            triple_idx = defaultdict(list)
            for i, attrs in enumerate(node_attrs):
                key = (str(attrs.get('chunk_no', '')),
                       str(attrs.get('lemma', '')),
                       str(attrs.get('cng', '')))
                triple_idx[key].append(i)
            gold_local = reconstruct_gold_path(
                gold_entries, triple_idx, succ, sources, sinks)
            if gold_local:
                resolved_sentences += 1

        # =====================================================================
        # Emit nodes for this sentence with an explicit super-source / super-sink.
        #
        # The raw SHR lattices have MANY start nodes and MANY end nodes (one per
        # candidate first/last word). Without a single source and sink the
        # decoder can only enumerate paths that happen to end at the last node id
        # — silently dropping ~92% of valid segmentations. We add a per-sentence
        # super-source (before all words) and super-sink (after all words) so that
        # "the path from source to sink" is well-defined and top-K is complete.
        #
        # Local layout:  0 = super-source, 1..num_word = words, num_word+1 = super-sink
        # =====================================================================
        SRC = 0
        SINK = num_word + 1
        num_nodes = num_word + 2
        max_word_level = max(word_topo.values()) if word_topo else 0

        # node features / length / topo / gold mask, in local-id order
        local_gold_mask = [0] * num_nodes
        gold_set = set(g + 1 for g in gold_local)   # shift to word-local+1 space
        for g in gold_set:
            local_gold_mask[g] = 1

        # super-source
        global_node_features.append(BOUNDARY_ID)
        global_node_length.append(0)
        global_topolevel.append(0)
        global_goldpathmask.append(0)
        # words
        for i, n in enumerate(node_list):
            raw_morph = node_attrs[i].get('morph', 'UNKNOWN')
            global_node_features.append(morph_vocab[raw_morph])
            try:
                wlen = int(node_attrs[i].get('length_word', 0))
            except (TypeError, ValueError):
                wlen = 0
            global_node_length.append(wlen)
            global_topolevel.append(word_topo[n] + 1)
            global_goldpathmask.append(local_gold_mask[i + 1])
        # super-sink
        global_node_features.append(BOUNDARY_ID)
        global_node_length.append(0)
        global_topolevel.append(max_word_level + 2)
        global_goldpathmask.append(0)

        # ----- edges (local), with super-source / super-sink wiring -----
        # adjacency: super-source -> every original source word
        #            word edges (shifted by +1)
        #            every original sink word -> super-sink
        local_adj = [[] for _ in range(num_nodes)]
        for i in sorted(sources):
            local_adj[SRC].append(i + 1)
        for i in range(num_word):
            for tgt in sorted(succ[i]):
                local_adj[i + 1].append(tgt + 1)
        for i in sorted(sinks):
            local_adj[i + 1].append(SINK)
        # SINK has no outgoing edges

        # build CSR for this sentence and shift to global ids
        edge_count = global_rowptr[-1]
        for local in range(num_nodes):
            for tgt_local in local_adj[local]:
                global_colidx.append(tgt_local + current_global_node_index)
                edge_count += 1
            global_rowptr.append(edge_count)

        # ----- explicit gold path as global ids (word nodes only) -----
        if gold_local:
            for g in gold_local:
                gold_path_flat.append(g + 1 + current_global_node_index)
        gold_path_offsets.append(len(gold_path_flat))

        # ----- advance sentence boundary -----
        current_global_node_index += num_nodes
        sentence_offsets.append(current_global_node_index)

    # -----------------------------------------------------------------------
    # write archive
    # -----------------------------------------------------------------------
    print("\nextraction complete! packaging into NumPy archive...")
    np.savez_compressed(
        output_filename,
        node_features=np.array(global_node_features, dtype=np.int32),
        node_length=np.array(global_node_length, dtype=np.int32),
        rowptr=np.array(global_rowptr, dtype=np.int32),
        colidx=np.array(global_colidx, dtype=np.int32),
        topolevel=np.array(global_topolevel, dtype=np.int32),
        sentenceoffsets=np.array(sentence_offsets[:-1], dtype=np.int32),
        goldpathmask=np.array(global_goldpathmask, dtype=np.int8),
        gold_path_nodes=np.array(gold_path_flat, dtype=np.int32),
        gold_path_offsets=np.array(gold_path_offsets, dtype=np.int32),
    )

    with open("morph_vocabulary.txt", "w", encoding="utf-8") as f:
        for tag, tag_id in sorted(morph_vocab.items(), key=lambda x: x[1]):
            f.write(f"{tag_id}\t{tag}\n")

    with open(index_filename, "w", encoding="utf-8") as f:
        json.dump(sentence_index, f)

    # -----------------------------------------------------------------------
    # summary
    # -----------------------------------------------------------------------
    n_sent = len(sentence_offsets) - 1
    print(f"\nsuccess! artifact saved as: {output_filename}")
    print(f"total nodes:            {len(global_node_features):,}")
    print(f"total edges:            {len(global_colidx):,}")
    print(f"total sentences:        {n_sent:,}")
    print(f"unique morph tags:      {len(morph_vocab):,}")
    print()
    print("--- gold-path coverage ---")
    print(f"sentences with .p file:    {n_with_pfile:,}")
    print(f"sentences without .p file: {n_without_pfile:,}")
    rate = 100.0 * resolved_sentences / n_sent if n_sent else 0.0
    print(f"sentences with a fully resolved gold path: "
          f"{resolved_sentences:,} / {n_sent:,}  ({rate:.2f}%)")
    print(f"total gold nodes flagged: {int(np.sum(global_goldpathmask)):,}")
    print(f"wrote sentence index map: {index_filename}")


if __name__ == "__main__":
    GRAPHML_DIR = "../../SIGHUM_database/After_graphml"
    P_DIR = "../../SIGHUM_database_gold_path/DCS_pick"

    process_corpus(
        GRAPHML_DIR,
        P_DIR,
        output_filename="cushr_data_full_with_gold.npz",
    )
