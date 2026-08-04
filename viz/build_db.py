"""
Preprocess a compact demo subset of SIGHUM sentences into a single SQLite file
that the visualizer ships with. The advisor never needs the raw corpus — the
deployed app reads only this .db.

For each npz sentence index 0..N-1 we read the matching .graphml, export the
segmentation DAG (every candidate word + the transitions between them), and
resolve the gold path with the SAME logic ingest.py uses, so the highlight in
the visualizer matches the decoder's gold path exactly.

Usage:
    python build_db.py [--n 2000] [--out cushr_viz.db]
"""
import os
import sys
import json
import zlib
import argparse
import sqlite3

import networkx as nx
from collections import defaultdict

# Reuse the gold-path reconstruction from the ingest pipeline so the visualizer
# and the C++ decoder agree on what "the gold path" is.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ingest'))
import ingest as ig  # noqa: E402

GRAPHML_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'SIGHUM_database', 'After_graphml')
P_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'SIGHUM_database_gold_path', 'DCS_pick')
INDEX_JSON = os.path.join(os.path.dirname(__file__), '..', 'ingest', 'sentence_index.json')


def build_sentence(stem, edge_order="id"):
    """Return a compact dict {nodes, edges, gold_path} for one sentence, or None."""
    fp = os.path.join(GRAPHML_DIR, f'{stem}.graphml')
    if not os.path.exists(fp):
        return None
    try:
        G = nx.read_graphml(fp)
    except Exception:
        return None

    # Same DAG filtering as ingest. Must match the --edge-order the npz this db
    # accompanies was built with, or node numbering and edges disagree with it.
    ig.forward_edge_filter(G, edge_order)

    try:
        node_list = sorted(G.nodes(), key=lambda x: int(x[1:]) if x.startswith('n') else int(x))
    except ValueError:
        node_list = list(G.nodes())
    if not node_list:
        return None

    lid = {n: i for i, n in enumerate(node_list)}
    attrs = [G.nodes[n] for n in node_list]
    nw = len(node_list)

    # word adjacency / sources / sinks (local word space)
    succ = {i: set() for i in range(nw)}
    for i, n in enumerate(node_list):
        for t in G.successors(n):
            succ[i].add(lid[t])
    has_in = set()
    for i in succ:
        has_in |= succ[i]
    sources = {i for i in range(nw) if i not in has_in}
    sinks = {i for i in range(nw) if not succ[i]}

    # topo levels (for layered layout)
    topo = {n: 0 for n in node_list}
    try:
        for n in nx.topological_sort(G):
            for c in G.successors(n):
                topo[c] = max(topo[c], topo[n] + 1)
    except nx.NetworkXUnfeasible:
        return None

    # resolve gold path -> local word ids -> graphml node ids
    gold_ids = []
    ge = ig.load_gold_entries(os.path.join(P_DIR, f'{stem}.p'))
    if ge is not None:
        ti = defaultdict(list)
        for i, a in enumerate(attrs):
            ti[(str(a.get('chunk_no', '')), str(a.get('lemma', '')), str(a.get('cng', '')))].append(i)
        gold_local = ig.reconstruct_gold_path(ge, ti, succ, sources, sinks)
        gold_ids = [node_list[i] for i in gold_local]
    gold_set = set(gold_ids)

    nodes = []
    for i, n in enumerate(node_list):
        a = attrs[i]
        nodes.append({
            'id': n,
            'word': a.get('word', ''),
            'lemma': a.get('lemma', ''),
            'morph': a.get('morph', ''),
            'cng': a.get('cng', ''),
            'chunk': a.get('chunk_no', ''),
            'level': topo[n],
            'gold': n in gold_set,
        })
    edges = []
    for i, n in enumerate(node_list):
        for t in sorted(succ[i]):
            edges.append({'source': n, 'target': node_list[t]})

    return {'stem': stem, 'nodes': nodes, 'edges': edges, 'gold_path': gold_ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=0,
                    help='number of sentences (npz indices 0..n-1); 0 = all')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'cushr_viz.db'))
    ap.add_argument('--edge-order', choices=('id', 'position'), default='id',
                    help='must match the --edge-order the companion npz was '
                         'ingested with; see ingest.forward_edge_filter')
    args = ap.parse_args()

    if not os.path.exists(INDEX_JSON):
        sys.exit(f'missing {INDEX_JSON} — run ingest/ingest.py first to emit the sentence index map')
    with open(INDEX_JSON, encoding='utf-8') as f:
        sentence_index = json.load(f)

    n = len(sentence_index) if args.n <= 0 else min(args.n, len(sentence_index))
    print(f'building DB for npz indices 0..{n-1} -> {args.out}')

    if os.path.exists(args.out):
        os.remove(args.out)
    con = sqlite3.connect(args.out)
    # `data` holds zlib-compressed UTF-8 JSON (the lattice compresses ~5-10x,
    # which keeps the full-corpus DB small enough to ship in a free Space).
    con.execute('CREATE TABLE sentences (idx INTEGER PRIMARY KEY, stem TEXT, '
                'has_gold INTEGER, n_nodes INTEGER, data BLOB)')

    n_gold = 0
    for idx in range(n):
        if idx % 2000 == 0 and idx:
            print(f'  {idx}/{n}  ({n_gold} with gold)')
        s = build_sentence(sentence_index[idx], args.edge_order)
        if s is None:
            continue
        has_gold = 1 if s['gold_path'] else 0
        n_gold += has_gold
        blob = zlib.compress(json.dumps(s, ensure_ascii=False).encode('utf-8'), 9)
        con.execute('INSERT INTO sentences VALUES (?,?,?,?,?)',
                    (idx, s['stem'], has_gold, len(s['nodes']), blob))
        if idx % 5000 == 0:
            con.commit()
    con.commit()

    size_mb = os.path.getsize(args.out) / 1e6
    print(f'done: {n} sentences, {n_gold} with a gold path, {size_mb:.1f} MB')
    con.close()


if __name__ == '__main__':
    main()
