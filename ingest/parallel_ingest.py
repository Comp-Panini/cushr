#!/usr/bin/env python3
"""Ingest the corpus in parallel shards, then merge them into one archive.

Sequential ingest runs at ~17 sentences/s, so the 119,503-sentence corpus takes
close to two hours in a single process -- long enough that a run can be killed
before it writes anything, losing all of it. The work is embarrassingly
parallel (each graphml file is independent), so this splits the file list
across processes, has each write a self-contained shard, and merges the shards
afterwards. On 12 cores that turns ~110 minutes into ~10-15.

Sharding is by contiguous slices of the *sorted* file list, and shards are
merged in that same order, so the resulting sentence order -- and therefore the
md5-bucketed train/dev/test split, which is keyed on sentence index -- is
identical to a sequential run.

The only thing that cannot be done independently is vocabulary numbering: each
worker interns morph tags, forms, lemmas and preverbs into its own id space.
The merge builds a union vocabulary and remaps every shard's ids through it.

    python parallel_ingest.py --out ../data/raw.npz --workers 12
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import ingest

# Arrays that are simply concatenated, with no index fix-up.
PLAIN = ["node_features", "node_length", "topolevel", "goldpathmask",
         "node_position", "node_chunk", "node_form_id", "node_lemma_id",
         "node_preverb_id", "node_char_start"]

# (array name, vocabulary file) for the id columns that need remapping.
VOCAB_COLUMNS = [
    ("node_features", "morph_vocabulary.txt"),
    ("node_form_id", "form_vocabulary.txt"),
    ("node_lemma_id", "lemma_vocabulary.txt"),
    ("node_preverb_id", "preverb_vocabulary.txt"),
]


def worker(job):
    idx, files, graphml_dir, p_dir, shard_dir = job
    out = os.path.join(shard_dir, f"shard_{idx:03d}.npz")
    index = os.path.join(shard_dir, f"shard_{idx:03d}_index.json")
    # Each shard writes its vocabularies beside itself (ingest keys them off the
    # output path), so workers never contend for the same files.
    sys.stdout = open(os.path.join(shard_dir, f"shard_{idx:03d}.log"), "w")
    ingest.process_corpus(graphml_dir, p_dir, output_filename=out,
                          index_filename=index, filepaths=files,
                          progress_every=200, vocab_prefix=f"shard_{idx:03d}_")
    return out


def read_vocab(path):
    """id -> token list, as written by ingest._write_vocab."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tok_id, tok = line.split("\t", 1) if "\t" in line else (line, "")
            pairs.append((int(tok_id), tok))
    out = [""] * (max(p[0] for p in pairs) + 1)
    for tok_id, tok in pairs:
        out[tok_id] = tok
    return out


def merge(shards, shard_dir, out_path, index_paths):
    """Concatenate shards, fixing node/edge indices and vocabulary ids."""
    # ---- pass 1: build the union vocabularies -----------------------------
    # Seeded in the same order ingest itself would produce for the first shard,
    # so ids stay stable and low-numbered for the common tags.
    globals_ = {}
    for _, vocab_file in VOCAB_COLUMNS:
        globals_[vocab_file] = {}
    per_shard_maps = []
    for s_idx, shard in enumerate(shards):
        d = os.path.dirname(shard)
        maps = {}
        for _, vocab_file in VOCAB_COLUMNS:
            local = read_vocab(os.path.join(d, f"shard_{s_idx:03d}_" + vocab_file))
            g = globals_[vocab_file]
            remap = np.empty(len(local), dtype=np.int32)
            for local_id, tok in enumerate(local):
                if tok not in g:
                    g[tok] = len(g)
                remap[local_id] = g[tok]
            maps[vocab_file] = remap
        per_shard_maps.append(maps)

    # ---- pass 2: concatenate ----------------------------------------------
    acc = {k: [] for k in PLAIN}
    rowptr_parts, colidx_parts = [], []
    sent_off_parts, gold_nodes_parts, gold_off_parts = [], [], []
    text_parts, text_off_parts = [], []
    node_base = edge_base = gold_base = text_base = 0
    sentence_index = []

    for s_idx, shard in enumerate(shards):
        z = np.load(shard)
        n_nodes = len(z["node_features"])
        maps = per_shard_maps[s_idx]

        for name in PLAIN:
            arr = z[name]
            for col, vocab_file in VOCAB_COLUMNS:
                if col == name:
                    arr = maps[vocab_file][arr]
                    break
            acc[name].append(arr)

        rp = z["rowptr"].astype(np.int64)
        rowptr_parts.append((rp[1:] + edge_base).astype(np.int32))
        colidx_parts.append((z["colidx"].astype(np.int64) + node_base).astype(np.int32))
        sent_off_parts.append((z["sentenceoffsets"].astype(np.int64) + node_base).astype(np.int32))
        gold_nodes_parts.append((z["gold_path_nodes"].astype(np.int64) + node_base).astype(np.int32))
        go = z["gold_path_offsets"].astype(np.int64)
        gold_off_parts.append((go[1:] + gold_base).astype(np.int32))
        text_parts.append(z["surface_text"])
        to = z["surface_text_offsets"].astype(np.int64)
        text_off_parts.append(to[1:] + text_base)

        node_base += n_nodes
        edge_base += int(rp[-1])
        gold_base += int(go[-1])
        text_base += int(to[-1])
        with open(index_paths[s_idx], encoding="utf-8") as f:
            sentence_index.extend(json.load(f))
        del z

    def cat32(parts, lead=None):
        arrs = ([np.array([lead], dtype=np.int32)] if lead is not None else []) + parts
        return np.concatenate(arrs).astype(np.int32)

    save = {name: np.concatenate(acc[name]) for name in PLAIN}
    save["node_length"] = save["node_length"].astype(np.int32)
    save["rowptr"] = cat32(rowptr_parts, lead=0)
    save["colidx"] = np.concatenate(colidx_parts).astype(np.int32)
    save["sentenceoffsets"] = np.concatenate(sent_off_parts).astype(np.int32)
    save["goldpathmask"] = save["goldpathmask"].astype(np.int8)
    save["gold_path_nodes"] = np.concatenate(gold_nodes_parts).astype(np.int32)
    save["gold_path_offsets"] = cat32(gold_off_parts, lead=0)
    save["surface_text"] = np.concatenate(text_parts).astype(np.uint8)
    save["surface_text_offsets"] = np.concatenate(
        [np.array([0], dtype=np.int64)] + text_off_parts).astype(np.int64)

    print(f"merged: {len(save['node_features']):,} nodes, "
          f"{len(save['colidx']):,} edges, "
          f"{len(save['sentenceoffsets']):,} sentences")
    np.savez_compressed(out_path, **save)

    vocab_dir = os.path.dirname(os.path.abspath(out_path))
    for _, vocab_file in VOCAB_COLUMNS:
        g = globals_[vocab_file]
        with open(os.path.join(vocab_dir, vocab_file), "w", encoding="utf-8") as f:
            for tok, tok_id in sorted(g.items(), key=lambda x: x[1]):
                f.write(f"{tok_id}\t{tok}\n")
        print(f"  {vocab_file}: {len(g):,} entries")
    return sentence_index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphml-dir", default="../../SIGHUM_database/After_graphml")
    ap.add_argument("--p-dir", default="../../SIGHUM_database_gold_path/DCS_pick")
    ap.add_argument("--out", default="../data/cushr_data_full_with_gold.npz")
    ap.add_argument("--index", default="sentence_index.json")
    ap.add_argument("--shard-dir", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-shards", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.graphml_dir, "*.graphml")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no graphml under {args.graphml_dir}")

    shard_dir = args.shard_dir or (os.path.abspath(args.out) + ".shards")
    os.makedirs(shard_dir, exist_ok=True)

    n = args.workers
    bounds = [round(i * len(files) / n) for i in range(n + 1)]
    jobs = [(i, files[bounds[i]:bounds[i + 1]], args.graphml_dir, args.p_dir, shard_dir)
            for i in range(n) if bounds[i + 1] > bounds[i]]
    print(f"{len(files):,} sentences across {len(jobs)} shards "
          f"(~{len(files)//len(jobs):,} each)")

    t0 = time.time()
    with mp.Pool(len(jobs)) as pool:
        shards = pool.map(worker, jobs)
    print(f"shards done in {(time.time() - t0)/60:.1f} min; merging ...")

    index_paths = [os.path.join(shard_dir, f"shard_{i:03d}_index.json")
                   for i in range(len(jobs))]
    sentence_index = merge(shards, shard_dir, args.out, index_paths)
    with open(args.index, "w", encoding="utf-8") as f:
        json.dump(sentence_index, f)

    if not args.keep_shards:
        for f in glob.glob(os.path.join(shard_dir, "shard_*")):
            os.remove(f)
        os.rmdir(shard_dir)
    print(f"TOTAL_MIN {(time.time() - t0)/60:.1f}")
    print(f"success! artifact saved as: {args.out}")


if __name__ == "__main__":
    main()
