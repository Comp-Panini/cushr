#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
import zipfile

import numpy as np


def extract_array(npz_path, z, name, out):
    """Write one archive member to `out` as a .npy, without loading it whole.

    node_features is ~1.15 GB at corpus scale, and `np.save(out, z[name])`
    materialises all of it in RAM. Members of an uncompressed .npz are stored
    verbatim and each one already *is* a .npy file, so for those we can copy
    the bytes straight through in 64 KB chunks. Compressed archives fall back
    to the plain path.
    """
    try:
        with zipfile.ZipFile(npz_path) as zf:
            info = zf.getinfo(name + ".npy")
            if info.compress_type == zipfile.ZIP_STORED:
                with zf.open(info) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 16)
                return info.file_size, "streamed"
    except (KeyError, zipfile.BadZipFile):
        pass
    arr = z[name]
    np.save(out, arr)
    return arr.nbytes, "loaded"

ARRAYS = ["node_features", "node_word_length", "row_ptr", "col_idx", "topo_level", "sentence_offsets","gold_path_mask", "gold_path_nodes", "gold_path_offsets"]

# Present only when build_features.py ran with --emit-ids; required by the
# learned featurizers, ignored by the precomputed ones.
OPTIONAL_ARRAYS = ["node_ids", "id_vocab_sizes"]


#hashing it into 100 bins which can then be divided among dev, test, train
def bucket(sentence_id: int) -> int:
    h = hashlib.md5(str(sentence_id).encode()).hexdigest()
    return int(h[:8], 16) % 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="../data/new_cushr_data_fixed_USE_THIS.npz")
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--dev-pct", type=int, default=5) #define dev = 5
    ap.add_argument("--test-pct", type=int, default=5) #define test = 5
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    z = np.load(args.npz)

    # extract arrays from npz (zipped) to npy
    for name in ARRAYS + [a for a in OPTIONAL_ARRAYS if a in z.files]:
        out = os.path.join(args.cache, name + ".npy")
        if os.path.exists(out) and not args.force:
            print(f"  {name:<20} cached")
            continue
        nbytes, how = extract_array(args.npz, z, name, out)
        print(f"  {name:<20} {nbytes / 1e6:>10.1f} MB  {how}")

    #find what sentences have a gold path
    go = np.load(os.path.join(args.cache, "gold_path_offsets.npy"))
    n_sent = len(go) - 1
    gold_len = np.diff(go)
    has_gold = np.nonzero(gold_len > 0)[0]

    #assign sentences to dev test or train
    dev_hi = args.dev_pct
    test_hi = args.dev_pct + args.test_pct
    split = {"train": [], "dev": [], "test": []}
    for s in has_gold.tolist():
        b = bucket(s)
        if b < dev_hi:
            split["dev"].append(s)
        elif b < test_hi:
            split["test"].append(s)
        else:
            split["train"].append(s)

    # Carry the featurizer provenance through to the cache so training can
    # report and record which featurization it is fitting, and so a cache built
    # from one featurizer cannot be silently reused with another.
    meta = {
        "npz": os.path.abspath(args.npz),
        "featurizer": (str(z["featurizer_name"]) if "featurizer_name" in z.files
                       else "unknown"),
        "feat_dim": (int(z["feat_dim"]) if "feat_dim" in z.files
                     else int(z["node_features"].shape[1])),
        "num_sentences": int(n_sent),
        "num_with_gold": int(len(has_gold)),
        "gold_len_mean": float(gold_len[has_gold].mean()),
        "gold_len_max": int(gold_len.max()),
        "counts": {k: len(v) for k, v in split.items()},
    }
    with open(os.path.join(args.cache, "splits.json"), "w") as f:
        #write splits.json
        json.dump({"meta": meta, "splits": split}, f)

    print("\nsplit summary")
    print(f"  sentences total      : {n_sent}")
    print(f"  with gold path       : {len(has_gold)} "
          f"({100.0 * len(has_gold) / n_sent:.1f}%)")
    for k, v in split.items():
        print(f"  {k:<21}: {len(v)}")
    print(f"\nwrote {os.path.join(args.cache, 'splits.json')}")


if __name__ == "__main__":
    main()
