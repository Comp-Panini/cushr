"""Can a train-derived SHR->DCS lemma rewrite table close the L gap?

The oracle's lemma disagreements with DCS are dominated by a regular pattern:
SHR gives the participial stem, DCS gives the verbal root (ukta/vac, gata/gam,
kfta/kf). If that mapping is deterministic, a lookup applied at OUTPUT time
fixes it without touching the lattice -- which would mean the L gap is a
convention mismatch, not an architectural limit.

The table MUST be built from the training split only. Deriving it from test and
scoring on test would be leakage, and would overstate the achievable gain.
"""
import csv, json, os, random, sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = r"C:/Users/nishk/OneDrive - The University of Texas at Austin/Sanskrit/cushr/cushr_train"
sys.path.insert(0, os.path.join(TRAIN_DIR, "..", "ingest"))
os.chdir(TRAIN_DIR)
import ingest as ig

PD = "../../SIGHUM_database_gold_path/DCS_pick"
index = json.load(open("../data/sentence_index_repaired.json"))
pos = {s: i for i, s in enumerate(index)}
train = set(json.load(open("splits95_ex4200.json"))["train"])

z = np.load("../data/cushr_data_g95.npz")
lid, cs = z["node_lemma_id"], z["node_char_start"]
go = z["gold_path_offsets"].astype(np.int64)
gn = z["gold_path_nodes"].astype(np.int64)
lv = [l.split("\t", 1)[1].rstrip("\n") if "\t" in l else ""
      for l in open("../data/lemma_vocabulary.txt", encoding="utf-8")]


def gold_lemmas(stem):
    with open(os.path.join(PD, stem + ".p"), "rb") as f:
        d = ig._DCSUnpickler(f, encoding="utf-8").load()
    return [ig.normalize_lemma(str(w))
            for cl in getattr(d, "lemmas", []) for w in cl]


def aligned(s):
    """(shr_lemmas, dcs_lemmas) for one sentence, or None if not comparable."""
    if go[s + 1] - go[s] == 0:
        return None
    try:
        g = gold_lemmas(index[s])
    except Exception:
        return None
    nodes = sorted(gn[go[s]:go[s + 1]].tolist(), key=lambda x: int(cs[x]))
    p = [lv[lid[x]] for x in nodes]
    return (p, g) if len(p) == len(g) else None


# ---- build from a TRAIN sample (pickle IO is the bottleneck, not the stats) --
random.seed(0)
sample = random.sample(sorted(train), 25000)
m = defaultdict(Counter)
used = 0
for s in sample:
    a = aligned(s)
    if not a:
        continue
    used += 1
    for x, y in zip(*a):
        m[x][y] += 1

MAP = {}
for shr, c in m.items():
    top, cnt = c.most_common(1)[0]
    # Only confident, non-identity rewrites: seen 3+ times and >=90% consistent.
    if top != shr and cnt >= 3 and cnt / sum(c.values()) >= 0.90:
        MAP[shr] = top
print(f"train sentences used : {used:,}")
print(f"rewrite entries      : {len(MAP):,}")
print("examples             : " +
      ", ".join(f"{k}->{v}" for k, v in list(MAP.items())[:10]))

# ---- apply to TEST ----------------------------------------------------------
rows = list(csv.DictReader(open("sighum_test_4200.tsv", encoding="utf-8-sig"),
                           delimiter="\t"))
tot = h_before = h_after = 0
n = pm_before = pm_after = 0
for r in rows:
    s = pos[str(r["DCS-ID"]).strip()]
    a = aligned(s)
    if not a:
        continue
    p, g = a
    q = [MAP.get(x, x) for x in p]
    n += 1
    pm_before += (p == g)
    pm_after += (q == g)
    for x, y, t in zip(p, q, g):
        tot += 1
        h_before += (x == t)
        h_after += (y == t)

print()
print("ORACLE lemma accuracy on TEST, rewrite map built from train only")
print(f"  word-level  : {100*h_before/tot:6.2f}  ->  {100*h_after/tot:6.2f}"
      f"   ({100*(h_after-h_before)/tot:+.2f})")
print(f"  sentence PM : {100*pm_before/n:6.2f}  ->  {100*pm_after/n:6.2f}"
      f"   ({100*(pm_after-pm_before)/n:+.2f})")
print(f"  ByT5 measured: word 99.34, sentence PM 90.55")
