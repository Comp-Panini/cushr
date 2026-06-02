import numpy as np
import os

src = '../data/cushr_data_full_with_gold.npz'
dst = '../data/cushr_data_fixed.npz'
VOCAB_PATH = '../ingest/morph_vocabulary.txt'

# ---------------------------------------------------------------------------
# Why we parse tags instead of one-hot encoding:
# One-hot over 847 tags × 4.2 M nodes × 4 bytes ≈ 14 GB uncompressed.
# cnpy (the C++ .npz reader) does not support compressed archives, so that
# would be unloadable. Parsing each SHR tag string into 43 binary features
# covering case, number, gender, tense, voice, person, and word class keeps
# the matrix at ~730 MB and gives the LogLinearScorer semantically meaningful
# weights (e.g. one weight for "accusative", one for "singular", etc.)
# rather than treating tag IDs as a linear scale.
# ---------------------------------------------------------------------------

# Fixed ordered list of tokens to test for in each tag string.
# Presence (1.0) / absence (0.0) of each token becomes one feature.
# SHR abbreviations: g. = genitive, i. = instrumental, pr. = present, etc.
FEATURE_TOKENS = [
    # Nominal case (8)
    'nom', 'acc', 'dat', 'abl', 'g', 'loc', 'voc', 'i',
    # Grammatical number (3)
    'sg', 'du', 'pl',
    # Gender (4;  '*' = gender-indeterminate form)
    'm', 'f', 'n', '*',
    # Tense / mood (8)
    'pr', 'pft', 'impft', 'fut', 'aor', 'opt', 'imp', 'per',
    # Voice (3)
    'ac', 'md', 'ps',
    # Person (3)
    '1', '2', '3',
    # Participial and verbal-noun forms (4)
    'ppr', 'pfp', 'pp', 'ppa',
    # Indeclinables and other word classes (9)
    'adv', 'ind', 'abs', 'inf', 'iic', 'conj', 'prep', 'part', 'ca',
]
# +1 for the explicit UNKNOWN flag
FEAT_DIM = len(FEATURE_TOKENS) + 1   # 43


def load_vocab(path):
    """Return {tag_id: tag_string} from morph_vocabulary.txt."""
    id_to_tag = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tag_id_str, tag_str = line.split('\t', 1)
            id_to_tag[int(tag_id_str)] = tag_str
    return id_to_tag


def tag_to_vec(tag_str):
    """Convert one SHR tag string to a FEAT_DIM float32 list."""
    tokens = set(tag_str.replace('.', ' ').split())
    vec = [1.0 if t in tokens else 0.0 for t in FEATURE_TOKENS]
    vec.append(1.0 if tag_str == 'UNKNOWN' else 0.0)
    return vec


# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
print(f'Loading {src} ...')
z = np.load(src)

morph_ids = z['node_features'].astype(np.int32).ravel()   # (N,)
N = len(morph_ids)
print(f'  {N:,} nodes  raw tag-id range: [{morph_ids.min()}, {morph_ids.max()}]')

# ---------------------------------------------------------------------------
# Build the feature lookup table: shape (max_id+1, FEAT_DIM)
# Index it with morph_ids to expand all nodes at once (vectorised, fast).
# ---------------------------------------------------------------------------
print(f'Reading vocabulary from {VOCAB_PATH} ...')
if not os.path.exists(VOCAB_PATH):
    raise FileNotFoundError(
        f'Vocabulary file not found: {VOCAB_PATH}\n'
        'Re-run ingest/ingest.py to regenerate morph_vocabulary.txt, '
        'or adjust VOCAB_PATH above.'
    )
id_to_tag = load_vocab(VOCAB_PATH)

max_id = max(id_to_tag.keys())
lookup = np.zeros((max_id + 1, FEAT_DIM), dtype=np.float32)
for tag_id, tag_str in id_to_tag.items():
    lookup[tag_id] = tag_to_vec(tag_str)

print(f'Expanding {N:,} nodes → ({N:,}, {FEAT_DIM}) feature matrix ...')
nf = lookup[morph_ids]   # vectorised index; shape (N, FEAT_DIM)

# Quick sanity check: every row should have at least one non-zero feature
# (UNKNOWN rows have the last bit set; real tags have at least one token hit).
zero_rows = int((nf.sum(axis=1) == 0).sum())
if zero_rows:
    print(f'  [warn] {zero_rows} nodes produced all-zero feature vectors '
          '(tag IDs not in vocabulary?)')

# ---------------------------------------------------------------------------
# Compute node_word_length from topo-level spans.
#
# ingest.py never extracted surface-form character lengths. We derive a proxy:
# for each node v, node_word_length[v] = the maximum topo-level gap across all
# incoming edges (u → v).  In the SHR lattice topo_level tracks position within
# the sentence, so topo[v] - topo[u] ≈ how many segmentation positions the word
# at v spans — a reliable proxy for actual word length.
#
# This is what LengthScorer (scorer.cpp) reads to score each edge as
# log(1 + word_length), which biases toward fewer, longer words — exactly the
# right prior for Sanskrit, where correct analyses use compact compound words.
# ---------------------------------------------------------------------------
print('Computing node_word_length from topo-level spans ...')
row_ptr_arr  = z['rowptr'].astype(np.int32)
col_idx_arr  = z['colidx'].astype(np.int32)
topo_arr     = z['topolevel'].astype(np.int32)

# Build source-node index for every edge: u_idx[e] = source of edge e.
out_degrees  = np.diff(row_ptr_arr)           # [N]
u_idx        = np.repeat(np.arange(N, dtype=np.int32), out_degrees)  # [E]
v_idx        = col_idx_arr                    # [E]  destination of edge e

spans        = topo_arr[v_idx] - topo_arr[u_idx]   # [E]  >= 1 by construction

# For each destination node, keep the maximum span across all incoming edges.
node_word_length = np.zeros(N, dtype=np.int32)
np.maximum.at(node_word_length, v_idx, spans)

print(f'  word_length range: [{node_word_length.min()}, {node_word_length.max()}]  '
      f'mean: {node_word_length.mean():.2f}')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
print(f'Saving to {dst} ...')
np.savez(dst,
    node_features    = nf,
    node_word_length = node_word_length,
    row_ptr          = row_ptr_arr,
    col_idx          = col_idx_arr,
    topo_level       = topo_arr,
    sentence_offsets = np.append(
        z['sentenceoffsets'], morph_ids.shape[0]
    ).astype(np.int32),
    gold_path_mask   = z['goldpathmask'].astype(np.int8),
)

size_mb = os.path.getsize(dst) / 1e6
print(f'done  node_features shape: {nf.shape}  feat_dim={FEAT_DIM}  '
      f'file size: {size_mb:.0f} MB')
print()
print('Run the length scorer to use word spans:')
print('  ./cushr_evaluate ../data/cushr_data_fixed.npz --scorer length --K 10')
