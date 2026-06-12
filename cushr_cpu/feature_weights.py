import numpy as np

NPZ_PATH = '../data/new_cushr_data_fixed.npz'

FEATURE_TOKENS = [
    'nom', 'acc', 'dat', 'abl', 'g', 'loc', 'voc', 'i',
    'sg', 'du', 'pl',
    'm', 'f', 'n', '*',
    'pr', 'pft', 'impft', 'fut', 'aor', 'opt', 'imp', 'per',
    'ac', 'md', 'ps',
    '1', '2', '3',
    'ppr', 'pfp', 'pp', 'ppa',
    'adv', 'ind', 'abs', 'inf', 'iic', 'conj', 'prep', 'part', 'ca',
    'UNKNOWN',
]

data = np.load(NPZ_PATH)
feats = data['node_features']          # (N, 43) float32
mask  = data['gold_path_mask'].ravel() # (N,) uint8

gold  = feats[mask == 1]
other = feats[mask == 0]

eps = 1e-6
gold_freq  = gold.mean(axis=0)
other_freq = other.mean(axis=0)
weights    = np.log((gold_freq + eps) / (other_freq + eps))

print(f"{'feature':<12}  {'gold_freq':>10}  {'other_freq':>10}  {'weight':>8}")
print("-" * 48)
for name, gf, of, w in zip(FEATURE_TOKENS, gold_freq, other_freq, weights):
    print(f"{name:<12}  {gf:>10.4f}  {of:>10.4f}  {w:>8.4f}")

weight_str = ','.join(f'{w:.4f}' for w in weights)
print(f"\n--weights {weight_str}")
