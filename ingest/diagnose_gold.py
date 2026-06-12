"""
Quick diagnostic: sample N .p files and report how many load successfully,
how many have lemmas/cng attributes, and how many produce gold triples.
"""
import glob, os, sys, pickle, random

P_DIR = "../../SIGHUM_database_gold_path/DCS_pick"
N_SAMPLE = 50
SEED = 42

class DCS(object):
    pass

sys.modules['__main__'].DCS = DCS

files = glob.glob(os.path.join(P_DIR, '*.p'))
print(f'Total .p files: {len(files):,}')

random.seed(SEED)
sample = random.sample(files, min(N_SAMPLE, len(files)))

ok, no_attr, failed = 0, 0, 0
errors = {}

for path in sample:
    try:
        with open(path, 'rb') as f:
            dcs = pickle.load(f, encoding='utf-8')
        lemmas = getattr(dcs, 'lemmas', None)
        cngs   = getattr(dcs, 'cng', None)
        if lemmas is None or cngs is None:
            no_attr += 1
            # show what attributes exist
            attrs = vars(dcs) if hasattr(dcs, '__dict__') else dir(dcs)
            print(f'  no lemmas/cng in {os.path.basename(path)}: attrs={list(attrs)[:10]}')
        else:
            ok += 1
    except Exception as e:
        failed += 1
        key = type(e).__name__
        errors[key] = errors.get(key, 0) + 1
        if failed <= 3:
            print(f'  load error {os.path.basename(path)}: {e}')

print(f'\nSample of {len(sample)} files:')
print(f'  loaded OK with lemmas+cng : {ok}')
print(f'  loaded but missing attrs  : {no_attr}')
print(f'  failed to load            : {failed}')
if errors:
    print(f'  error types: {errors}')
