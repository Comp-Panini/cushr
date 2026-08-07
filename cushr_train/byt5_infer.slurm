#!/bin/bash
#----------------------------------------------------------------------
# byt5_infer.slurm — run the RELEASED ByT5-Sanskrit on our SIGHUM 4,200.
#
# Purpose: a true head-to-head at S / L / M against cuSHR. Both systems get
# scored on the same 4,200 sentences, against the same external reference
# (forms from sighum_test_4200.tsv, lemma+cng from the DCS .p files), by the
# same scorer (eval_slm.py). Nothing here is quoted from either paper.
#
# We run THEIR inference code rather than reimplementing the serialization --
# task prefixes and output post-processing are the easiest thing to get subtly
# wrong in a way that still produces plausible numbers.
#
#   repo   github.com/sebastian-nehrdich/byt5-sanskrit-analyzers
#   model  chronbmm/sanskrit5-multitask   (the fine-tuned multitask checkpoint)
#
# TWO-STAGE. Run the self-test first and paste back its output:
#     sbatch --export=ALL,LIMIT=20 byt5_infer.slurm
# Confirm the .tsv looks like segmented Sanskrit with lemma_TAG fields, then:
#     sbatch byt5_infer.slurm
#
# RUNTIME. This is inference only -- no training, no gradients. It is not
# instant because ByT5 is byte-level and AUTOREGRESSIVE: output is produced one
# byte at a time and each byte costs a decoder pass, so a 41-char sentence needs
# ~150 sequential steps at the S+L+M level. Batching amortises across sentences
# but not across those steps.
#
#   A100, batch 32 : ~150 steps x ~131 batches  ->  5-15 min for all 4,200
#   CPU            : ~1-2 h (not 12-46 h as an earlier draft of this said)
#
# It runs here rather than locally for MEMORY, not time: 582M fp32 params need
# ~2.3 GB resident and the dev box has ~1 GB free, so it would page and crawl.
# Walltime is set to 30 min -- generous against 15, and short enough to queue
# quickly. Raise it if the checkpoint download is slow on a cold HF cache.
#
# Submit:  sbatch byt5_infer.slurm
# Output:  byt5_infer.o<jobid>, byt5_out/*.tsv, byt5_preds_*.jsonl
#----------------------------------------------------------------------
#SBATCH -J byt5_infer
#SBATCH -o byt5_infer.o%j
#SBATCH -e byt5_infer.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
###SBATCH -A your_allocation
#----------------------------------------------------------------------

# NOTE: `set -e` is deliberately NOT enabled here. Under PREFLIGHT=1 every check
# must run and report even after an earlier one fails -- otherwise each job
# reveals exactly one problem, which is what turned this into three wasted
# submissions. It is turned on for the real run further down.
set -uo pipefail

# `module reset`, NOT `module purge`. THIS WAS THE ROOT CAUSE of jobs 3343791,
# 3346304 and their successors.
#
# python3/3.9.7 is loaded BY DEFAULT on Lonestar6 (`module avail python` shows
# it flagged L). It lives in /opt/apps/intel19/impi19_0/modulefiles, and that
# directory is only on MODULEPATH because the intel19 + impi19_0 modules put it
# there. `module purge` unloads those, so python3 becomes UNFINDABLE rather than
# merely unloaded -- `module load python3` then fails, the shell falls back to
# /usr/bin/python3 = 3.6.8, and the newest torch wheel for 3.6 is 1.10.2+cu102.
# Every "old torch" symptom traced back to this one line.
#
# `module reset` restores the system default set, which is what TACC recommends.
module reset 2>/dev/null || module purge
module load cuda
PREFLIGHT="${PREFLIGHT:-0}"

cd "$SLURM_SUBMIT_DIR"

LIMIT="${LIMIT:-0}"          # 0 = full 4,200; set to 20 for the self-test
BATCH="${BATCH:-32}"
STEM="sighum_test"
[ "$LIMIT" != "0" ] && STEM="sighum_selftest"

echo "==== node / gpu ===="
hostname
nvidia-smi
echo "LIMIT=$LIMIT  BATCH=$BATCH  STEM=$STEM"
echo

# ---- environment ----------------------------------------------------
# Kept in $SCRATCH, not $HOME: the checkpoint is ~2.3 GB and HOME quotas on
# Lonestar6 are small enough that a silent write failure is a real risk.
export HF_HOME="${SCRATCH:-$PWD}/hf_cache"
VENV="${SCRATCH:-$PWD}/byt5_venv"

# Belt-and-braces after the `module reset` fix above: with reset, python3/3.9.7
# is already loaded and this loop is a no-op. It stays because it is the thing
# that makes a wrong interpreter LOUD rather than silent, and PYMOD lets a
# specific version be forced from the command line:
#     sbatch --export=ALL,PYMOD=python3/3.9.7 byt5_infer.slurm
for m in ${PYMOD:-} python/3.12.11 python3/3.9.7 python python3; do
    [ -z "$m" ] && continue
    if module load "$m" 2>/dev/null; then
        echo "loaded module: $m"
        break
    fi
done
echo "python3: $(python3 --version 2>&1)  ($(command -v python3))"

# Check the INTERPRETER before building anything. The venv takes ~4 minutes and
# on Python 3.6 it can only ever produce torch 1.10, so discovering that after
# the build wastes the whole job (which is what 3346304 did).
python3 - <<'EOF'
import sys
if sys.version_info < (3, 9):
    raise SystemExit(
        f"python {sys.version.split()[0]} is too old -- torch >=2 needs 3.9+.\n"
        "  On a login node, find a usable module with:\n"
        "      module avail python 2>&1 | head -40\n"
        "      module spider python3\n"
        "  then re-submit with:  sbatch --export=ALL,PYMOD=<name> byt5_infer.slurm\n"
        "  Also: rm -rf $SCRATCH/byt5_venv, or the stale env is reused.")
EOF

# Rebuild the venv when it is unusable, rather than trusting that the directory
# existing means it is good. Job 3346304 built one from Python 3.6.8; on the
# next run the `-d "$VENV"` test passed, the stale env was reused, and the job
# failed with the identical error -- indistinguishable from "the fix did not
# land". The venv carries its OWN interpreter, so upgrading the module does not
# upgrade an env already built from the old one.
VENV_OK=0
if [ -x "$VENV/bin/python" ]; then
    if "$VENV/bin/python" - <<'EOF' 2>/dev/null
import sys
assert sys.version_info >= (3, 9), sys.version
import torch, transformers, pandas          # noqa: F401
assert int(torch.__version__.split('.')[0]) >= 2, torch.__version__
EOF
    then
        VENV_OK=1
        echo "reusing venv at $VENV ($("$VENV/bin/python" --version 2>&1))"
    else
        echo "existing venv is unusable ($("$VENV/bin/python" --version 2>&1)," \
             "or missing/old torch) -- rebuilding"
        rm -rf "$VENV"
    fi
fi

if [ "$VENV_OK" = "0" ]; then
    echo "==== creating venv ===="
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    # pandas is imported by their run_inf_batch.py and was the actual crash in
    # job 3343791; the rest are transformers' runtime deps.
    "$VENV/bin/pip" install --quiet \
        torch transformers sentencepiece accelerate pandas protobuf
fi
PY="$VENV/bin/python"

echo "==== environment ===="
$PY - <<'EOF'
import sys
print("python      ", sys.version.split()[0])
import torch, transformers, pandas
print("torch       ", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("pandas      ", pandas.__version__)
EOF
echo

# ---- module state, printed so the root cause cannot recur silently ----
echo "==== module state ===="
module list 2>&1 | head -20
echo "MODULEPATH=$MODULEPATH" | tr ':' '\n' | head -8
echo

# ---- their inference code -------------------------------------------
# Cloned into $SCRATCH, NOT the submit dir. A git clone inside the repo makes
# `git add .` try to stage an embedded repository, which git refuses to handle
# sanely and which is a nuisance to unpick afterwards.
CLONE="${SCRATCH:-$PWD}/byt5-sanskrit-analyzers"
if [ ! -d "$CLONE" ]; then
    echo "==== cloning byt5-sanskrit-analyzers -> $CLONE ===="
    git clone --depth 1 \
        https://github.com/sebastian-nehrdich/byt5-sanskrit-analyzers.git "$CLONE"
fi
APP="$CLONE/applications/segmentation-lemma-tagging"

# ---- prepare input (SLP1 -> IAST, order pinned by a manifest) --------
echo "==== preparing input ===="
$PY make_byt5_input.py --out-dir byt5_in --stem "$STEM" --limit "$LIMIT"
echo

# ---- preflight: run EVERY check, report, do not abort -----------------
# The point is that one job answers every open question. Each check prints
# [ok] or [FAIL] and execution continues regardless, so a single log tells us
# what to fix instead of revealing one problem per five-minute cycle.
if [ "$PREFLIGHT" = "1" ]; then
    echo "======================= PREFLIGHT ======================="
    FAILED=0
    chk () {  # chk "<label>" <command...>
        local label="$1"; shift
        local out
        if out=$("$@" 2>&1); then
            echo "[ok]   $label"
            [ -n "$out" ] && echo "$out" | sed 's/^/         /' | head -6
        else
            echo "[FAIL] $label"
            echo "$out" | sed 's/^/         /' | head -12
            FAILED=1
        fi
    }

    chk "python >= 3.9" $PY -c \
        "import sys; assert sys.version_info>=(3,9), sys.version; print(sys.version.split()[0])"
    chk "torch >= 2 and CUDA visible" $PY -c \
        "import torch; assert int(torch.__version__.split('.')[0])>=2, torch.__version__; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

    # pypi and huggingface are tested SEPARATELY: a compute node can reach one
    # and not the other, and pip already worked, so HF is the real question.
    chk "network: pypi.org" $PY -c \
        "import urllib.request; urllib.request.urlopen('https://pypi.org/simple/', timeout=20); print('reachable')"
    chk "network: huggingface.co" $PY -c \
        "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=20); print('reachable')"

    # The real thing: pull the 2.3 GB checkpoint. Never attempted before.
    chk "load chronbmm/sanskrit5-multitask" $PY -c "
import time, transformers
t=time.time()
tok=transformers.AutoTokenizer.from_pretrained('chronbmm/sanskrit5-multitask')
m=transformers.AutoModelForSeq2SeqLM.from_pretrained('chronbmm/sanskrit5-multitask')
print('loaded in %.1fs, %.0fM params'%(time.time()-t, sum(p.numel() for p in m.parameters())/1e6))"

    echo "         HF cache: $(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"

    # Two sentences end to end through THEIR script. Settles both output-format
    # unknowns: where the .tsv lands, and what the `analyzed` column looks like.
    echo
    echo "---- 2-sentence end-to-end ----"
    PFD="$SCRATCH/byt5_preflight"; rm -rf "$PFD"; mkdir -p "$PFD"
    head -n 2 "byt5_in/$STEM.txt" > "$PFD/pf.txt" 2>/dev/null || \
        printf 'cikṣepa me suto rājan\naham saubhapateḥ senām\n' > "$PFD/pf.txt"
    cat "$PFD/pf.txt"
    ( cd "$APP" && $PY run_inf_batch.py --input-folder "$PFD" \
        --mode segmentation-lemma-morphosyntax --batch-size 2 ) 2>&1 | tail -15
    echo "---- files produced ----"
    ls -la "$PFD"
    echo "---- .tsv verbatim ----"
    cat "$PFD"/*.tsv 2>/dev/null || echo "(no .tsv produced)"

    echo
    echo "======================= PREFLIGHT DONE (FAILED=$FAILED) ======="
    exit $FAILED
fi

# Real run from here on: now abort on the first error.
set -e

# ---- inference ------------------------------------------------------
# Two modes. `segmentation` gives the S number on its own; the combined mode is
# what S+L+M is scored from. Running both means a disagreement between them is
# visible rather than averaged away.
for MODE in segmentation segmentation-lemma-morphosyntax; do
    echo "==== mode=$MODE ===="
    RUN="byt5_out/$MODE"
    mkdir -p "$RUN"
    cp "byt5_in/$STEM.txt" "$RUN/"
    ( cd "$APP" && $PY run_inf_batch.py \
        --input-folder "$SLURM_SUBMIT_DIR/$RUN" \
        --mode "$MODE" \
        --batch-size "$BATCH" )
    echo "---- first 3 output rows ----"
    head -n 4 "$RUN/$STEM.tsv" || true
    echo
done

# ---- convert to the scorer's JSONL ----------------------------------
echo "==== converting to JSONL ===="
for MODE in segmentation segmentation-lemma-morphosyntax; do
    $PY byt5_to_jsonl.py \
        --tsv "byt5_out/$MODE/$STEM.tsv" \
        --manifest "byt5_in/$STEM.manifest.json" \
        --mode "$MODE" \
        --out "byt5_preds_${MODE}_${STEM}.jsonl"
done

echo
echo "==== done. score with: ===="
echo "  python eval_slm.py --cache ./cache95_ctx_ex4200 \\"
echo "      --model model95_ctx_ex4200.npz \\"
echo "      --pred-jsonl byt5_preds_segmentation-lemma-morphosyntax_${STEM}.jsonl"
