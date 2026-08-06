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

set -euo pipefail

module purge
module load cuda

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

if [ ! -d "$VENV" ]; then
    echo "==== creating venv ===="
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet torch transformers sentencepiece accelerate
fi
PY="$VENV/bin/python"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo

# ---- their inference code -------------------------------------------
if [ ! -d byt5-sanskrit-analyzers ]; then
    echo "==== cloning byt5-sanskrit-analyzers ===="
    git clone --depth 1 https://github.com/sebastian-nehrdich/byt5-sanskrit-analyzers.git
fi
APP="byt5-sanskrit-analyzers/applications/segmentation-lemma-tagging"

# ---- prepare input (SLP1 -> IAST, order pinned by a manifest) --------
echo "==== preparing input ===="
$PY make_byt5_input.py --out-dir byt5_in --stem "$STEM" --limit "$LIMIT"
echo

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
