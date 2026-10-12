#!/bin/bash
#----------------------------------------------------------------------
# eval_union.slurm -- the two-model sum-rule ensemble on SIGHUM 4,200.
#
# WHY HERE AND NOT LOCALLY. The ensemble needs both models' materialized
# feature caches open in turn, and cache95_lattice_attn_char_s0 only exists on
# the cluster (3.6 GB, like the seg one).
#
# WHAT IT ANSWERS. On the g95 test split the ensemble was worth +1.02 PM over
# the better single model (93.94 vs 92.92) on the reranker's own label metric.
# That is not a number anyone else reports. This produces the published-benchmark
# version: word-sequence PM and macro word F on the same 4,200 sentences
# TransLIST reports 93.97 / 98.86 on.
#
# THE RECIPE IS FROZEN. K=16 distinct segmentations per model, 3 rerankers per
# model, equal weight, exactly as on g95. Do not adjust K, the weights or which
# rerankers are included after reading the output -- that is tuning on the test
# set and the number stops meaning anything. If it disagrees with g95, that is
# the finding.
#
# CPU-only: the decode is the same k-best path eval_surface.py uses.
#     sbatch eval_union.slurm
#----------------------------------------------------------------------
#SBATCH -J eval_union
#SBATCH -o eval_union.o%j
#SBATCH -e eval_union.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

set -euo pipefail

module reset 2>/dev/null || module purge
cd "$SLURM_SUBMIT_DIR"

PY="${SCRATCH:-$PWD}/attn_venv/bin/python"
[ -x "$PY" ] || { echo "no venv python at $PY"; exit 1; }

CACHE_SEG="${CACHE_SEG:-./cache95_seg_ex4200}"
MODEL_SEG="${MODEL_SEG:-model95_seg_ex4200.npz}"
CACHE_ATT="${CACHE_ATT:-./cache95_lattice_attn_char_s0}"
MODEL_ATT="${MODEL_ATT:-model95_lattice_attn_char_s0.npz}"
K="${K:-16}"

RR_SEG="segrr_k16_featseq_loss_s0.pt segrr_k16_featseq_loss_s1.pt segrr_k16_featseq_loss_s2.pt"
RR_ATT="segrr_lattice_attn_char_s0_s0.pt segrr_lattice_attn_char_s0_s1.pt segrr_lattice_attn_char_s0_s2.pt"

for f in "$CACHE_SEG/splits.json" "$CACHE_ATT/splits.json" "$MODEL_SEG" \
         "$MODEL_ATT" sighum_test_4200.tsv ../data/cushr_data_g95.npz \
         ../data/sentence_index_g95.json ../data/form_vocabulary.txt \
         $RR_SEG $RR_ATT; do
    [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
    echo "  ok  $f"
done

$PY eval_union.py \
    --cache-a "$CACHE_SEG" --model-a "$MODEL_SEG" --rerank-a $RR_SEG \
    --name-a seg \
    --cache-b "$CACHE_ATT" --model-b "$MODEL_ATT" --rerank-b $RR_ATT \
    --name-b attn \
    --k "$K" --json-out "eval_union_sighum_k${K}.json" \
    --dump "eval_union_sighum_k${K}_preds.json"

echo
echo "compare the ensemble PM against: cuSHR-seg +rerank 93.43, TransLIST 93.97"
echo "and the macro word F against TransLIST 98.86"
