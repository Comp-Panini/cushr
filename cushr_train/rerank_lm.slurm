#!/bin/bash
#----------------------------------------------------------------------
# rerank_lm.slurm -- path reranker over cuSHR-lm's beams (the L+M pipeline).
#
# WHY. cuSHR-lm scores 54.67 L+M (+maps) while cuSHR-node + reranker scores
# 57.57: a reranked OLD model beats the unreranked new one, so the published
# comparison is not apples to apples. This closes that gap.
#
# WHAT IT CANNOT FIX. An edge score is src_proj(u).dst_proj(v) + bias -- two
# node vectors. Choosing nom over acc on a homograph needs to know a nominative
# is already spoken for elsewhere in the clause; no first-order edge score can
# say that. rerank.py runs a BiLSTM over the whole candidate path, which can.
# That is why reranking is worth +8.95 S+M on the old model (52.26 -> 61.21)
# while retraining the base scorer was worth +2.3.
#
# THE CEILING. S+M recall@16 is 73.67 and recall@64 is 76.52, against a 77.65
# oracle: the beam nearly always holds the right analysis. The reranker reorders
# these candidates and can never add one, so recall@16 is a hard bound on this
# whole script. Expect ~63 L+M, not 76.
#
# NOTE the label. --label node (the default, so not passed) supervises the FULL
# analysis including lemma and morph tag. The segmentation pipeline in
# train_attn.slurm passes --label surface --dedup-surface --seg-features
# instead; do not mix the two, they train different rerankers.
#
# STAGES. Data dump first -- it is the expensive part (~25 min on CPU) and both
# later stages read it:
#     sbatch --export=ALL,STAGE=data  rerank_lm.slurm
#     sbatch --export=ALL,STAGE=train rerank_lm.slurm
#     sbatch --export=ALL,STAGE=eval  rerank_lm.slurm
#----------------------------------------------------------------------
#SBATCH -J rerank_lm
#SBATCH -o rerank_lm.o%j
#SBATCH -e rerank_lm.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00

set -euo pipefail

STAGE="${STAGE:-data}"
TAG="${TAG:-lm_ex4200}"
K="${K:-16}"
KDECODE="${KDECODE:-64}"
EPOCHS="${EPOCHS:-14}"
PATIENCE="${PATIENCE:-3}"
SEEDS="${SEEDS:-0 1 2}"

CACHE="./cache95_${TAG}"
MODEL="model95_${TAG}.npz"
RAW=../data/cushr_data_g95.npz
SPLITS=splits95_ex4200.json
PREFIX="rr_${TAG}_k${K}"

module reset 2>/dev/null || module purge
cd "$SLURM_SUBMIT_DIR"

# Same venv the training jobs built; networkx was added for eval_slm.py.
PY="${SCRATCH:-$PWD}/attn_venv/bin/python"
[ -x "$PY" ] || { echo "no venv python at $PY"; exit 1; }

echo "=== stage=$STAGE tag=$TAG k=$K k_decode=$KDECODE"
$PY -c "import torch, numpy, networkx; print('torch', torch.__version__)"

if [ "$STAGE" = "data" ]; then
    for f in "$CACHE/splits.json" "$MODEL" "$RAW" "$SPLITS"; do
        [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
        echo "  ok  $f"
    done
    # Candidate lists + gold-path labels. --k-decode 64 decodes deeper than it
    # keeps, so the kept 16 are the 16 best of a wider search.
    $PY make_rerank_data.py --cache "$CACHE" --model "$MODEL" --raw "$RAW" \
        --splits-override "$SPLITS" --k "$K" --k-decode "$KDECODE" \
        --out-prefix "$PREFIX"
    echo "wrote ${PREFIX}_{train,dev,test}.npz"
    echo "now: sbatch --export=ALL,STAGE=train,TAG=$TAG rerank_lm.slurm"
    exit 0
fi

if [ "$STAGE" = "train" ]; then
    # Three seeds: the reported segmentation reranker moved 0.13 between seeds,
    # so a single seed is not a result. Report the mean, never the best.
    for s in $SEEDS; do
        echo "=== seed $s"
        $PY rerank.py --train "${PREFIX}_train.npz" --dev "${PREFIX}_dev.npz" \
            --test "${PREFIX}_test.npz" --raw "$RAW" \
            --epochs "$EPOCHS" --patience "$PATIENCE" --seed "$s" \
            --out "rr_${TAG}_s${s}.pt"
    done
    echo "now: sbatch --export=ALL,STAGE=eval,TAG=$TAG rerank_lm.slurm"
    exit 0
fi

if [ "$STAGE" = "eval" ]; then
    # Raw and +maps, for each seed. --kbest K --rerank is the only pair of flags
    # that changes the cuSHR column; everything else matches the Oct 6 run that
    # produced eval_lm_raw.json / eval_lm_maps.json, so the rows are comparable.
    for s in $SEEDS; do
        echo "=== seed $s, raw (compare against L+M 47.79)"
        $PY eval_slm.py --cache "$CACHE" --model "$MODEL" \
            --index ../data/sentence_index_g95.json --raw "$RAW" \
            --kbest "$K" --rerank "rr_${TAG}_s${s}.pt" \
            --json-out "eval_${TAG}_rr_raw_s${s}.json"

        echo "=== seed $s, +maps (compare against L+M 54.67; ByT5 is 76.50)"
        $PY eval_slm.py --cache "$CACHE" --model "$MODEL" \
            --index ../data/sentence_index_g95.json --raw "$RAW" \
            --lemma-map lemma_map.json --convention-map convention_map.json \
            --tag-bridge tag_bridge.json \
            --pred-jsonl byt5_preds_segmentation-lemma-morphosyntax_sighum_test.jsonl \
            --pred-name ByT5 \
            --kbest "$K" --rerank "rr_${TAG}_s${s}.pt" \
            --json-out "eval_${TAG}_rr_maps_s${s}.json"
    done
    echo "report the 3-seed mean of the +maps L+M row, not the best seed"
    exit 0
fi

echo "unknown STAGE=$STAGE (data|train|eval)"; exit 1
