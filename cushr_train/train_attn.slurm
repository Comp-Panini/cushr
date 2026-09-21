#!/bin/bash
#----------------------------------------------------------------------
# train_attn.slurm -- train cuSHR-seg with the lattice-attention encoder.
#
# WHAT THIS IS FOR. The base scorer is first-order: an edge score is a function
# of exactly two node vectors, so it cannot express "this candidate is wrong
# given a word ten characters away". TransLIST gets that from self-attention
# over all candidate words of a sentence, biased by how their spans sit relative
# to one another. `lattice_attn_char` is that, on cuSHR's lattice.
#
# The target: base decoder > 92.76 PM on SIGHUM (cuSHR-seg today), and the full
# pipeline > 93.43 (cuSHR-seg + reranker). TransLIST is 93.97 on the identical
# test set. See COMPARING_MODELS.md.
#
# NOTHING IN THE C++/CUDA DECODER CHANGES. The encoder output is still a
# per-node vector that depends on the sentence but never on the path, so
# --materialize freezes it and cushr_gpu/ consumes it unchanged.
#
# WHAT TO STAGE (~1.9 GB; everything else is produced here):
#     cushr_train/cache95_ngrams80/        1.8 GB  the only cache with node_ids
#                                                  + surface_text, both required
#     data/cushr_data_g95.npz              56 MB   for --surface-raw
#     data/sentence_index_g95.json         1.2 MB  eval only
#     cushr_train/splits95_ex4200.json     0.8 MB  holds the 4,200 out of train
#     cushr_train/sighum_test_4200.tsv     0.4 MB  eval only
#     cushr_train/*.py
#
# THREE STAGES. Run the preflight first and read its output before anything
# else -- it is cheap and it catches every environment problem:
#     sbatch --export=ALL,STAGE=preflight train_attn.slurm
#     sbatch --export=ALL,STAGE=train,ENC=lattice_attn_char train_attn.slurm
#     sbatch --export=ALL,STAGE=eval train_attn.slurm
#
# RUNTIME. 8 epochs over 104,159 sentences took ~70 min on CPU for the
# BiLSTM-only model. Attention adds ~306K params but the per-sentence node count
# is small (mean 37.6, p99 104), so on an A100 expect to be data-bound, not
# compute-bound: budget 2 h and raise -t if the log says otherwise.
#----------------------------------------------------------------------
#SBATCH -J attn_train
#SBATCH -o attn_train.o%j
#SBATCH -e attn_train.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

set -euo pipefail

STAGE="${STAGE:-preflight}"
ENC="${ENC:-lattice_attn_char}"
EPOCHS="${EPOCHS:-8}"
SEED="${SEED:-0}"
OBJ="${OBJ:-surface}"              # surface = S model; lm = L+M model
EXTRA="${EXTRA:-}"                 # e.g. EXTRA="--no-span-bias" for the ablation
TAG="${TAG:-${ENC}_${OBJ}_s${SEED}}"
if [ "$OBJ" = surface ]; then TAG="${TAG/_surface_/_}"; fi   # keep old S tag names (set -e safe)

CACHE=./cache95_ngrams80
RAW=../data/cushr_data_g95.npz
SPLITS=splits95_ex4200.json
MODEL="model95_${TAG}.npz"
MATZ="../data/g95_${TAG}.npz"
EVCACHE="./cache95_${TAG}"

module reset 2>/dev/null || module purge
module load cuda
cd "$SLURM_SUBMIT_DIR"

# Kept in $SCRATCH: HOME quotas are small and a torch wheel is ~2.5 GB.
VENV="${SCRATCH:-$PWD}/attn_venv"
if [ ! -d "$VENV" ]; then
    echo "creating venv at $VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    # Match the cluster's CUDA. If `module load cuda` gives 12.x keep cu121.
    "$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu121
    "$VENV/bin/pip" install numpy
fi
PY="$VENV/bin/python"

echo "=== stage=$STAGE encoder=$ENC seed=$SEED epochs=$EPOCHS extra='$EXTRA'"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
       torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

if [ "$STAGE" = "preflight" ]; then
    # 1. the inputs exist
    for f in "$CACHE/node_ids.npy" "$CACHE/surface_text.npy" \
             "$CACHE/node_char_start.npy" "$RAW" "$SPLITS" sighum_test_4200.tsv; do
        [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
        echo "  ok  $f"
    done
    # 2. GPU actually usable
    $PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'"
    # 3. the encoder's invariants hold here, in both dtypes. These failures are
    #    silent in training -- the loss still falls and the score is just worse.
    $PY test_lattice_attn.py --cache "$CACHE"
    $PY test_lattice_attn.py --cache "$CACHE" --double
    # 4. a 200-sentence run end to end, so a typo does not surface an hour in
    $PY train.py --cache "$CACHE" --learned hybrid_tag --encoder "$ENC" \
        --epochs 1 --limit-train 200 --batch 16 --device cuda \
        --splits-override "$SPLITS" \
        --cost "$OBJ" --gold-target "$OBJ" --select-by "${OBJ}_pm" \
        --surface-raw "$RAW" $EXTRA \
        --out /tmp/attn_preflight.npz --log /tmp/attn_preflight.json
    echo "PREFLIGHT OK -- now: sbatch --export=ALL,STAGE=train,ENC=$ENC train_attn.slurm"
    exit 0
fi

if [ "$STAGE" = "train" ]; then
    # Same recipe as cuSHR-seg (COMPARING_MODELS.md sec 9); only --encoder changes.
    $PY train.py --cache "$CACHE" --learned hybrid_tag --encoder "$ENC" \
        --word-dropout 0.1 --epochs "$EPOCHS" --seed "$SEED" --device cuda \
        --splits-override "$SPLITS" \
        --cost "$OBJ" --gold-target "$OBJ" --select-by "${OBJ}_pm" \
        --surface-raw "$RAW" $EXTRA \
        --out "$MODEL" --log "log95_${TAG}.json" --materialize "$MATZ"
    echo "wrote $MODEL and $MATZ -- now: sbatch --export=ALL,STAGE=eval,TAG=$TAG train_attn.slurm"
    exit 0
fi

if [ "$STAGE" = "eval" ]; then
    $PY prepare.py --npz "$MATZ" --cache "$EVCACHE" --force

    echo "=== base decoder (compare against 92.76 PM)"
    $PY eval_surface.py --cache "$EVCACHE" --model "$MODEL" \
        --index ../data/sentence_index_g95.json --raw "$RAW" \
        --json-out "eval_${TAG}_base.json"

    echo "=== + segmentation reranker (compare against 93.43 PM)"
    $PY make_rerank_data.py --cache "$EVCACHE" --model "$MODEL" \
        --splits-override "$SPLITS" --label surface --dedup-surface \
        --k 16 --k-decode 64 --limit 30000 --out-prefix "segrr_${TAG}"
    for s in 0 1 2; do
        $PY rerank.py --train "segrr_${TAG}_train.npz" \
            --dev "segrr_${TAG}_dev.npz" --test "segrr_${TAG}_test.npz" \
            --seg-features --epochs 20 --patience 6 --select-by loss --seed "$s" \
            --out "segrr_${TAG}_s${s}.pt"
        $PY eval_surface.py --cache "$EVCACHE" --model "$MODEL" \
            --index ../data/sentence_index_g95.json --raw "$RAW" \
            --kbest 16 --k-decode 64 --dedup-surface \
            --rerank "segrr_${TAG}_s${s}.pt" \
            --json-out "eval_${TAG}_rerank_s${s}.json"
    done
    echo "report the 3-seed mean, not the best seed"
    exit 0
fi

echo "unknown STAGE=$STAGE (preflight|train|eval)"; exit 1
