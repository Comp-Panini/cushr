#!/bin/bash
#----------------------------------------------------------------------
# train.slurm -- Week-9 biaffine scorer training on Lonestar6.
#
# The model is ~11K parameters and the decode is a level sweep over small
# graphs, so this is not a compute-heavy job -- it fits comfortably in a
# gpu-a100-small slot and finishes in minutes, not hours. The GPU mostly buys
# faster scatter_reduce over the batched edge list.
#
# One-time setup on the login node (the cache build reads the 1.5 GB npz and
# writes ~1 GB of .npy, so do it once and not inside the job):
#   module load python3
#   python3 -m pip install --user torch
#   cd cushr_train && python3 prepare.py --npz ../data/new_cushr_data_fixed_USE_THIS.npz
#   python3 smoke_test.py          # numpy-only, no GPU needed
#
# Submit:  sbatch train.slurm
# Output:  train_biaffine.o<jobid>, model_biaffine.npz, train_log.json
#----------------------------------------------------------------------
#SBATCH -J train_biaffine
#SBATCH -o train_biaffine.o%j
#SBATCH -e train_biaffine.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

set -euo pipefail

module load python3 || true
cd "$SLURM_SUBMIT_DIR"

echo "=== node: $(hostname) ==="
nvidia-smi || true
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [ ! -f cache/splits.json ]; then
    echo "cache/splits.json missing -- run prepare.py on the login node first" >&2
    exit 1
fi

# Baselines on the same split, so the writeup compares like with like.
python3 baseline.py --cache ./cache --out baseline_results.json

python3 train.py \
    --cache ./cache \
    --epochs 10 \
    --batch 64 \
    --lr 1e-3 \
    --margin 1.0 \
    --hidden 128 \
    --out model_biaffine.npz \
    --log train_log.json

# Flat blob for cushr_cpu's BiaffineScorer and for the week-10 K4 kernels, which
# read the same CSB3 .bin. The --edge-scores table is NOT what the GPU consumes
# -- K4 computes edge scores on the device -- but check_export.py uses it to
# confirm the exported weights reproduce the trained model.
python3 export_weights.py \
    --model model_biaffine.npz \
    --bin model_biaffine.bin \
    --cache ./cache \
    --edge-scores edge_score.npy

echo "=== done ==="
