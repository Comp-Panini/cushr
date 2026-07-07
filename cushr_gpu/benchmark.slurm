#!/bin/bash
#----------------------------------------------------------------------
# benchmark.slurm — k-best GPU decode timing + accuracy on Lonestar6.
#
# Builds cushr_kbest and runs it over several K values, reporting the
# per-sentence loop time, kernel-only time, and score-equivalence vs the
# CPU TopKDecoder (epsilon 1e-4).
#
# Submit:   sbatch benchmark.slurm
# Watch:    squeue -u $USER
# Output:   kbest_bench.o<jobid>   (stdout+stderr, see -o/-e below)
#----------------------------------------------------------------------
#SBATCH -J kbest_bench              # job name
#SBATCH -o kbest_bench.o%j          # stdout -> file (%j = job id)
#SBATCH -e kbest_bench.e%j          # stderr -> file
#SBATCH -p gpu-a100-small           # single-A100 queue (1/3 node)
#SBATCH -N 1                        # one node
#SBATCH -n 1                        # one task
#SBATCH -t 00:30:00                 # wall-clock limit HH:MM:SS
###SBATCH -A your_allocation        # uncomment + set if you have >1 allocation
#----------------------------------------------------------------------

set -euo pipefail

module purge
module load cuda

cd "$SLURM_SUBMIT_DIR"

DATA=/scratch/11503/njhavar/cushr/data/new_cushr_data_fixed_USE_THIS.npz

echo "==== node / gpu ===="
hostname
nvidia-smi
echo

echo "==== build ===="
make clean
make cushr_kbest
echo

echo "==== run: k-best decode timing + accuracy ===="
./cushr_kbest "$DATA" --scorer log_linear --K 1,8,16,32,64 --limit 1000

echo
echo "==== done ===="
