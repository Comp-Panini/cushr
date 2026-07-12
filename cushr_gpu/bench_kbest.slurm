#!/bin/bash
#----------------------------------------------------------------------
# bench_kbest.slurm — Week-7 full-dataset K-best benchmark on Lonestar6.
#
# Runs cushr_kbest over the WHOLE SIGHUM dataset at K in {1,5,16,32,64},
# writing a machine-readable kbest_bench.csv (one row per K) that
# make_benchmark_md.py turns into KBEST_BENCHMARK.md + plots.
#
# Columns: K,n_sentences,n_gold,recall_at_K,us_per_sent_loop,
#          us_per_sent_kernel,sent_per_sec_kernel,gpu_table_MB,
#          gpu_used_MB,score_mismatch,count_mismatch
#
# Submit:  sbatch bench_kbest.slurm
# Watch:   squeue -u $USER
# Output:  kbest_full.o<jobid>  and  kbest_bench.csv
#----------------------------------------------------------------------
#SBATCH -J kbest_full
#SBATCH -o kbest_full.o%j
#SBATCH -e kbest_full.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00                 # full 119K decode on CPU+GPU is slow; bump if needed
###SBATCH -A your_allocation
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

# Fast smoke first (2000 sentences) so a broken run fails in ~1 min, not 4 h.
# Comment this out once you trust the pipeline.
echo "==== smoke: 2000 sentences ===="
./cushr_kbest "$DATA" --scorer log_linear --K 1,5,16,32,64 --limit 2000 \
    --csv kbest_bench_smoke.csv
echo

echo "==== full run: all sentences, writing kbest_bench.csv ===="
./cushr_kbest "$DATA" --scorer log_linear --K 1,5,16,32,64 --limit -1 \
    --csv kbest_bench.csv

echo
echo "==== kbest_bench.csv ===="
cat kbest_bench.csv
echo
echo "==== done ===="
