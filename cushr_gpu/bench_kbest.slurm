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

# Hand-tuned LogLinearScorer weights (one per feat_dim=43), from
# cushr_cpu/feature_weights.py = log(gold_freq / other_freq). WITHOUT these the
# driver defaults every weight to 1.0, which ranks paths near-randomly and makes
# top-K recall meaningless (recall@1 ~5%). With them, recall reflects the real
# scorer. Regenerate with: python ../cushr_cpu/feature_weights.py
WEIGHTS=0.0559,-0.4269,-0.5438,-0.3353,0.5429,-0.2211,-1.4043,0.3214,0.0776,-2.3854,0.1326,0.4872,-1.1342,-0.4812,0.0558,1.3182,-0.0024,-0.6225,1.2320,-1.0156,1.7814,-1.0106,-1.0189,-0.1246,0.5901,0.9213,-1.0925,-1.4549,1.3426,-6.2178,-7.1641,-3.5924,-3.3199,1.6020,-1.6993,1.7712,2.3289,0.3599,1.6179,0.5901,0.9764,-0.9046,0.0000

echo "==== node / gpu ===="
hostname
nvidia-smi
echo

echo "==== build ===="
# NO `make clean` here -- it rm's every binary in the shared submit dir, including
# cushr_batched, which killed a concurrently-running bench_batched.slurm (job
# 3312687). `make cushr_kbest` rebuilds from sources on its own.
make cushr_kbest
echo

# K=24 and 48 added to match bench_batched.slurm, so the K2-vs-K3 comparison
# plot has both curves defined at every K. They need no kernel change: the
# bitonic network is templated on the capacity (32 / 64) and K only bounds the
# write-back, so 24 runs the cap-32 network and 48 the cap-64 one.
#
# NOTE: rerunning this overwrites kbest_bench.csv, the Week-7 "frozen" result.
# The 1/5/16/32/64 rows should come back identical -- if they do not, something
# changed underneath and that is worth knowing before the numbers go in a paper.
KLIST=1,5,16,24,32,48,64

# Fast smoke first (2000 sentences) so a broken run fails in ~1 min, not 4 h.
# Comment this out once you trust the pipeline.
echo "==== smoke: 2000 sentences ===="
./cushr_kbest "$DATA" --scorer log_linear --weights "$WEIGHTS" \
    --K "$KLIST" --limit 2000 --csv kbest_bench_smoke.csv
echo

echo "==== full run: all sentences, writing kbest_bench.csv ===="
./cushr_kbest "$DATA" --scorer log_linear --weights "$WEIGHTS" \
    --K "$KLIST" --limit -1 --csv kbest_bench.csv

echo
echo "==== kbest_bench.csv ===="
cat kbest_bench.csv
echo
echo "==== done ===="
