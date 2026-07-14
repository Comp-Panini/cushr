#!/bin/bash
#----------------------------------------------------------------------
# bench_batched.slurm — Week-8 batched K-best benchmark on Lonestar6.
#
# Runs cushr_batched (K3 global topo sweep + K5 path reconstruction) over the
# WHOLE SIGHUM dataset at K in {1,5,16,32,64}. The batched driver processes many
# sentences per level launch, so throughput should be far above the per-sentence
# K2 driver (Week-7 ~2,775 sent/sec at K=32).
#
# We sweep a few --batch sizes to show the memory-vs-batch tradeoff: each chunk's
# k-best table is sized to just that chunk's node span, so peak device memory
# scales with --batch, not with the whole corpus.
#
# CSV columns match kbest_bench.csv (make_benchmark_md.py parses either), but the
# batched outputs use DISTINCT filenames (batched_bench*.csv) so the frozen
# Week-7 files (kbest_bench.csv, KBEST_BENCHMARK.md, *_vs_k.png) are never
# overwritten. Turn the CSV into tables/plots with:
#   python make_benchmark_md.py --bench batched_bench.csv \
#       --title "cuSHR Batched (K3 + K5) Results — Week 8" --md BATCHED_RESULTS.md \
#       --recall-png batched_recall_vs_k.png --thru-png batched_throughput_vs_k.png
#
# Submit:  sbatch bench_batched.slurm
# Output:  batched_full.o<jobid>  and  batched_bench_b<N>.csv
#----------------------------------------------------------------------
#SBATCH -J batched_full
#SBATCH -o batched_full.o%j
#SBATCH -e batched_full.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
###SBATCH -A your_allocation
#----------------------------------------------------------------------

set -euo pipefail

module purge
module load cuda

cd "$SLURM_SUBMIT_DIR"

DATA=/scratch/11503/njhavar/cushr/data/new_cushr_data_fixed_USE_THIS.npz

# Same hand-tuned LogLinearScorer weights as bench_kbest.slurm (feat_dim=43).
# Regenerate with: python ../cushr_cpu/feature_weights.py
WEIGHTS=0.0559,-0.4269,-0.5438,-0.3353,0.5429,-0.2211,-1.4043,0.3214,0.0776,-2.3854,0.1326,0.4872,-1.1342,-0.4812,0.0558,1.3182,-0.0024,-0.6225,1.2320,-1.0156,1.7814,-1.0106,-1.0189,-0.1246,0.5901,0.9213,-1.0925,-1.4549,1.3426,-6.2178,-7.1641,-3.5924,-3.3199,1.6020,-1.6993,1.7712,2.3289,0.3599,1.6179,0.5901,0.9764,-0.9046,0.0000

echo "==== node / gpu ===="
hostname
nvidia-smi
echo

echo "==== build ===="
make clean
make cushr_batched
echo

# Smoke first: whole-corpus sweep, correctness-check the first 2000 sentences.
echo "==== smoke: batch=1024, check 2000 ===="
./cushr_batched "$DATA" --scorer log_linear --weights "$WEIGHTS" \
    --K 1,5,16,32,64 --batch 1024 --check 2000 --csv batched_bench_smoke.csv
echo

# Memory-vs-batch sweep. --check 1000 keeps verification cheap; throughput is
# still measured over the whole corpus (all chunks are swept).
for B in 256 1024 4096; do
    echo "==== full run: batch=$B ===="
    ./cushr_batched "$DATA" --scorer log_linear --weights "$WEIGHTS" \
        --K 1,5,16,32,64 --batch "$B" --check 1000 --csv "batched_bench_b${B}.csv"
    echo
done

# Whole-corpus single sweep (max batch) = headline throughput number.
echo "==== full run: batch=-1 (whole corpus) ===="
./cushr_batched "$DATA" --scorer log_linear --weights "$WEIGHTS" \
    --K 1,5,16,32,64 --batch -1 --check 1000 --csv batched_bench.csv

echo
echo "==== batched_bench.csv ===="
cat batched_bench.csv
echo
echo "==== done ===="
