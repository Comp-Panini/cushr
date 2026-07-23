#!/bin/bash
#----------------------------------------------------------------------
# bench_batched.slurm — Week-8 batched K-best benchmark on Lonestar6.
#
# Runs cushr_batched (K3 global topo sweep + K5 path reconstruction) over the
# WHOLE SIGHUM dataset at K in {1,5,16,24,32,48,64}. The batched driver processes
# many sentences per level launch, so throughput should be far above the
# per-sentence K2 driver (Week-7 ~2,775 sent/sec at K=32).
#
# We sweep --batch from 256 up to 65536 in powers of 2, plus the whole corpus
# (batch=-1, 119,503 sentences), to show the memory-vs-batch tradeoff: each
# chunk's k-best table is sized to just that chunk's node span, so peak device
# memory scales with --batch, not with the whole corpus.
#
# Memory headroom at the top end: the whole-corpus table is span*K*12 B, i.e.
# 1.66 GB at K=32 and 3.30 GB at K=64 (measured), so K=48 lands near 2.5 GB --
# all comfortable on a 40 GB A100.
#
# CSV columns match kbest_bench.csv (make_benchmark_md.py parses either), but the
# batched outputs use DISTINCT filenames (batched_bench*.csv) so the frozen
# Week-7 files (kbest_bench.csv, KBEST_BENCHMARK.md, *_vs_k.png) are never
# overwritten. Turn the CSV into tables/plots with:
#   python make_benchmark_md.py --bench batched_bench.csv \
#       --title "cuSHR Batched (K3 + K5) Results — Week 8" \
#       --md BATCHED_BENCHMARK.md --inject \
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

# K list now includes the non-power-of-2 points 24 and 48. These need no kernel
# change: kbest_merge_level templates on the *capacity* (32 or 64), which is the
# power of 2 the bitonic network needs, while kb.K only bounds how many ranks get
# written back. So K=24 runs the cap-32 network and K=48 the cap-64 one.
#
# That is exactly why they are worth measuring: if throughput is set by the cap
# and not by K, then 24 should cost the same as 32 and 48 the same as 64, making
# the throughput-vs-K curve a step function rather than a slope. The k-best table
# still scales linearly in K, so memory should NOT step -- it should sit between
# the neighbouring powers of 2. Those two predictions are the point of the run.
KLIST=1,5,16,24,32,48,64

# Smoke + correctness for the new K values, at a chunked batch size so the
# chunk-boundary logic is exercised too.
echo "==== smoke: batch=8192, check 2000 ===="
./cushr_batched "$DATA" --scorer log_linear --weights "$WEIGHTS" \
    --K "$KLIST" --batch 8192 --check 2000 --csv batched_bench_smoke.csv
echo

# Memory-vs-batch sweep, 256 .. 65536 in powers of 2 (65536 is the last power of
# 2 below the 119,503-sentence corpus; the whole-corpus point is the batch=-1 run
# below, which is the real "119K" end of the sweep).
#
# --check 0 here, deliberately: the CPU oracle inside the driver decodes ALL
# sentences once per K regardless of how small --check is, so any non-zero check
# would add a full-corpus CPU decode per K per batch size and blow the walltime.
# These runs exist to measure throughput and device memory; correctness is
# covered by the smoke run above and the check-everything run below.
for B in 256 1024 4096 8192 16384 32768 65536; do
    echo "==== full run: batch=$B (throughput/memory only) ===="
    ./cushr_batched "$DATA" --scorer log_linear --weights "$WEIGHTS" \
        --K "$KLIST" --batch "$B" --check 0 --csv "batched_bench_b${B}.csv"
    echo
done

# Whole-corpus single sweep (max batch) = headline throughput number.
# --check -1 verifies EVERY sentence, so recall@K here is the full-corpus number
# and is directly comparable to the K2 recall in KBEST_BENCHMARK.md (they must be
# identical: the batched decode is score-equivalent). The CPU oracle already
# decodes all S regardless of --check, so this costs only the compare loop.
echo "==== full run: batch=-1 (whole corpus), check ALL ===="
./cushr_batched "$DATA" --scorer log_linear --weights "$WEIGHTS" \
    --K "$KLIST" --batch -1 --check -1 --csv batched_bench.csv

echo
echo "==== batched_bench.csv ===="
cat batched_bench.csv
echo
echo "==== done ===="
