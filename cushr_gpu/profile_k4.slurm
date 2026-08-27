#!/bin/bash
#----------------------------------------------------------------------
# profile_k4.slurm — Week-10 / CP-5 Nsight Compute profiling of the K4 edge
# scorer on Lonestar6.
#
# WHAT THIS ANSWERS THAT bench_k4.slurm DOES NOT
# ----------------------------------------------
# bench_k4.slurm measures WHICH K4 variant is faster. It cannot say WHY, and
# "why" is the whole design question here:
#
#   fused    -- one warp per edge, re-projects both endpoints from scratch.
#               17.6x the FLOPs (a node sources 17.6 edges on average).
#               Predicted: compute bound.
#   twopass  -- project each node once into S/D, then one warp per edge does a
#               length-128 dot. Far fewer FLOPs, but streams S and D out of
#               global memory for every edge.
#               Predicted: bandwidth bound.
#
# Three things make the wall-clock number alone insufficient:
#
#   1. The driver's cudaEvents wrap BOTH twopass kernels in a single span, so
#      us_per_sent_k4 cannot separate project_nodes (the math) from
#      score_edges_twopass (the streaming). ncu reports per kernel.
#
#   2. There was a real confound in the fused path: a slow fused kernel could be
#      redundant FLOPs OR cache thrashing on W (192 KiB of matrices against the
#      A100's 192 KB per-SM L1/shared). Same wall-clock signature, completely
#      different finding. RESOLVED by the first run: it is NEITHER. L1 hit rate
#      is 82.6% on the fused kernel and 98.3% on project_nodes, DRAM throughput
#      is 0.01%, and compute is 5.4%. Nothing thrashes and nothing computes --
#      see the answer below.
#
#   3. "Memory bound" needs a number, not an assertion. The pair
#      sm__throughput vs dram__throughput as percentages of peak is the actual
#      evidence, and it is what decides whether the "tensor cores are overkill"
#      claim in score_edges.cuh survives.
#
# THE METRICS THAT MATTER HERE (--set full collects all of them, plus roofline)
#   sm__throughput.avg.pct_of_peak_sustained_elapsed      compute utilisation
#   dram__throughput.avg.pct_of_peak_sustained_elapsed    memory utilisation
#       -> whichever is near peak is the bottleneck. Both low means latency
#          bound (occupancy / stalls), which the next two explain.
#   sm__warps_active.avg.pct_of_peak_sustained_active     achieved occupancy
#   smsp__average_warps_issue_stalled_*                   why warps stall
#   l1tex__t_sector_hit_rate.pct                          is W actually cached?
#   lts__t_sector_hit_rate.pct                            L2 fallback
#
# THE ANSWER, from the run of 2026-08-20 (A100, --set full, 100/115 launches).
# The three predictions originally written here were ALL WRONG; they are kept
# below so the correction is legible.
#
#   kernel               duration    SM%    L1/TEX%   DRAM%   occ%   L1 hit%
#   project_nodes         24.12 ms   5.98    99.24    0.14    92.8    98.3
#   score_edges_twopass    0.35 ms  59.61    45.11    9.25    85.8    50.1
#   score_edges_fused    413.42 ms   5.38    99.80    0.01    91.6    82.6
#
#   predicted: project_nodes compute bound (it is the matmul)
#     ACTUAL:  5.98% compute. 1% of the device's fp32 peak.
#   predicted: score_edges_twopass bandwidth bound (it streams S and D)
#     ACTUAL:  the MOST compute-heavy of the three, 59.6% SM vs 9.25% DRAM.
#   predicted: score_edges_fused compute bound (17.6x redundant FLOPs)
#     ACTUAL:  5.38% compute, 0.01% DRAM.
#
# Both slow kernels are bound by L1/TEX REQUEST THROUGHPUT at ~99% of peak, with
# the FMA pipes idle and DRAM untouched. Cause: row_dot gave lane l the hidden
# dim l, so the 32 lanes of a warp read 32 different rows of W, 768 B apart --
# 32 sectors per load instruction. project_nodes is 98.6% of twopass's K4 time.
# Fixed by uploading W transposed (4 sectors per load); see the row_dot comment
# in score_edges.cu and K4_BENCHMARK.md. THE TABLE ABOVE PREDATES THAT FIX --
# re-run this job to find out what the new limiter is.
#
# Outputs (TAG keeps runs from clobbering each other):
#   ncu_k4_twopass_<TAG>.ncu-rep / .csv   project_nodes + score_edges_twopass
#   ncu_k4_fused_<TAG>.ncu-rep   / .csv   score_edges_fused
#
# Submit:  sbatch profile_k4.slurm
#          sbatch --export=ALL,TAG=mytag profile_k4.slurm
#
# Prerequisites are the same two as bench_k4.slurm (repacked lattice .npz +
# exported CSB3 .bin); see that file's header.
#----------------------------------------------------------------------
#SBATCH -J k4_ncu
#SBATCH -o k4_ncu.o%j
#SBATCH -e k4_ncu.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
###SBATCH -A your_allocation
#----------------------------------------------------------------------

set -euo pipefail

module purge
module load cuda

cd "$SLURM_SUBMIT_DIR"

DATA=/work/11503/njhavar/ls6/cushr/data/g95_ctx_mat.npz
MODEL=../cushr_train/model95_ctx.bin

TAG="${TAG:-$(date +%Y%m%d_%H%M)}"

echo "==== node / gpu ===="
hostname
nvidia-smi
echo

echo "==== build ===="
# No `make clean` -- $SLURM_SUBMIT_DIR is shared with the benchmark jobs and
# cleaning would delete a binary out from under a running sweep.
make cushr_batched
echo

# --K 32 only, deliberately. K4's output does not depend on K at all: it writes
# edge_score, which K3 then reads for whatever K it was asked for. Profiling the
# full KLIST would instrument seven identical sets of K4 launches.
#
# --batch 1024 bounds device memory and gives ~117 chunks, so there are plenty
# of launches to sample. --check 0 skips the CPU oracle, which is pure host time
# and would otherwise dominate the job's walltime.
#
# --launch-count 200 caps how many launches ncu instruments. --set full is very
# heavy (many replays per launch), and 200 is already far more than needed for
# stable per-kernel counters.
#
# NB, inherited from profile_batched.slurm and worth repeating: -o writes the
# .ncu-rep binary and SUPPRESSES the console report, so adding
# `--csv --log-file X.csv` here would capture only ncu's "==PROF==" chatter and
# not a single metric. Profile to the report, then export the CSV from it below.

echo "==== ncu profile: twopass (project_nodes + score_edges_twopass) ===="
# The regex catches BOTH twopass kernels, which is the point -- their separate
# rows are what split the math from the streaming.
ncu --set full \
    --launch-count 200 \
    --kernel-name-base demangled \
    --kernel-name "regex:project_nodes|score_edges_twopass" \
    -o "ncu_k4_twopass_${TAG}" -f \
    ./cushr_batched "$DATA" --scorer biaffine --model "$MODEL" --k4 twopass \
        --K 32 --batch 1024 --check 0
echo

echo "==== ncu profile: fused (score_edges_fused) ===="
ncu --set full \
    --launch-count 200 \
    --kernel-name-base demangled \
    --kernel-name "regex:score_edges_fused" \
    -o "ncu_k4_fused_${TAG}" -f \
    ./cushr_batched "$DATA" --scorer biaffine --model "$MODEL" --k4 fused \
        --K 32 --batch 1024 --check 0
echo

# Export to the WIDE csv shape (one row per launch, metric names as headers),
# the same shape make_benchmark_md.py's --ncu-prefix parsing expects. This step
# needs no GPU: it can be re-run on a login node against existing .ncu-rep
# files if the export is ever wanted in a different form.
for V in twopass fused; do
    echo "==== export $V -> ncu_k4_${V}_${TAG}.csv ===="
    ncu --import "ncu_k4_${V}_${TAG}.ncu-rep" --csv --page raw > "ncu_k4_${V}_${TAG}.csv"
    wc -l "ncu_k4_${V}_${TAG}.csv"
done
echo

# A human-readable summary of just the decisive metrics, printed into the slurm
# log so the answer is visible without opening the GUI or the wide CSV. The
# --page details view is per-kernel, which is exactly the split that
# us_per_sent_k4 in bench_k4.slurm cannot give.
for V in twopass fused; do
    echo "==== $V: bottleneck summary ===="
    ncu --import "ncu_k4_${V}_${TAG}.ncu-rep" --page details \
        --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct \
        || echo "  (metric subset unavailable in this ncu version; read the CSV)"
    echo
done

echo "==== done ===="
echo "Read it as: whichever of sm__throughput / dram__throughput is closer to"
echo "peak is that kernel's bottleneck. If both are low, the kernel is latency"
echo "bound -- check occupancy and the stall reasons in the full CSV."
echo
echo "Then fold the findings into K4_BENCHMARK.md alongside the C-vs-D"
echo "wall-clock rows from bench_k4.slurm."
