#!/bin/bash
#----------------------------------------------------------------------
# profile_kbest.slurm — Week-7 Nsight Compute profiling of the K-best
# merge kernel on Lonestar6.
#
# Profiles kbest_merge_level for K=32 and K=64 over a small sentence
# sample and exports:
#   - ncu_kbest_K32.csv / ncu_kbest_K64.csv  (parsed by make_benchmark_md.py)
#   - ncu_kbest_K32.ncu-rep / K64.ncu-rep    (open in the Nsight Compute GUI)
#
# Metrics of interest for KBEST_BENCHMARK.md:
#   sm__warps_active.avg.pct_of_peak_sustained_active            (occupancy)
#   smsp__average_warps_issue_stalled_*                          (warp stalls)
#   dram__throughput.avg.pct_of_peak_sustained_elapsed           (memory BW)
#
# Submit:  sbatch profile_kbest.slurm
#----------------------------------------------------------------------
#SBATCH -J kbest_ncu
#SBATCH -o kbest_ncu.o%j
#SBATCH -e kbest_ncu.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
###SBATCH -A your_allocation
#----------------------------------------------------------------------

set -euo pipefail

module purge
module load cuda

cd "$SLURM_SUBMIT_DIR"

DATA=/scratch/11503/njhavar/cushr/data/new_cushr_data_fixed_USE_THIS.npz

# Output tag so a new run does not clobber previous profiles. Override on submit
# with:  sbatch --export=ALL,TAG=mylabel profile_kbest.slurm
# Defaults to a timestamp (e.g. ncu_kbest_K64_20260721_1830.csv).
TAG="${TAG:-$(date +%Y%m%d_%H%M)}"

echo "==== build ===="
make clean
make cushr_kbest
echo

# Small --limit so ncu (heavy instrumentation) finishes quickly; --launch-count
# caps how many kernel launches are profiled.
for K in 32 64; do
    echo "==== ncu profile K=$K ===="
    ncu --set full \
        --launch-count 200 \
        --kernel-name-base demangled \
        --kernel-name "regex:kbest_merge_level" \
        --csv --log-file "ncu_kbest_K${K}_${TAG}.csv" \
        -o "ncu_kbest_K${K}_${TAG}" -f \
        ./cushr_kbest "$DATA" --scorer log_linear --K "$K" --limit 200
    echo
done

echo "==== done ===="
