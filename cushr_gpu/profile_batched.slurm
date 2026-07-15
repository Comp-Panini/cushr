#!/bin/bash
#----------------------------------------------------------------------
# profile_batched.slurm — Week-8 Nsight Compute profiling of the batched
# k-best decode on Lonestar6.
#
# The batched driver reuses the SAME merge kernel (kbest_merge_level) as the
# K2 driver, but launches it over cross-sentence worklists. This profiles that
# kernel as it runs inside the batched sweep and exports:
#   - ncu_batched_K32.csv / ncu_batched_K64.csv  (parsed by make_benchmark_md.py)
#   - ncu_batched_K32.ncu-rep / K64.ncu-rep      (open in the Nsight Compute GUI)
#
# Metrics of interest (injected into BATCHED_BENCHMARK.md):
#   sm__warps_active.avg.pct_of_peak_sustained_active            (occupancy)
#   smsp__average_warps_issue_stalled_*                          (warp stalls)
#   dram__throughput.avg.pct_of_peak_sustained_elapsed           (memory BW)
#
# Filenames are DISTINCT from profile_kbest.slurm's (ncu_kbest_K*), so the
# Week-7 profiling artifacts are never overwritten.
#
# Submit:  sbatch profile_batched.slurm
#----------------------------------------------------------------------
#SBATCH -J batched_ncu
#SBATCH -o batched_ncu.o%j
#SBATCH -e batched_ncu.e%j
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

echo "==== build ===="
make clean
make cushr_batched
echo

# --check 0 skips the CPU oracle (we only want kernel counters here), and
# --batch 1024 bounds device memory. --launch-count caps how many kernel
# launches ncu instruments so the heavy profiling finishes quickly.
for K in 32 64; do
    echo "==== ncu profile K=$K (batched) ===="
    ncu --set full \
        --launch-count 200 \
        --kernel-name-base demangled \
        --kernel-name "regex:kbest_merge_level" \
        --csv --log-file "ncu_batched_K${K}.csv" \
        -o "ncu_batched_K${K}" -f \
        ./cushr_batched "$DATA" --scorer log_linear --K "$K" --batch 1024 --check 0
    echo
done

echo "==== done ===="
echo "Fold these into BATCHED_BENCHMARK.md with:"
echo "  python make_benchmark_md.py --bench batched_bench.csv \\"
echo "      --title \"cuSHR Batched (K3 + K5) Results — Week 8\" \\"
echo "      --md BATCHED_BENCHMARK.md --inject \\"
echo "      --recall-png batched_recall_vs_k.png --thru-png batched_throughput_vs_k.png \\"
echo "      --ncu-prefix ncu_batched"
