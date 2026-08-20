#!/bin/bash
#----------------------------------------------------------------------
# test_k4.slurm — build + run the K4 (score_edges) unit tests on an A100.
#
# Synthetic graphs and random weights, no npz and no CPU library needed. Checks
# that both K4 kernels match a host reference, that they match EACH OTHER
# bitwise, and that the CSB3 loader rejects the same files cushr_cpu rejects.
# See tests/test_score_edges.cu for why each of those matters.
#
# Submit:   sbatch test_k4.slurm
# Output:   k4_tests.o<jobid>, k4_tests_results.txt
#----------------------------------------------------------------------
#SBATCH -J k4_tests
#SBATCH -o k4_tests.o%j
#SBATCH -e k4_tests.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00
###SBATCH -A your_allocation
#----------------------------------------------------------------------

set -euo pipefail

module purge
module load cuda

cd "$SLURM_SUBMIT_DIR"

echo "==== node / gpu ===="
hostname
nvidia-smi
echo

echo "==== build ===="
# No `make clean` -- $SLURM_SUBMIT_DIR is shared with the benchmark jobs and
# cleaning would delete a binary out from under a running sweep.
make cushr_score_tests
echo

echo "==== run: K4 score_edges unit tests ===="
RESULTS=k4_tests_results.txt
./cushr_score_tests | tee "$RESULTS"
status=${PIPESTATUS[0]}

echo
echo "==== results written to $RESULTS (exit $status) ===="
echo "==== done ===="
exit $status
