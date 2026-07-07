#!/bin/bash
#----------------------------------------------------------------------
# test_kbest.slurm — build + run the k-best merge unit tests on an A100.
#
# Compiles cushr_kbest_tests (synthetic lattices, no npz needed) and runs
# it. Exits non-zero if any CHECK fails, so the job state reflects pass/fail.
#
# Submit:   sbatch test_kbest.slurm
# Watch:    squeue -u $USER
# Output:   kbest_tests.o<jobid>
#----------------------------------------------------------------------
#SBATCH -J kbest_tests
#SBATCH -o kbest_tests.o%j
#SBATCH -e kbest_tests.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00
###SBATCH -A your_allocation        # uncomment + set if you have >1 allocation
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
make clean
make cushr_kbest_tests
echo

echo "==== run: k-best merge unit tests ===="
./cushr_kbest_tests    # returns 0 = ALL PASSED, 1 = a CHECK failed

echo
echo "==== done ===="
