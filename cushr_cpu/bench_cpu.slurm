#!/bin/bash
#----------------------------------------------------------------------
# bench_cpu.slurm — Week-11 / CP-6: the CPU baseline, measured.
#
# WHY THIS JOB EXISTS
#
# The paper's headline claim is a GPU speedup, and the denominator of that
# speedup has never been measured. The only CPU throughput figure anywhere in
# the repo is prose:
#
#     cushr_cpu/README.md:55   "~100 sentences/sec single-threaded"
#
# That is a DESIGN TARGET written before the decoder existed, not a
# measurement. cushr_evaluate has always printed a real timing line --
#
#     Decoded <N> sentences in <s> s  (<r> sent/s)
#
# -- but only to stdout, so it was never captured anywhere a results table
# could read. Week 11 adds `--csv` to apps/evaluate.cpp to write that row to
# disk. This job runs it.
#
# RUN IT ON A COMPUTE NODE, which is the entire point of the SBATCH block
# below. A login node is shared with every other user on the system and its
# timings are meaningless; worse, they would be quoted against GPU numbers
# taken from a dedicated A100 node. Same corpus, same weights, same machine
# class is what makes the ratio a ratio.
#
# WHY --K 1
#
# The GPU rows report K=32 because a beam is what the k-best merge exists to
# produce. The CPU comparison the paper needs is top-1 decode throughput --
# the cheapest thing the CPU can do -- because that is the fairest denominator
# and the one that makes the speedup claim conservative rather than inflated.
# If you also want a CPU K=32 row for the same-K comparison, run the second
# invocation below; expect it to be very slow, which is itself the result.
#
# COLUMN NAMES: the CSV writes `sent_per_sec`, NOT the GPU CSVs'
# `sent_per_sec_kernel`. The names differ deliberately -- this is end-to-end
# wall clock and the GPU figure excludes host time, so silently equating the
# two would overstate the speedup. make_results_matrix.py reads them
# separately for exactly that reason.
#
# OUTPUT to copy back to cushr_cpu/:  cpu_bench.csv
#
# THEN, on the laptop, point the manifest at it:
#   cushr_train/results_manifest.json  ->  "bench": { "cpu_csv":
#       "../cushr_cpu/cpu_bench.csv" }
#   cd cushr_train && python make_results_matrix.py
# That fills the cushr_cpu row and puts a second point on the Pareto plot.
#----------------------------------------------------------------------
#SBATCH -J cpu_bench
#SBATCH -o cpu_bench.o%j
#SBATCH -e cpu_bench.e%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

set -euo pipefail

module purge

cd "$SLURM_SUBMIT_DIR"

DATA=/work/11503/njhavar/ls6/cushr/data/g95_ctx_mat.npz
MODEL=../cushr_train/model95_ctx.bin

echo "=== preflight ==="
for f in "$DATA" "$MODEL"; do
    if [ ! -f "$f" ]; then
        echo "MISSING: $f" >&2
        exit 1
    fi
    ls -la "$f"
done
lscpu | grep -E "^Model name|^CPU\(s\):|^Thread|^Core"
echo

# The --csv flag is new in Week 11 and has NOT been compiled anywhere yet, so
# a compile error here is expected-if-unlucky rather than surprising. It is a
# self-contained block at the end of main() in apps/evaluate.cpp.
echo "=== build ==="
make cushr_evaluate
echo

echo "=== top-1 decode, whole corpus ==="
time ./cushr_evaluate "$DATA" --scorer biaffine --model "$MODEL" \
    --K 1 --csv cpu_bench.csv

echo
echo "=== results ==="
cat cpu_bench.csv

# Optional same-K comparison against the GPU's headline beam. Uncomment if you
# want the K=32 row; it is much slower than the K=1 run above.
#
# echo
# echo "=== K=32 decode, whole corpus ==="
# time ./cushr_evaluate "$DATA" --scorer biaffine --model "$MODEL" \
#     --K 32 --csv cpu_bench_k32.csv
# cat cpu_bench_k32.csv

echo
echo "Copy cpu_bench.csv back to cushr_cpu/."
