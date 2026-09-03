#!/bin/bash
#----------------------------------------------------------------------
# bench_k64.slurm — Week-11 / CP-6: the K=64 checked run.
#
# WHY THIS JOB EXISTS
#
# The paper's results table reports top-K recall at K = 1, 5, 16, 32, 64.
# Every artifact we have stops at 32: bench_k4.slurm's row E dumped K=32
# (gpu_k32_K32.npz) and that is the only CHECKED whole-corpus figure in the
# repo (recall@32 = 0.973053, k4_bench_E_k32_checked.csv). The K=64 column
# cannot be filled by extrapolation, so it needs one more run.
#
# The four A/B/C/D sweeps in bench_k4.slurm do cover K=64, but they ran with
# --check 0 and therefore wrote recall_at_K=NA. They measure throughput and
# memory only. This job re-runs K=64 WITH the CPU cross-check so the recall
# number exists and is verified.
#
# EXPECT THIS TO BE SLOW, and not for the obvious reason.
#
# k4_bench_C_biaffine_twopass.csv shows a throughput cliff between K=32 and
# K=48 that is entirely in K3:
#
#     K    sent/sec    us_per_sent_k4   us_per_sent_k3
#     32   464,995     1.092            1.058
#     48   185,587     1.085            4.303      <- 4.1x, K4 unchanged
#     64   182,091     1.087            4.405
#
# K4 is flat, so this is not the scorer. It is the k-best merge losing
# occupancy once the per-thread candidate array stops fitting the register
# budget. So budget ~2.5x the per-sentence time of the K=32 run, and note that
# --check -1 (every sentence, against the CPU decoder) is itself the long pole:
# row E at K=32 took the bulk of its wallclock in the check, not the kernels.
# 6 hours is deliberately generous; the job writes its CSV before it would be
# killed, but the dump comes after, so do not let it time out.
#
# OUTPUTS to copy back to cushr_gpu/ afterwards:
#   k4_bench_F_k64_checked.csv   the recall + throughput + memory row
#   gpu_k64_K64.npz              the K-best paths (~325 MB; K=32 was 163 MB)
#
# The driver appends _K<K> to --dump-paths, which is why the file lands as
# gpu_k64_K64.npz and not gpu_k64.npz.
#
# THEN, back on the laptop:
#   cd cushr_train
#   python gpu_paths_to_rerank.py --gpu ../cushr_gpu/gpu_k64_K64.npz \
#       --cache ./cache95_ctx_ex4200 --raw ../data/cushr_data_g95.npz \
#       --out gpu_rerank_k64.npz
#   # its printed recall@64 must equal recall_at_K in the CSV below. If it does
#   # not, stop: the candidate lists are not comparable. (The same check caught
#   # the n vs n_gold denominator bug in CP-5.)
#
# PREREQUISITES: identical to bench_k4.slurm -- see its header. Both the
# materialized lattice and the exported weights must already exist:
#   ls -la /work/11503/njhavar/ls6/cushr/data/g95_ctx_mat.npz
#   ls -la ../cushr_train/model95_ctx.bin
#----------------------------------------------------------------------
#SBATCH -J k64_bench
#SBATCH -o k64_bench.o%j
#SBATCH -e k64_bench.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00

set -euo pipefail

module purge
module load cuda

cd "$SLURM_SUBMIT_DIR"

DATA=/work/11503/njhavar/ls6/cushr/data/g95_ctx_mat.npz
MODEL=../cushr_train/model95_ctx.bin

echo "=== preflight ==="
for f in "$DATA" "$MODEL"; do
    if [ ! -f "$f" ]; then
        echo "MISSING: $f" >&2
        echo "See the PREREQUISITES block in bench_k4.slurm." >&2
        exit 1
    fi
    ls -la "$f"
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h . | tail -1
echo

echo "=== build ==="
make cushr_batched cushr_score_tests
./cushr_score_tests
echo

# Same scorer, k4 mode and batching as row E of bench_k4.slurm, so this row and
# k4_bench_E_k32_checked.csv differ ONLY in K and are directly comparable.
echo "=== K=64, checked, whole corpus ==="
time ./cushr_batched "$DATA" --scorer biaffine --model "$MODEL" --k4 twopass \
    --K 64 --batch -1 --check -1 --csv k4_bench_F_k64_checked.csv \
    --dump-paths gpu_k64.npz

echo
echo "=== results ==="
cat k4_bench_F_k64_checked.csv
ls -la gpu_k64_K64.npz

echo
echo "score_mismatch and count_mismatch must both be 0 above."
echo "Copy k4_bench_F_k64_checked.csv and gpu_k64_K64.npz back to cushr_gpu/."
