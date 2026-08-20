#!/bin/bash
#----------------------------------------------------------------------
# bench_k4.slurm — Week-10 / CP-5: the learned scorer running inside the
# pipeline, on Lonestar6.
#
# Every GPU benchmark before this one scored edges with the hand-tuned 43-weight
# LogLinearScorer (see bench_batched.slurm) because make_scorer() in the GPU
# drivers could not build anything else. That is why BATCHED_BENCHMARK.md's
# recall@64 is 0.3679 and why those numbers must never be quoted alongside the
# trained model's. This job replaces the scorer with the trained biaffine and
# computes it on the device (K4), so the pipeline is
#     load batch -> K4 (score edges) -> K3 (topo sweep) -> K5 (reconstruct)
#
# THE FOUR ROWS THIS JOB EXISTS TO PRODUCE, and what each one answers:
#
#   A. uniform,    --k4 host      the UNSCORED decoder -- precisely what the
#                                 "at most 2x slowdown" target is stated
#                                 against. Measured in this same job on this
#                                 same node, so the ratio to C is a real ratio
#                                 and not a comparison across jobs.
#
#                                 NOT log_linear: the 43 hand-tuned weights in
#                                 bench_batched.slurm belong to the old
#                                 feat_dim=43 lattice, and LogLinearScorer
#                                 throws when weights.size() != lat.feat_dim()
#                                 (cushr_cpu/src/scorer.cpp:29). This corpus is
#                                 feat_dim=192, so passing them would abort the
#                                 job -- and passing 192 invented weights would
#                                 measure nothing.
#   B. biaffine,   --k4 host      the trained model, scored on the CPU and
#                                 uploaded. Same F1 as row C, no K4. Isolates
#                                 "what does the learned model cost the
#                                 decoder" from "what does scoring on the
#                                 device cost".
#   C. biaffine,   --k4 twopass   the actual week-10 deliverable.
#   D. biaffine,   --k4 fused     the literal one-warp-per-edge kernel from the
#                                 spec. C/D is the measured price of projecting
#                                 each node once per edge instead of once.
#
# The target is C within 2x of A on sent_per_sec_kernel, and >= 25,000
# sentences/sec at K=32. C and D must produce IDENTICAL scores (the kernels
# reduce in the same order by construction, and cushr_score_tests asserts it),
# so any difference in recall@K between them is a bug, not a tradeoff.
#
# PREREQUISITES. Both are cheap; neither retrains anything.
#
#   1. A lattice .npz with MATERIALIZED node features. The headline model scores
#      over a char-BiLSTM and no kernel can run that encoder -- but that
#      materialization ALREADY HAPPENED. PAPER_COMPARISON.md:693-697 shows
#      model95_ctx_ex4200 was trained with
#          --materialize ../data/g95_ctx_ex4200.npz
#      and cache95_ctx_ex4200 was then built from that file by prepare.py, so
#      its node_features are already 192 dense columns (96 hybrid_tag + 96
#      char_bilstm). Verified: the model's projections applied straight to those
#      columns reproduce the documented base top-1, and zeroing either
#      96-column block collapses it. See K4_BENCHMARK.md for the numbers.
#
#      The intermediate g95_ctx_ex4200.npz was deleted (~3.3 GiB), so repack the
#      cache back into the single .npz Lattice::load_npz wants:
#        cd ../cushr_train
#        python cache_to_lattice_npz.py --cache ./cache95_ctx_ex4200 --out ../data/g95_ctx_mat.npz
#      That is a repack, not a conversion: same arrays, same names, same dtypes.
#
#   2. Export the weights the kernel reads. This writes the CSB3 .bin that both
#      the CPU BiaffineScorer and K4 read. --edge-scores is NOT needed --
#      precomputing those per-edge scores is exactly what K4 replaces:
#        python export_weights.py --model model95_ctx_ex4200.npz --bin model95_ctx.bin
#
#   3. The .bin and the .npz must agree on feat_dim (192 here). The driver
#      checks this and refuses the run otherwise.
#
# Submit:  sbatch bench_k4.slurm
# Output:  k4_bench.o<jobid>, k4_bench_*.csv, gpu_k32_K32.npz
#----------------------------------------------------------------------
#SBATCH -J k4_bench
#SBATCH -o k4_bench.o%j
#SBATCH -e k4_bench.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
###SBATCH -A your_allocation
#----------------------------------------------------------------------

set -euo pipefail

module purge
module load cuda

cd "$SLURM_SUBMIT_DIR"

# The MATERIALIZED lattice (prerequisite 1), not the raw one.
DATA=/scratch/11503/njhavar/cushr/data/g95_ctx_mat.npz
MODEL=../cushr_train/model95_ctx.bin

KLIST=1,5,16,24,32,48,64

echo "==== node / gpu ===="
hostname
nvidia-smi
echo

echo "==== build ===="
# No `make clean`: other jobs share $SLURM_SUBMIT_DIR and would lose their
# binaries mid-run. See the same note in bench_batched.slurm.
make cushr_batched cushr_score_tests
echo

echo "==== K4 unit tests ===="
./cushr_score_tests
echo

# ---- correctness first, on a small check, before spending hours on timing ----
#
# --check verifies the GPU's top-K score multiset against the CPU TopKDecoder
# running the SAME biaffine model. K4 reduces in a different order than the CPU
# scorer, so this is a tolerance comparison (--tol, default 1e-4) rather than
# the bit-exact one the log_linear path uses; the path and count comparisons
# are still exact.
echo "==== smoke: biaffine + K4 twopass, batch=8192, check 2000 ===="
./cushr_batched "$DATA" --scorer biaffine --model "$MODEL" --k4 twopass \
    --K "$KLIST" --batch 8192 --check 2000 --csv k4_bench_smoke.csv
echo

echo "==== smoke: biaffine + K4 fused (must match twopass exactly) ===="
./cushr_batched "$DATA" --scorer biaffine --model "$MODEL" --k4 fused \
    --K "$KLIST" --batch 8192 --check 2000 --csv k4_bench_smoke_fused.csv
echo

# ---- the four throughput rows, whole-corpus sweep ----
#
# --check 0 throughout: the driver's CPU oracle decodes all S sentences once per
# K no matter how small --check is, and with the biaffine that CPU decode is far
# more expensive than the GPU run being measured. Correctness came from the
# smoke runs above and the check-everything run at the end.
echo "==== A: uniform, host-scored (the UNSCORED decoder baseline) ===="
./cushr_batched "$DATA" --scorer uniform --k4 host \
    --K "$KLIST" --batch -1 --check 0 --csv k4_bench_A_unscored.csv
echo

echo "==== B: biaffine, host-scored (learned model, no K4) ===="
./cushr_batched "$DATA" --scorer biaffine --model "$MODEL" --k4 host \
    --K "$KLIST" --batch -1 --check 0 --csv k4_bench_B_biaffine_host.csv
echo

echo "==== C: biaffine, K4 twopass (the deliverable) ===="
./cushr_batched "$DATA" --scorer biaffine --model "$MODEL" --k4 twopass \
    --K "$KLIST" --batch -1 --check 0 --csv k4_bench_C_biaffine_twopass.csv
echo

echo "==== D: biaffine, K4 fused (spec-literal, redundant projection) ===="
./cushr_batched "$DATA" --scorer biaffine --model "$MODEL" --k4 fused \
    --K "$KLIST" --batch -1 --check 0 --csv k4_bench_D_biaffine_fused.csv
echo

# ---- the F1 run: full verification + candidate dump at K=32 ----
#
# --check -1 makes recall@K the full-corpus number, directly comparable to the
# K2/K3 recall in KBEST_BENCHMARK.md. --dump-paths writes the top-32 node
# sequences for EVERY sentence; feed them to eval_slm.py via
#   cd ../cushr_train
#   python gpu_paths_to_rerank.py --gpu ../cushr_gpu/gpu_k32_K32.npz \
#       --cache ./cache95_ctx_ex4200 --out gpu_rerank_k32.npz
# and cross-check that script's printed recall@32 against the number below.
echo "==== E: K=32, check ALL, dump candidate paths ===="
./cushr_batched "$DATA" --scorer biaffine --model "$MODEL" --k4 twopass \
    --K 32 --batch -1 --check -1 --csv k4_bench_E_k32_checked.csv \
    --dump-paths gpu_k32.npz
echo

echo "==== CSVs ===="
for f in k4_bench_*.csv; do echo "--- $f"; cat "$f"; echo; done
echo "==== done ===="
