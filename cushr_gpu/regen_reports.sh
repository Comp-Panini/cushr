#!/bin/bash
#----------------------------------------------------------------------
# regen_reports.sh -- rebuild every plot and markdown report from the CSVs.
#
# Run this AFTER copying fresh CSVs back from Lonestar6. It reads only CSVs and
# writes only PNGs + markdown, so it is safe to re-run and needs no GPU. Run it
# wherever you have matplotlib (your laptop is fine).
#
#   ./regen_reports.sh              # batched (week 8) + comparison
#   ./regen_reports.sh --with-k2    # also rebuild the week-7 K2 report
#
# The --sweep list is built from whatever batched_bench_b<N>.csv files exist, so
# adding batch sizes to bench_batched.slurm needs no change here.
#----------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

WITH_K2=0
[ "${1:-}" = "--with-k2" ] && WITH_K2=1

# --- batch-sweep args, numerically sorted, whole-corpus row last -------------
SWEEP_ARGS=()
for f in $(ls batched_bench_b*.csv 2>/dev/null \
           | sed -E 's/batched_bench_b([0-9]+)\.csv/\1 &/' \
           | sort -n | awk '{print $2}'); do
    n=$(echo "$f" | sed -E 's/batched_bench_b([0-9]+)\.csv/\1/')
    SWEEP_ARGS+=(--sweep "${n}=${f}")
done
if [ -f batched_bench.csv ]; then
    SWEEP_ARGS+=(--sweep "-1 (whole corpus)=batched_bench.csv")
fi

echo "==== week 8: batched report ===="
if [ -f batched_bench.csv ]; then
    python3 make_benchmark_md.py \
        --bench batched_bench.csv \
        --title "cuSHR Batched (K3 + K5) Results — Week 8" \
        --md BATCHED_BENCHMARK.md --inject \
        --ncu-prefix ncu_batched \
        --recall-png batched_recall_vs_k.png \
        --thru-png batched_throughput_vs_k.png \
        "${SWEEP_ARGS[@]}"
else
    echo "  skipped: batched_bench.csv not found"
fi
echo

echo "==== week 7: K2 report ===="
if [ "$WITH_K2" = "1" ] && [ -f kbest_bench.csv ]; then
    python3 make_benchmark_md.py \
        --bench kbest_bench.csv \
        --md KBEST_BENCHMARK.md \
        --ncu-prefix ncu_kbest \
        --recall-png recall_vs_k.png \
        --thru-png throughput_vs_k.png
else
    echo "  skipped (pass --with-k2 to rebuild; it overwrites the frozen week-7 report)"
fi
echo

echo "==== K2 vs K3 comparison ===="
python3 compare_k2_k3.py
echo
echo "done. changed files:"
git status --short -- '*.png' '*.md' 2>/dev/null || true
