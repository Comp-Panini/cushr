#!/usr/bin/env python3
"""make_benchmark_md.py -- turn the Week-7 benchmark CSVs into KBEST_BENCHMARK.md.

Inputs (all optional except the timing CSV):
  kbest_bench.csv        written by cushr_kbest --csv (one row per K)
  ncu_kbest_K<K>.csv     Nsight Compute --csv exports (profile_kbest.slurm)

Outputs (written next to this script):
  KBEST_BENCHMARK.md     the CP-4 deliverable
  recall_vs_k.png        top-K recall curve
  throughput_vs_k.png    sentences/sec vs K

Runs anywhere with Python + numpy + matplotlib -- no GPU needed. The ncu parsing
is best-effort: if the profiling CSVs are absent or in an unexpected shape, the
markdown is still generated from the timing CSV and the Nsight section is left
as a TODO.

Usage:
  # Week-7 per-sentence (K2): writes KBEST_BENCHMARK.md + recall_vs_k.png + throughput_vs_k.png
  python make_benchmark_md.py                          # defaults, cwd = cushr_gpu/
  python make_benchmark_md.py --bench kbest_bench.csv --outdir .

  # Week-8 batched (K3+K5): separate outputs, leaves the Week-7 files untouched
  python make_benchmark_md.py --bench batched_bench.csv \
      --title "cuSHR Batched (K3 + K5) Benchmark — Week 8" \
      --md BATCHED_BENCHMARK.md \
      --recall-png batched_recall_vs_k.png --thru-png batched_throughput_vs_k.png
"""
import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- timing / recall CSV -------------------------------------------------
def load_bench(path):
    """Return list of per-K dicts, sorted by K. recall_at_K may be None (NA)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            def num(key, cast=float):
                v = r.get(key, "").strip()
                if v == "" or v.upper() == "NA":
                    return None
                try:
                    return cast(v)
                except ValueError:
                    return None
            rows.append({
                "K": int(r["K"]),
                "n_sentences": num("n_sentences", int),
                "n_gold": num("n_gold", int),
                "recall": num("recall_at_K"),
                "us_loop": num("us_per_sent_loop"),
                "us_kernel": num("us_per_sent_kernel"),
                "sent_per_sec": num("sent_per_sec_kernel"),
                "table_MB": num("gpu_table_MB"),
                "used_MB": num("gpu_used_MB"),
                "score_mismatch": num("score_mismatch", int),
                "count_mismatch": num("count_mismatch", int),
                # batched-only; None for the frozen K2 CSV, which lacks them.
                "n_check": num("n_check", int),
                "n_chunks": num("n_chunks", int),
                "n_launches": num("n_launches", int),
            })
    rows.sort(key=lambda d: d["K"])
    return rows


# ---- Nsight Compute CSV --------------------------------------------------
# Metric-name substrings we care about (ncu names vary by version, so match loosely).
OCC_KEYS  = ["sm__warps_active.avg.pct_of_peak_sustained_active", "Achieved Occupancy"]
DRAM_KEYS = ["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
             "dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM Throughput"]
# Canonical Nsight "Warp State" stall metrics: avg warps stalled for each reason
# per issue-active cycle. Prefer these over pcsamp sampling columns.
STALL_RE  = re.compile(r"average_warps_issue_stalled_.+_per_issue_active", re.IGNORECASE)


def _to_float(s):
    try:
        return float(str(s).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def parse_ncu_csv(path):
    """Best-effort: return {metric_name: mean_value} averaged over launches.

    Handles both ncu CSV shapes:
      * LONG  -- one row per (kernel, metric) with "Metric Name"/"Metric Value"
                 columns (ncu --csv default / --page details).
      * WIDE  -- one row per kernel launch, one column per metric (the metric
                 name IS the column header). This is what `ncu --import
                 report.ncu-rep --csv --page raw` produces. A units row (with a
                 blank Kernel Name) and non-numeric attribute columns are simply
                 skipped because their values don't parse as floats.
    """
    acc = {}
    cnt = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        name_col = next((c for c in fields if c.strip().lower() == "metric name"), None)
        val_col  = next((c for c in fields if c.strip().lower() == "metric value"), None)
        if name_col and val_col:
            # LONG format.
            for row in reader:
                m = (row.get(name_col) or "").strip()
                v = _to_float(row.get(val_col))
                if not m or v is None:
                    continue
                acc[m] = acc.get(m, 0.0) + v
                cnt[m] = cnt.get(m, 0) + 1
        else:
            # WIDE format: every column whose cells parse as numbers is a metric.
            for row in reader:
                for col, cell in row.items():
                    if not col:
                        continue
                    v = _to_float(cell)
                    if v is None:
                        continue
                    acc[col] = acc.get(col, 0.0) + v
                    cnt[col] = cnt.get(col, 0) + 1
    return {m: acc[m] / cnt[m] for m in acc}


def pick(metrics, keys):
    for k in keys:
        for m in metrics:
            if k.lower() in m.lower():
                return m, metrics[m]
    return None, None


def top_stalls(metrics, n=2):
    stalls = [(m, v) for m, v in metrics.items() if STALL_RE.search(m)]
    stalls.sort(key=lambda kv: kv[1], reverse=True)
    return stalls[:n]


def load_ncu(outdir, prefix="ncu_kbest"):
    """Return {K: {metric: value}} for every <prefix>_K*.csv found.

    prefix defaults to the Week-7 K2 files (ncu_kbest_K*.csv); pass
    prefix="ncu_batched" to pick up the Week-8 batched profiles instead.
    """
    out = {}
    for path in glob.glob(os.path.join(outdir, f"{prefix}_K*.csv")):
        m = re.search(re.escape(prefix) + r"_K(\d+)\.csv$", os.path.basename(path))
        if not m:
            continue
        parsed = parse_ncu_csv(path)
        if parsed:
            out[int(m.group(1))] = parsed
    return out


# ---- plots ---------------------------------------------------------------
def plot_recall(rows, path):
    xs = [r["K"] for r in rows if r["recall"] is not None]
    ys = [r["recall"] for r in rows if r["recall"] is not None]
    if not xs:
        return False
    plt.figure(figsize=(5, 3.2))
    plt.plot(xs, ys, "o-", color="#2b6cb0")
    plt.xlabel("K (beam width)")
    plt.ylabel("Top-K recall vs gold")
    plt.title("Top-K recall vs K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return True


def plot_throughput(rows, path):
    xs = [r["K"] for r in rows if r["sent_per_sec"] is not None]
    ys = [r["sent_per_sec"] for r in rows if r["sent_per_sec"] is not None]
    if not xs:
        return False
    plt.figure(figsize=(5, 3.2))
    plt.plot(xs, ys, "s-", color="#c05621")
    plt.xlabel("K (beam width)")
    plt.ylabel("Sentences / sec (kernel-only)")
    plt.title("Throughput vs K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return True


# ---- markdown ------------------------------------------------------------
def fmt(v, spec="{:.3f}", na="—"):
    return na if v is None else spec.format(v)


def build_sweep_md(sweeps):
    """Batch-size comparison tables. `sweeps` is [(label, rows), ...], one entry
    per --batch value benchmarked. Memory and throughput are both per-K, so each
    is a batch x K matrix. chunks/launches are constant across K within a run, so
    they collapse into single columns."""
    if not sweeps:
        return []
    Ks = sorted({r["K"] for _, rows in sweeps for r in rows})
    n_sent = next((r["n_sentences"] for _, rows in sweeps for r in rows
                   if r["n_sentences"]), None)

    def matrix(heading, note, field, spec):
        L = [f"### {heading}\n", note, ""]
        L.append("| batch | chunks | launches | "
                 + " | ".join(f"K={K}" for K in Ks) + " |")
        L.append("|---|-------:|---------:|"
                 + "".join("------:|" for _ in Ks))
        for label, rows in sweeps:
            by_K = {r["K"]: r for r in rows}
            first = rows[0] if rows else {}
            cells = [fmt(by_K[K][field], spec) if K in by_K else "—" for K in Ks]
            L.append(f"| {label} | {fmt(first.get('n_chunks'), '{:d}')} | "
                     f"{fmt(first.get('n_launches'), '{:d}')} | "
                     + " | ".join(cells) + " |")
        L.append("")
        return L

    L = ["## Batch-size sweep (memory vs speed)\n"]
    L.append("`--batch N` sizes each chunk's k-best table to that chunk's node span "
             "instead of the whole corpus, so peak device memory scales with N. "
             "Smaller chunks cost more launches (each chunk repeats the level loop) "
             "and lose cross-sentence parallelism per launch"
             + (f". Throughput is over all {n_sent} sentences in every row." if n_sent else "."))
    L.append("")
    L += matrix("Peak device memory (used MB)",
                "Whole-corpus (`--batch -1`) allocates the full table; this is the "
                "column the K2 driver has no answer to.",
                "used_MB", "{:.0f}")
    L += matrix("Throughput (sent/sec, kernel)",
                "Same total work in every row — only the chunking differs.",
                "sent_per_sec", "{:.0f}")
    return L


def build_md(rows, ncu, have_recall_png, have_thru_png,
             title="cuSHR K-best (K2) Benchmark — Week 7 (CP-4)",
             recall_png="recall_vs_k.png", thru_png="throughput_vs_k.png",
             body_only=False, recall_note="", ncu_prefix="ncu_kbest",
             sweeps=None):
    # body_only=True skips the H1 title + intro paragraph, so the tables can be
    # INJECTED into an existing narrative file (e.g. BATCHED_BENCHMARK.md)
    # between markers without duplicating its heading.
    L = []
    if not body_only:
        L.append(f"# {title}\n")
        L.append("Warp-level k-best merge kernel (`kbest_merge_level`) benchmarked over "
                 "the SIGHUM dataset. Kernel-only timing uses CUDA events around each "
                 "per-level merge launch (no H2D / reconstruction overhead). Correctness "
                 "is checked against the Week-3 CPU `TopKDecoder` at the same K.\n")

    # correctness line. NB: n_sentences is the corpus size / throughput
    # denominator, NOT the verified count -- reporting it here overstated the
    # check by ~120x. The verified count is n_check (the driver's --check).
    total_mm = sum((r["score_mismatch"] or 0) for r in rows)
    n_sent  = next((r["n_sentences"] for r in rows if r["n_sentences"]), None)
    n_check = next((r["n_check"] for r in rows if r["n_check"] is not None), None)
    status = "SCORE-EQUIVALENT to CPU at every K" if total_mm == 0 else \
             f"**{total_mm} score mismatches** — investigate before CP-4 sign-off"
    if n_check is not None:
        scope = f"spot-checked on the first {n_check} of {n_sent} sentences per K" \
                if n_sent else f"spot-checked on {n_check} sentences per K"
    else:
        scope = "checked sentence count not recorded in this CSV"
    L.append(f"**Correctness:** {status} ({scope}).\n")

    # throughput + memory table
    L.append("## Throughput and memory vs K\n")
    L.append("| K | sent/sec (kernel) | µs/sent (kernel) | µs/sent (loop) | "
             "table MB | used MB |")
    L.append("|---|------------------:|-----------------:|---------------:|"
             "---------:|--------:|")
    for r in rows:
        L.append("| {K} | {sps} | {usk} | {usl} | {tmb} | {umb} |".format(
            K=r["K"],
            sps=fmt(r["sent_per_sec"], "{:.0f}"),
            usk=fmt(r["us_kernel"], "{:.1f}"),
            usl=fmt(r["us_loop"], "{:.1f}"),
            tmb=fmt(r["table_MB"], "{:.1f}"),
            umb=fmt(r["used_MB"], "{:.1f}"),
        ))
    L.append("")

    # batched launch table: the K3 result in one number. Skipped for the K2 CSV,
    # which has no n_launches column.
    if any(r["n_launches"] is not None for r in rows):
        n_sent_l = next((r["n_sentences"] for r in rows if r["n_sentences"]), None)
        L.append("## Launch count (batched sweep)\n")
        L.append("One `kbest_merge_level` launch per topo level per chunk, covering that "
                 "level's nodes across *all* sentences in the chunk. Launches therefore "
                 "scale with depth and chunk count, not with sentence count"
                 + (f" ({n_sent_l} sentences)." if n_sent_l else "."))
        L.append("")
        L.append("| K | chunks | launches | sentences / launch |")
        L.append("|---|-------:|---------:|-------------------:|")
        for r in rows:
            spl = (n_sent_l / r["n_launches"]
                   if n_sent_l and r["n_launches"] else None)
            L.append("| {K} | {ch} | {la} | {spl} |".format(
                K=r["K"],
                ch=fmt(r["n_chunks"], "{:d}"),
                la=fmt(r["n_launches"], "{:d}"),
                spl=fmt(spl, "{:.0f}"),
            ))
        L.append("")

    L += build_sweep_md(sweeps or [])

    # recall table
    L.append("## Top-K recall vs gold\n")
    if any(r["recall"] is not None for r in rows):
        n_gold = next((r["n_gold"] for r in rows if r["n_gold"]), None)
        L.append(f"Measured over the {n_gold if n_gold else '?'} checked sentences that "
                 "carry a resolved gold path.\n")
        if recall_note:
            L.append(recall_note + "\n")
        L.append("| K | recall@K |")
        L.append("|---|---------:|")
        for r in rows:
            L.append(f"| {r['K']} | {fmt(r['recall'], '{:.4f}')} |")
    else:
        L.append("_No gold paths were available in the npz, so recall is N/A. "
                 "Re-run once gold paths are populated._")
    L.append("")

    # plots
    if have_recall_png:
        L.append(f"![Top-K recall vs K]({os.path.basename(recall_png)})\n")
    if have_thru_png:
        L.append(f"![Throughput vs K]({os.path.basename(thru_png)})\n")

    # nsight
    L.append("## Nsight Compute summary\n")
    if ncu:
        L.append("| K | occupancy % | DRAM throughput % | top warp-stall reasons |")
        L.append("|---|------------:|------------------:|------------------------|")
        for K in sorted(ncu):
            m = ncu[K]
            _, occ = pick(m, OCC_KEYS)
            _, dram = pick(m, DRAM_KEYS)
            stalls = top_stalls(m, 2)
            def stall_name(n):
                n = re.sub(r'.*stalled_', '', n)
                return n.replace('_per_issue_active.ratio', '').replace('_per_issue_active', '')
            stall_txt = "; ".join(f"{stall_name(name)} ({val:.2f})"
                                  for name, val in stalls) or "—"
            L.append(f"| {K} | {fmt(occ, '{:.1f}')} | {fmt(dram, '{:.1f}')} | {stall_txt} |")
        L.append("")
        L.append("_Occupancy = `sm__warps_active` % of peak; DRAM % = "
                 "`dram__throughput` % of peak sustained. Stall values are avg "
                 "warps stalled per active cycle for that reason._")
    else:
        prof_job = "profile_batched.slurm" if ncu_prefix == "ncu_batched" else "profile_kbest.slurm"
        L.append(f"_Profiling CSVs (`{ncu_prefix}_K*.csv`) not found. Run "
                 f"`{prof_job}`, copy the CSVs next to this script, and "
                 "re-run `make_benchmark_md.py`._ **TODO**")
    L.append("")
    return "\n".join(L)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.path.join(here, "kbest_bench.csv"))
    ap.add_argument("--outdir", default=here)
    # Output names are configurable so the batched (Week-8) run can write to
    # separate files and NOT overwrite the frozen Week-7 artifacts.
    ap.add_argument("--md", default="KBEST_BENCHMARK.md",
                    help="output markdown filename (relative to --outdir)")
    ap.add_argument("--title", default="cuSHR K-best (K2) Benchmark — Week 7 (CP-4)",
                    help="H1 title for the generated markdown")
    ap.add_argument("--recall-png", default="recall_vs_k.png",
                    help="recall plot filename (relative to --outdir)")
    ap.add_argument("--thru-png", default="throughput_vs_k.png",
                    help="throughput plot filename (relative to --outdir)")
    ap.add_argument("--ncu-prefix", default="ncu_kbest",
                    help="filename prefix for Nsight CSVs (<prefix>_K*.csv); "
                         "use ncu_batched for the Week-8 batched profiles")
    ap.add_argument("--recall-note", default="",
                    help="extra caption under the recall heading, e.g. to note "
                         "that batched recall is an invariant spot-check vs K2")
    ap.add_argument("--sweep", action="append", default=[], metavar="LABEL=CSV",
                    help="add a row to the batch-size sweep tables, e.g. "
                         "--sweep 256=batched_bench_b256.csv . Repeatable; rows "
                         "appear in the order given. Omit for no sweep section.")
    ap.add_argument("--inject", action="store_true",
                    help="inject the generated tables between the AUTO-GENERATED "
                         "markers in --md (preserving the rest of that file) "
                         "instead of overwriting the whole file")
    args = ap.parse_args()

    if not os.path.exists(args.bench):
        raise SystemExit(f"bench CSV not found: {args.bench}\n"
                         "Run bench_kbest.slurm (kbest_bench.csv) or "
                         "bench_batched.slurm (batched_bench.csv) first.")

    rows = load_bench(args.bench)
    ncu = load_ncu(args.outdir, prefix=args.ncu_prefix)

    sweeps = []
    for spec in args.sweep:
        if "=" not in spec:
            raise SystemExit(f"--sweep needs LABEL=CSV, got: {spec}")
        label, path = spec.split("=", 1)
        if not os.path.isabs(path):
            path = os.path.join(args.outdir, path)
        if not os.path.exists(path):
            raise SystemExit(f"--sweep CSV not found: {path}")
        sweeps.append((label, load_bench(path)))

    recall_png = os.path.join(args.outdir, args.recall_png)
    thru_png = os.path.join(args.outdir, args.thru_png)
    have_recall = plot_recall(rows, recall_png)
    have_thru = plot_throughput(rows, thru_png)

    md_path = os.path.join(args.outdir, args.md)
    BEGIN = "<!-- BEGIN AUTO-GENERATED RESULTS (make_benchmark_md.py --inject) -->"
    END   = "<!-- END AUTO-GENERATED RESULTS -->"

    if args.inject:
        if not os.path.exists(md_path):
            raise SystemExit(f"--inject needs an existing {md_path} with the "
                             f"markers:\n  {BEGIN}\n  {END}")
        text = open(md_path, encoding="utf-8").read()
        bi, ei = text.find(BEGIN), text.find(END)
        if bi == -1 or ei == -1 or ei < bi:
            raise SystemExit(f"markers not found in {md_path}. Add:\n"
                             f"  {BEGIN}\n  ...\n  {END}")
        body = build_md(rows, ncu, have_recall, have_thru, title=args.title,
                        recall_png=recall_png, thru_png=thru_png, body_only=True,
                        recall_note=args.recall_note, ncu_prefix=args.ncu_prefix,
                        sweeps=sweeps)
        after_begin = text.index("\n", bi) + 1        # keep the BEGIN marker line
        new_text = text[:after_begin] + "\n" + body + "\n" + text[ei:]
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(new_text)
        print(f"injected results into {md_path} (between markers)")
    else:
        md = build_md(rows, ncu, have_recall, have_thru,
                      title=args.title, recall_png=recall_png, thru_png=thru_png,
                      recall_note=args.recall_note, ncu_prefix=args.ncu_prefix,
                      sweeps=sweeps)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"wrote {md_path}")

    if have_recall:
        print(f"wrote {recall_png}")
    if have_thru:
        print(f"wrote {thru_png}")
    if not ncu:
        print(f"note: no {args.ncu_prefix}_K*.csv found; Nsight section left as TODO")


if __name__ == "__main__":
    main()
