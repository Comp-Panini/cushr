"""Build the paper's results table and its three plots from cached artifacts.

    python make_results_matrix.py --manifest results_manifest.json

Design
------
GPU and SLURM runs stay manual on Lonestar6 and leave .npz / .csv behind; this
script only ingests them. That keeps the table reproducible offline, without an
allocation, which a script that launched its own jobs would not be.

For each (dataset, system) cell the manifest names an artifact or null. Cells
with work to do are evaluated by shelling out to eval_slm.py / eval_surface.py
with --json-out -- the same code path that produced every published number, so
the table cannot drift from the scripts. Results are cached on a digest of the
command plus the size+mtime of every input file, so a re-run costs seconds.

A null cell prints an em-dash and is footnoted. Nothing is ever interpolated,
averaged across datasets, or carried over from another cell: a missing number
is information about the evaluation, and hiding it would be the one failure
mode this table exists to prevent.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

CACHE_DIR = "results/.cache"
# The K grid the paper reports. eval_slm.py emits every power of two it can
# reach plus 5; we select from that rather than make it recompute.
REPORT_KS = [1, 5, 16, 32, 64]
SLM_LEVELS = ["S", "L", "S+M", "L+M", "S+L+M"]
# Systems whose numbers live in the `pred` column of a cuSHR run rather than
# the `cushr` column. Reading the wrong one would report cuSHR's score under
# another system's name.
EXTERNAL = {"byt5"}


# ---------------------------------------------------------------- utilities
def _sig(paths, extra):
    """Digest of a command plus the identity of its inputs."""
    h = hashlib.sha256()
    h.update(json.dumps(extra, sort_keys=True).encode())
    for p in sorted(paths):
        h.update(p.encode())
        if os.path.exists(p):
            st = os.stat(p)
            h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
        else:
            h.update(b"MISSING")
    return h.hexdigest()[:16]


def run_eval(script, args, inputs, force=False):
    """Run one eval script with --json-out, cached on its inputs."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _sig(inputs, [script] + args)
    out = os.path.join(CACHE_DIR, f"{script.split('.')[0]}_{key}.json")
    log = out[:-5] + ".log"
    if os.path.exists(out) and not force:
        print(f"  cached  {os.path.basename(out)}")
        return json.load(open(out))
    cmd = [sys.executable, script] + args + ["--json-out", out]
    print(f"  running {' '.join(cmd[1:])[:110]}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           cwd=os.path.dirname(os.path.abspath(__file__)),
                           env=env)
    if r.returncode != 0 or not os.path.exists(out):
        # Surface the log tail. A silent null here would become an em-dash and
        # read as "not attempted" rather than "attempted and broken".
        tail = open(log, encoding="utf-8").read()[-800:]
        raise SystemExit(f"{script} failed (exit {r.returncode}):\n{tail}")
    return json.load(open(out))


def read_bench(path):
    """One bench CSV -> list of dicts.

    Three schemas exist in cushr_gpu/ (11, 14 and 18 columns); DictReader keys
    on the header so a column's position never matters.
    """
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            # The CSVs are written by the CUDA driver with CRLF endings, so the
            # last field of every row carries a trailing \r.
            v = v.strip() if isinstance(v, str) else v
            # "NA" is what the driver writes when it ran with --check 0 and so
            # never computed recall. It must stay None, not become 0.0.
            if v in (None, "", "NA"):
                r[k] = None
                continue
            try:
                r[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except (ValueError, TypeError):
                r[k] = v
    return rows


def fmt(v, nd=2):
    return "&mdash;" if v is None else f"{v:.{nd}f}"


# ------------------------------------------------------------------ collect
def collect(man, force=False):
    common, cells = man["common"], man["cells"]
    out, prov = {}, {}
    for key, spec in cells.items():
        ds_id, sys_id = key.split("/")
        ds = man["datasets"][ds_id]
        print(f"[{key}]")
        if spec is None:
            print("  not produced -- em-dash")
            out[key] = None
            prov[key] = []
            continue
        rec, src = {}, []

        if "published" in spec:
            rec["published"] = spec["published"]
            src.append(spec["published"].get("source", "published"))

        base = ["--cache", common["cache"], "--model", common["model"],
                "--index", common["index"], "--raw", common["raw"],
                "--tsv", ds["tsv"]]
        inputs = [common["model"], common["index"], common["raw"], ds["tsv"]]

        if "slm" in spec:
            s = spec["slm"]
            a = list(base)
            if ds.get("no_surface"):
                a.append("--no-surface")
            if s.get("cands"):
                a += ["--cands", s["cands"]]
                inputs.append(s["cands"])
            if s.get("kbest"):
                a += ["--kbest", str(s["kbest"])]
            if s.get("rerank"):
                a += ["--rerank", s["rerank"]]
                inputs.append(s["rerank"])
            if s.get("pred_jsonl"):
                a += ["--pred-jsonl", s["pred_jsonl"],
                      "--pred-name", s.get("pred_name", "pred")]
                inputs.append(s["pred_jsonl"])
            rec["slm"] = run_eval("eval_slm.py", a, inputs, force)
            src.append("eval_slm.py")

        if "surface" in spec:
            rec["surface"] = run_eval("eval_surface.py", base, inputs, force)
            src.append("eval_surface.py")

        out[key] = rec
        prov[key] = src

    # Resolved after the loop so the referenced cell is guaranteed present.
    # This is an asserted identity, not a measurement: see the "why" block in
    # the manifest. It borrows accuracy only -- throughput and memory still
    # come from the borrowing system's own bench CSV.
    for key, spec in cells.items():
        if not spec or "same_accuracy_as" not in spec:
            continue
        ds_id, _ = key.split("/")
        ref = f"{ds_id}/{spec['same_accuracy_as']}"
        if not out.get(ref):
            raise SystemExit(
                f"{key}: same_accuracy_as points at {ref}, which produced "
                "nothing. Fix the manifest rather than let the cell silently "
                "fall back to an em-dash.")
        out[key] = dict(out[ref])
        out[key]["_borrowed_from"] = ref
        prov[key] = [f"accuracy asserted identical to `{ref}` "
                     "(score_mismatch = 0); throughput measured separately"]
        print(f"[{key}]\n  accuracy borrowed from {ref}")
    return out, prov


def cell_levels(rec, sys_id):
    """The S/L/M ladder for one cell as {level: value or None}."""
    if not rec or "slm" not in rec:
        return {}
    lv = rec["slm"]["levels"]
    field = "pred" if sys_id in EXTERNAL else "cushr"
    return {k: (lv[k][field] if k in lv else None) for k in SLM_LEVELS}


# ------------------------------------------------------------------- render
def render(man, res, prov, bench, path):
    L = []
    A = L.append
    A("# cuSHR &mdash; full evaluation matrix\n")
    A("Generated by `make_results_matrix.py`. Every number traces to a named "
      "artifact; see **Provenance** at the end. An em-dash means the cell was "
      "not produced &mdash; no value in this file is inferred, interpolated, "
      "or carried over from another cell.\n")

    for ds_id, ds in man["datasets"].items():
        A(f"\n## {ds['label']}  (n = {ds['n']:,})\n")
        if ds.get("note"):
            A(f"> {ds['note']}\n")

        A("\n### Word-level segmentation\n")
        A("| System | P | R | F1 | Perfect match |")
        A("|---|---:|---:|---:|---:|")
        for sys_id, sy in man["systems"].items():
            rec = res.get(f"{ds_id}/{sys_id}")
            p = r = f = pm = None
            if rec and "surface" in rec:
                d = rec["surface"]
                p, r, f, pm = (d["p_macro"], d["r_macro"], d["f1_macro"],
                               d["pm"])
            elif rec and "published" in rec:
                d = rec["published"]
                p, r, f, pm = (d.get("p_macro"), d.get("r_macro"),
                               d.get("f1_macro"), d.get("pm"))
            elif rec and "slm" in rec:
                # The S level IS surface perfect match -- eval_surface.py and
                # eval_slm.py agree to the digit on SIGHUM-test. Report PM
                # alone; never a P/R/F1 that was not measured.
                pm = cell_levels(rec, sys_id).get("S")
            A(f"| {sy['label']} | {fmt(p)} | {fmt(r)} | {fmt(f)} | "
              f"{fmt(pm)} |")

        A("\n### Sentence-level perfect match by annotation level\n")
        A("| System | " + " | ".join(SLM_LEVELS) + " |")
        A("|---" * (len(SLM_LEVELS) + 1) + "|")
        for sys_id, sy in man["systems"].items():
            rec = res.get(f"{ds_id}/{sys_id}")
            lv = cell_levels(rec, sys_id)
            A(f"| {sy['label']} | "
              + " | ".join(fmt(lv.get(k)) for k in SLM_LEVELS) + " |")
        for sys_id in ("cushr_gpu_rerank", "cushr_gpu_top1"):
            rec = res.get(f"{ds_id}/{sys_id}")
            if rec and "slm" in rec:
                lv = rec["slm"]["levels"]
                A("| *ORACLE (ceiling)* | "
                  + " | ".join(fmt(lv[k]["oracle"]) if k in lv else "&mdash;"
                               for k in SLM_LEVELS) + " |")
                break

        # Only the beam, deliberately. recall@K is a property of the candidate
        # list, so the reranker cell reports numbers identical to the top-1
        # cell -- listing both would suggest the reranker improved recall, and
        # it cannot: it only reorders what the beam already contains. Take the
        # widest beam available, since that is the one that reaches K=64.
        rows, kb_used = [], None
        cand = [(res.get(f"{ds_id}/{s}"), s)
                for s in ("cushr_gpu_top1", "cushr_gpu_rerank")]
        cand = [(r, s) for r, s in cand
                if r and "slm" in r and r["slm"].get("recall")]
        if cand:
            rec, _ = max(cand, key=lambda t: t[0]["slm"].get("kbest") or 0)
            rc, kb_used = rec["slm"]["recall"], rec["slm"].get("kbest")
            for lvl in ("S", "L", "S+M"):
                if lvl in rc:
                    rows.append((f"beam [{lvl}]", rc[lvl]))
        if rows:
            A("\n### Top-K recall (cuSHR only &mdash; ByT5 has no beam)\n")
            A(f"Beam width K={kb_used}. This is the beam's own recall, so it "
              "is a hard ceiling on any reranker: the reranker reorders these "
              "candidates and cannot add one. Compare it against the top-1 "
              "row of the ladder above to see how much headroom reranking "
              "has.\n")
            A("| Beam | " + " | ".join(f"@{k}" for k in REPORT_KS) + " |")
            A("|---" * (len(REPORT_KS) + 1) + "|")
            for name, rc in rows:
                A(f"| {name} | "
                  + " | ".join(fmt(rc.get(str(k))) for k in REPORT_KS) + " |")

    A("\n## Throughput and memory (Lonestar6 A100)\n")
    if bench["k_sweep"]:
        A(f"`{man['bench'].get('k_sweep_label', '')}`\n")
        A("| K | recall@K | sentences/sec | GPU MB | us/sent K4 | us/sent K3 |")
        A("|---:|---:|---:|---:|---:|---:|")
        for r in bench["k_sweep"]:
            rc = r.get("recall_at_K")
            A(f"| {r['K']} | {fmt(100 * rc if rc is not None else None)} | "
              f"{r['sent_per_sec_kernel']:,.0f} | {r['gpu_used_MB']:,.0f} | "
              f"{fmt(r.get('us_per_sent_k4'), 3)} | "
              f"{fmt(r.get('us_per_sent_k3'), 3)} |")
        A("\nThe recall column is empty across this sweep: it was run with "
          "`--check 0`, which skips the CPU cross-check that computes recall. "
          "The checked runs are tabulated separately below. "
          "`batched_bench.csv` does carry a full recall curve, "
          "but that run used the hand-tuned `log_linear` scorer (recall@1 = "
          "8.31%) and must never be quoted alongside the trained model.")

    if bench["checked"]:
        A("\n### Whole-corpus recall, verified against the CPU decoder\n")
        A("| K | recall@K | n_gold | sentences/sec | GPU MB | source |")
        A("|---:|---:|---:|---:|---:|---|")
        for r in bench["checked"]:
            rc = r.get("recall_at_K")
            A(f"| {r['K']} | {fmt(100 * rc if rc is not None else None, 4)} | "
              f"{r['n_gold']:,} | {r['sent_per_sec_kernel']:,.0f} | "
              f"{r['gpu_used_MB']:,.0f} | `{r['_src']}` |")
        A("\nThese ran with `--check -1`, decoding every sentence on the CPU "
          "as well and comparing; `score_mismatch` and `count_mismatch` are 0 "
          "in both. The denominator is `n_gold`, not `n_sentences`: 4,056 of "
          "the 119,503 corpus sentences have an empty gold span and can never "
          "be hit by any decoder, so including them would understate recall.")
        # Computed, not transcribed: a hardcoded pair would silently go stale
        # the first time either CSV is regenerated.
        sweep_at = {r["K"]: r["sent_per_sec_kernel"] for r in bench["k_sweep"]}
        cmp = [(r["K"], r["sent_per_sec_kernel"], sweep_at[r["K"]])
               for r in bench["checked"] if r["K"] in sweep_at]
        if cmp:
            A("\nThese throughputs are **lower** than the same K in the sweep "
              "above ("
              + "; ".join(f"K={k}: {c:,.0f} vs {s:,.0f}" for k, c, s in cmp)
              + " sent/sec). Quote the sweep rows as the throughput result: "
                "these runs also paid for the path dump and ran alongside the "
                "full CPU cross-check.")

    if bench["cpu"]:
        c = bench["cpu"][0]
        A(f"\n**cushr_cpu, measured on a Lonestar6 compute node:** "
          f"{c['sent_per_sec']:,.0f} sentences/sec wall clock "
          f"(K={c['K']:.0f}, {c['n_sentences']:,} sentences in "
          f"{c['wall_sec']:.1f} s, `{c['scorer']}`). This supersedes the "
          "\"~100 sentences/sec single-threaded\" figure in "
          "`cushr_cpu/README.md:55`, which was a design target written before "
          "the decoder existed and is low by a factor of ~11. Any speedup "
          "claim must use this number. Note it is end-to-end wall clock while "
          "the GPU column is kernel-only, so the ratio of the two is an upper "
          "bound on the achievable end-to-end speedup, not a measurement of "
          "one.")
    else:
        A("\n**cushr_cpu throughput: not measured.** `cushr_evaluate` prints "
          "its `sent/s` line to stdout and writes it nowhere; the "
          "\"~100 sentences/sec\" in `cushr_cpu/README.md:55` is a design "
          "target, never a measurement. It has to be run on a Lonestar6 "
          "compute node before this row can be filled -- a laptop number is "
          "not \"the same hardware\".")

    A("\n## Provenance\n")
    A("| Cell | Produced by |")
    A("|---|---|")
    for k in man["cells"]:
        A(f"| `{k}` | {', '.join(prov.get(k, [])) or '&mdash; not produced'} |")

    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print(f"wrote {path}")


# -------------------------------------------------------------------- plots
def plots(man, res, bench, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping plots")
        return

    # 1. accuracy vs throughput
    pts = []
    for ds_id in man["datasets"]:
        for sys_id, sy in man["systems"].items():
            rec = res.get(f"{ds_id}/{sys_id}")
            if not rec:
                continue
            f1 = None
            if "surface" in rec:
                f1 = rec["surface"]["f1_macro"]
            elif "published" in rec:
                f1 = rec["published"].get("f1_macro")
            tp = bench["throughput"].get(sys_id)
            if f1 is not None and tp:
                pts.append((tp, f1, sy["label"]))
    if pts:
        plt.figure(figsize=(6.5, 3.8))
        xs = [p[0] for p in pts]
        # Labels sit beside their point, so the rightmost one needs room or it
        # runs off the canvas. Flip the anchor for points in the right half.
        mid = (min(xs) * max(xs)) ** 0.5
        for x, y, lab in pts:
            plt.scatter(x, y, s=60, zorder=3)
            right = x > mid
            plt.annotate(lab, (x, y), textcoords="offset points",
                         xytext=(-8 if right else 8, 6), fontsize=8,
                         ha="right" if right else "left")
        plt.xscale("log")
        plt.xlim(min(xs) / 4, max(xs) * 4)
        plt.xlabel("Sentences / sec (log scale)")
        plt.ylabel("Word-level F1 (macro)")
        plt.title("Accuracy vs throughput")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/pareto_accuracy_throughput.png", dpi=140)
        plt.close()
        print(f"wrote {outdir}/pareto_accuracy_throughput.png")
    else:
        print("skipped pareto plot: no cell has both an F1 and a throughput")

    # 2. recall vs K.
    #
    # Deliberately NOT from the bench CSVs. The four k4 sweeps ran with
    # --check 0 and wrote recall_at_K = NA; the one sweep that does carry a
    # full curve, batched_bench.csv, is the hand-tuned log_linear scorer
    # (recall@1 = 8.31%) and plotting it here would attribute the wrong
    # decoder's numbers to the trained model. eval_slm.py --kbest is the only
    # source of a trained-model recall curve.
    # Widest beam per dataset, not the first cell that happens to have a curve:
    # taking the first one truncated the SIGHUM line at K=32 (the reranker
    # cell) while a K=64 curve sat unused in the top-1 cell. recall@K is a
    # property of the beam, so the two cells agree wherever they overlap.
    found = []
    for ds_id, ds in man["datasets"].items():
        cand = [res.get(f"{ds_id}/{s}")
                for s in ("cushr_gpu_top1", "cushr_gpu_rerank")]
        cand = [r for r in cand
                if r and "slm" in r and r["slm"].get("recall")]
        if cand:
            rec = max(cand, key=lambda r: r["slm"].get("kbest") or 0)
            found.append((ds["label"].split("(")[0].strip(),
                          rec["slm"]["recall"]))
    # One level for every curve. A plot mixing S on one dataset with L on
    # another would show a gap that is mostly the level, not the domain, and a
    # reader would take it for the domain. S is preferred; L is the fallback
    # because it is the only level a --no-surface dataset can report.
    lvl = next((c for c in ("S", "L") if all(c in rc for _, rc in found)),
               None)
    curves = [(f"{lab} [{lvl}]",
               sorted((int(k), v) for k, v in rc[lvl].items()))
              for lab, rc in found] if lvl else []
    if curves:
        plt.figure(figsize=(5.4, 3.4))
        for lab, pts in curves:
            plt.plot([k for k, _ in pts], [v for _, v in pts], "o-", label=lab)
        plt.xscale("log", base=2)
        ticks = sorted({k for _, pts in curves for k, _ in pts})
        plt.xticks(ticks, [str(t) for t in ticks])
        plt.xlabel("K (beam width)")
        plt.ylabel("Top-K recall, sentence-level (%)")
        plt.title("Top-K recall vs K (trained biaffine)")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{outdir}/recall_vs_k.png", dpi=140)
        plt.close()
        print(f"wrote {outdir}/recall_vs_k.png")
    else:
        print("skipped recall plot: no cell was run with --kbest")

    # 3. memory vs K and vs batch size -- both sweeps already exist
    if bench["k_sweep"] or bench["batch"]:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        if bench["k_sweep"]:
            ks = [r["K"] for r in bench["k_sweep"]]
            ax[0].plot(ks, [r["gpu_used_MB"] for r in bench["k_sweep"]],
                       "s-", color="#c05621")
            ax[0].set_xscale("log", base=2)
            # The sweep includes K=24 and K=48, so power-of-two tick labels
            # would name values that were never run.
            ax[0].set_xticks(ks)
            ax[0].set_xticklabels([str(k) for k in ks], fontsize=7)
            ax[0].minorticks_off()
            ax[0].set_xlabel("K (beam width)")
            ax[0].set_ylabel("GPU MB in use")
            ax[0].set_title("Memory vs K")
            ax[0].grid(True, alpha=0.3)
        for b, rows in sorted(bench["batch"].items()):
            ax[1].plot([r["K"] for r in rows],
                       [r["gpu_used_MB"] for r in rows],
                       "o-", ms=3, label=f"batch {b:,}")
        if bench["batch"]:
            bks = sorted({r["K"] for rows in bench["batch"].values()
                          for r in rows})
            ax[1].set_xscale("log", base=2)
            ax[1].set_xticks(bks)
            ax[1].set_xticklabels([str(k) for k in bks], fontsize=7)
            ax[1].minorticks_off()
            ax[1].set_xlabel("K (beam width)")
            ax[1].set_ylabel("GPU MB in use")
            ax[1].set_title("Memory vs K, by batch size")
            ax[1].grid(True, alpha=0.3)
            ax[1].legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(f"{outdir}/memory_vs_k_batch.png", dpi=140)
        plt.close()
        print(f"wrote {outdir}/memory_vs_k_batch.png")


def load_bench(man):
    b = man["bench"]
    k_sweep = read_bench(b.get("k_sweep"))
    batch = {}
    for p in sorted(glob.glob(b.get("batch_sweep") or "")):
        rows = read_bench(p)
        # Anchor on the trailing _b<N>.csv: a plain split("_b") would fire on
        # the "_b" inside "batched_bench_" first.
        m = re.search(r"_b(\d+)\.csv$", os.path.basename(p))
        if rows and m:
            batch[int(m.group(1))] = rows
    cpu = read_bench(b.get("cpu_csv"))
    # The checked runs are separate one-row CSVs, not part of the K sweep: the
    # sweep ran --check 0 and so has no recall at any K. These are the only
    # whole-corpus recall figures that were verified against the CPU decoder.
    checked = []
    for key in ("checked_k32", "checked_k64"):
        for r in read_bench(b.get(key)):
            r["_src"] = os.path.basename(b[key])
            checked.append(r)
    checked.sort(key=lambda r: r["K"])
    thr = {}
    # K=32 is the headline beam; quote its throughput, not the fastest row.
    for r in k_sweep:
        if r["K"] == 32:
            thr["cushr_gpu_top1"] = r["sent_per_sec_kernel"]
            thr["cushr_gpu_rerank"] = r["sent_per_sec_kernel"]
    if cpu:
        # cushr_evaluate --csv writes `sent_per_sec` (wall clock), not the GPU
        # CSVs' `sent_per_sec_kernel`. The names differ on purpose: one is
        # end-to-end, the other excludes host time, and silently equating them
        # would overstate the GPU speedup.
        thr["cushr_cpu"] = cpu[0].get("sent_per_sec")
    return {"k_sweep": k_sweep, "batch": batch, "cpu": cpu,
            "checked": checked, "throughput": thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results_manifest.json")
    ap.add_argument("--out-md", default="RESULTS_MATRIX.md")
    ap.add_argument("--out-json", default="results/results_matrix.json")
    ap.add_argument("--plot-dir", default="results")
    ap.add_argument("--force", action="store_true",
                    help="ignore the cache and re-run every eval")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    os.makedirs(args.plot_dir, exist_ok=True)
    res, prov = collect(man, args.force)
    bench = load_bench(man)
    render(man, res, prov, bench, args.out_md)
    json.dump({"cells": res, "provenance": prov,
               "throughput": bench["throughput"]},
              open(args.out_json, "w"), indent=1, sort_keys=True)
    print(f"wrote {args.out_json}")
    plots(man, res, bench, args.plot_dir)


if __name__ == "__main__":
    main()
