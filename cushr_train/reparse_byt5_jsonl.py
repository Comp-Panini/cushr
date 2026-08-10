#!/usr/bin/env python3
"""Re-derive words/lemmas/tags from the `raw` field already in a preds JSONL.

`byt5_to_jsonl.py` keeps the model's verbatim output in `raw` precisely so a
parser fix does not require re-running a 582M model on a GPU. This applies the
current `parse_analyzed` to existing files in place.

Rewrites only the parsed fields; `id`, `mode` and `raw` are untouched, so the
link back to the model run is preserved.
"""
import argparse
import json
import shutil

from byt5_to_jsonl import parse_analyzed
from translit import iast_to_slp1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--keep-iast", action="store_true")
    args = ap.parse_args()

    for path in args.files:
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        shutil.copy(path, path + ".bak")
        changed = 0
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                w, l, t = parse_analyzed(r["raw"], r["mode"])
                if not args.keep_iast:
                    w = [iast_to_slp1(x) for x in w]
                    l = [iast_to_slp1(x) for x in l]
                if (w, l, t) != (r["words"], r["lemmas"], r["tags"]):
                    changed += 1
                r["words"], r["lemmas"], r["tags"] = w, l, t
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path}: {len(recs):,} records, {changed:,} changed "
              f"(backup at {path}.bak)")


if __name__ == "__main__":
    main()
