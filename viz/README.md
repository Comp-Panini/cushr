# cuSHR DAG visualizer

A tiny web app to explore the Sanskrit Heritage Reader segmentation lattices.
Type a sentence number and see the full DAG of candidate word-form sequences,
with an optional **gold-path highlight**. It ships a self-contained demo
database, so whoever opens the link needs nothing on their machine.

## What you get
- Look up a sentence by **npz sentence index** (the same index the CPU decoder
  and `golden_outputs.json` use), so the visualizer lines up with the decoder.
- Every candidate word rendered as a node (hover for lemma / morph / cng / chunk).
- Toggle "highlight gold path" to fade everything except the gold segmentation.

## Build the demo database (one-time, needs the corpus locally)
The deployed app only reads `cushr_viz.db`; building it needs the raw corpus and
the `sentence_index.json` emitted by `ingest/ingest.py`.

```bash
cd ingest && python ingest.py          # emits ingest/sentence_index.json
cd ../viz
pip install -r requirements.txt
python build_db.py --n 2000            # writes viz/cushr_viz.db (a few MB)
```

## Run locally
```bash
uvicorn app:app --reload
# open http://127.0.0.1:8000
```

## Deploy a shareable link (no data on the client)
The repo is too large for GitHub, but `cushr_viz.db` is only a few MB and ships
inside the deploy. Recommended: **Hugging Face Spaces (Docker SDK)**.

1. Create a new Space → SDK: **Docker**.
2. Upload the contents of this `viz/` folder **including** `cushr_viz.db`
   (`app.py`, `Dockerfile`, `requirements.txt`, `static/`, `cushr_viz.db`).
3. The Space builds and serves on port 7860; share the Space URL.

Render / Fly.io free tiers work the same way using the included `Dockerfile`.

## API
- `GET /api/index` — available indices + counts.
- `GET /api/sentence/{idx}` — `{nodes, edges, gold_path}` for one sentence.
