"""
cuSHR DAG visualizer — tiny FastAPI app.

Serves a self-contained demo database (cushr_viz.db) so an advisor can open one
URL, type a sentence index, and explore the full segmentation DAG with an
optional gold-path highlight. No corpus needed on the client.

Run locally:
    pip install -r requirements.txt
    python build_db.py            # produces cushr_viz.db
    uvicorn app:app --reload
"""
import os
import json
import zlib
import sqlite3

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(__file__)
DB_PATH = os.path.join(HERE, 'cushr_viz.db')

app = FastAPI(title='cuSHR DAG visualizer')


def _connect():
    if not os.path.exists(DB_PATH):
        raise HTTPException(503, 'database not built — run build_db.py')
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


@app.get('/api/index')
def index():
    con = _connect()
    rows = con.execute('SELECT idx, has_gold, n_nodes FROM sentences ORDER BY idx').fetchall()
    con.close()
    idxs = [r['idx'] for r in rows]
    return {
        'count': len(idxs),
        'min': idxs[0] if idxs else None,
        'max': idxs[-1] if idxs else None,
        'with_gold': sum(r['has_gold'] for r in rows),
        'available': idxs,
    }


@app.get('/api/sentence/{idx}')
def sentence(idx: int):
    con = _connect()
    row = con.execute('SELECT data FROM sentences WHERE idx=?', (idx,)).fetchone()
    con.close()
    if row is None:
        raise HTTPException(404, f'sentence {idx} not found')
    blob = row['data']
    # `data` is zlib-compressed JSON; older uncompressed DBs stored plain text.
    if isinstance(blob, (bytes, bytearray)):
        text = zlib.decompress(blob).decode('utf-8')
    else:
        text = blob
    return JSONResponse(json.loads(text))


@app.get('/api/lookup/{stem}')
def lookup(stem: str):
    """Resolve an original graphml stem / DCS sent_id to its npz index."""
    con = _connect()
    row = con.execute('SELECT idx FROM sentences WHERE stem=?', (stem,)).fetchone()
    con.close()
    if row is None:
        raise HTTPException(404, f'sent_id {stem} not found')
    return {'idx': row['idx'], 'stem': stem}


@app.get('/')
def root():
    return FileResponse(os.path.join(HERE, 'static', 'index.html'))


app.mount('/static', StaticFiles(directory=os.path.join(HERE, 'static')), name='static')
