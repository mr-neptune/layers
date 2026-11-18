"""Layers web server: upload a song, queue it, serve the results.

Run: .venv/bin/python -m layers.server  (listens on 127.0.0.1:8020 by default)
"""

from __future__ import annotations

import hmac
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("LAYERS_DATA", ROOT / "data"))
WEB = ROOT / "web"
MAX_UPLOAD = 100 * 1024 * 1024  # Cloudflare's request body limit on the free plan
# When set, /api requests must carry this value in X-Layers-Key (the website's
# Worker adds it), so the tunnel's own hostname can't be used directly.
KEY = os.environ.get("LAYERS_KEY", "")
AUDIO_EXT = {".mp3", ".m4a", ".aac", ".mp4", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aif", ".aiff", ".webm"}
ID_RE = re.compile(r"^[0-9a-f]{12}$")
FILE_RE = re.compile(r"^[a-z0-9-]+\.(mp3|bin|json)$")

DATA.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
jobs: queue.Queue[str] = queue.Queue()


def song_dir(sid: str) -> Path:
    return DATA / sid


def read_status(sid: str) -> dict:
    try:
        return json.loads((song_dir(sid) / "status.json").read_text())
    except (OSError, ValueError):
        return {}


def write_status(sid: str, **changes) -> dict:
    with lock:
        st = read_status(sid)
        st.update(changes)
        tmp = song_dir(sid) / "status.tmp"
        tmp.write_text(json.dumps(st))
        tmp.replace(song_dir(sid) / "status.json")
        return st


def summary(sid: str) -> dict:
    st = read_status(sid)
    out = {"id": sid, **st}
    res = song_dir(sid) / "out" / "result.json"
    if st.get("state") == "done" and res.exists():
        out["result"] = json.loads(res.read_text())
    return out


def worker() -> None:
    while True:
        sid = jobs.get()
        d = song_dir(sid)
        if not d.exists():
            continue
        st = read_status(sid)
        write_status(sid, state="processing", stage="Starting", progress=0.0, started=time.time())
        shutil.rmtree(d / "out", ignore_errors=True)
        cmd = [sys.executable, "-m", "layers.pipeline", str(d / st["input"]), str(d / "out"), "--json"]
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        error = None
        assert proc.stdout
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "error" in msg:
                error = msg["error"]
            elif "stage" in msg:
                write_status(sid, stage=msg["stage"], progress=msg["progress"])
        stderr = proc.stderr.read() if proc.stderr else ""
        code = proc.wait()
        if code == 0:
            write_status(sid, state="done", stage="Done", progress=1.0, finished=time.time())
        else:
            tail = stderr.strip().splitlines()[-1:] if stderr else []
            write_status(sid, state="failed", error=error or (tail[0] if tail else f"exit code {code}"))


def requeue_unfinished() -> None:
    pending = []
    for d in DATA.iterdir():
        if d.is_dir() and ID_RE.match(d.name):
            st = read_status(d.name)
            if st.get("state") in ("queued", "processing"):
                pending.append((st.get("created", 0), d.name))
    for _, sid in sorted(pending):
        write_status(sid, state="queued", stage="Waiting", progress=0.0)
        jobs.put(sid)


async def list_songs(request: Request) -> Response:
    songs = []
    for d in DATA.iterdir():
        if d.is_dir() and ID_RE.match(d.name):
            s = summary(d.name)
            s.pop("result", None)
            songs.append(s)
    songs.sort(key=lambda s: s.get("created", 0), reverse=True)
    return JSONResponse(songs)


async def upload(request: Request) -> Response:
    form = await request.form(max_files=1, max_fields=2, max_part_size=MAX_UPLOAD)
    f = form.get("file")
    if f is None or isinstance(f, str):
        return JSONResponse({"error": "No file."}, status_code=400)
    name = Path(f.filename or "song").name
    ext = Path(name).suffix.lower()
    if ext not in AUDIO_EXT:
        return JSONResponse({"error": "Unsupported format. Use MP3, M4A, WAV, FLAC, OGG or Opus."}, status_code=400)
    sid = uuid.uuid4().hex[:12]
    d = song_dir(sid)
    d.mkdir()
    size = 0
    with open(d / f"input{ext}", "wb") as fh:
        while chunk := await f.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                fh.close()
                shutil.rmtree(d)
                return JSONResponse({"error": "File too large (100 MB max)."}, status_code=413)
            fh.write(chunk)
    write_status(sid, name=Path(name).stem, input=f"input{ext}", size=size, created=time.time(),
                 state="queued", stage="Waiting", progress=0.0)
    jobs.put(sid)
    return JSONResponse(summary(sid), status_code=201)


async def get_song(request: Request) -> Response:
    sid = request.path_params["sid"]
    if not ID_RE.match(sid) or not song_dir(sid).exists():
        return JSONResponse({"error": "Not found."}, status_code=404)
    return JSONResponse(summary(sid))


async def delete_song(request: Request) -> Response:
    sid = request.path_params["sid"]
    if not ID_RE.match(sid) or not song_dir(sid).exists():
        return JSONResponse({"error": "Not found."}, status_code=404)
    if read_status(sid).get("state") == "processing":
        return JSONResponse({"error": "Still processing."}, status_code=409)
    shutil.rmtree(song_dir(sid))
    return Response(status_code=204)


async def song_file(request: Request) -> Response:
    sid, name = request.path_params["sid"], request.path_params["name"]
    if not ID_RE.match(sid) or not FILE_RE.match(name):
        return JSONResponse({"error": "Not found."}, status_code=404)
    p = song_dir(sid) / "out" / name
    if not p.exists():
        return JSONResponse({"error": "Not found."}, status_code=404)
    return FileResponse(p, headers={"Cache-Control": "private, max-age=86400"})


class RequireKey:
    """Reject /api requests without the shared key, when one is configured."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if KEY and scope["type"] == "http" and scope["path"].startswith("/api"):
            got = dict(scope["headers"]).get(b"x-layers-key", b"").decode("latin-1")
            if not hmac.compare_digest(got, KEY):
                await JSONResponse({"error": "Forbidden."}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


app = Starlette(routes=[
    Route("/api/songs", list_songs, methods=["GET"]),
    Route("/api/songs", upload, methods=["POST"]),
    Route("/api/songs/{sid}", get_song, methods=["GET"]),
    Route("/api/songs/{sid}", delete_song, methods=["DELETE"]),
    Route("/api/songs/{sid}/files/{name}", song_file, methods=["GET"]),
    Mount("/", StaticFiles(directory=WEB, html=True)),
])
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(RequireKey)

if __name__ == "__main__":
    requeue_unfinished()
    threading.Thread(target=worker, daemon=True).start()
    uvicorn.run(app, host=os.environ.get("LAYERS_HOST", "127.0.0.1"), port=int(os.environ.get("LAYERS_PORT", "8020")),
                log_level="warning")
