"""Local experiment dashboard over a runs/ directory written by `jem.runlog`.

Stdlib only: a threaded HTTP server with a small JSON API and one self-contained page
(static/index.html, no external resources), so it works offline and through SSH tunnels.

    GET /                                       the single-page app
    GET /api/runs                               run summaries, newest first
    GET /api/runs/<id>/config                   config.json: {"config": {...}, "meta": {...}}
    GET /api/runs/<id>/metrics?offset=<bytes>   {"rows": [...], "offset": int, "more": bool,
                                                 "reset": bool}
    GET /api/runs/<id>/figures                  {"<name>": [step, ...]}
    GET /api/info                               {"root": str, "host": str}
    GET /runs/<id>/figures/<name>/<file>.png    a figure image

The metrics endpoint returns only complete lines after `offset` and the byte offset to resume
from, so clients tail live runs cheaply; `more` means the size limit cut the response short
(ask again from the new offset), `reset` means the file shrank (rewritten) and rows restart
from byte 0. Non-finite numbers are sent as null. API responses carry Cache-Control: no-store.

Figure paths are validated segment by segment, then resolved and required to stay inside
their run directory; PNGs without their IEND trailer (still being written) are neither listed
nor served, so a browser never caches half an image.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import shutil
import socket
import socketserver
import sys
import threading
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

STATIC = Path(__file__).resolve().parent / "static"
HEADLINE = ("model.kind", "model.option_dim", "objective.name")
METRICS_LIMIT = 8 << 20  # max bytes of metrics.jsonl per response (the client pages with "more")
SCAN_TAIL = 1 << 20  # first look at a run reads at most this much of metrics.jsonl for /api/runs
_FIGURE_FILE = re.compile(r"^(\d+)\.png$")
_PNG_IEND = b"IEND\xaeB`\x82"  # the PNG trailer: absent while matplotlib is still writing
_RUN_FILES = ("config.json", "status.json", "metrics.jsonl")
_NO_CONFIG = {"config": {}, "meta": {}, "error": "config.json missing or unreadable"}
_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


def _safe_name(name: str) -> bool:
    """One path segment: no separators, no NUL, not hidden, not `.`/`..`."""
    return bool(name) and not name.startswith(".") and not any(c in name for c in "/\\\x00")


def _clean(obj: Any) -> Any:
    """Recursively replace non-finite floats with None (JSON has no NaN/Infinity)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _finite_or_none(value: Any) -> Any:
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _read_json(path: Path) -> Any:
    try:
        # NaN/Infinity literals in config files become strings so they still display.
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=str)
    except (OSError, ValueError):
        return None


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _lookup(tree: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(tree, dict) or part not in tree:
            return None
        tree = tree[part]
    return tree


def _parse_rows(data: bytes) -> list[dict[str, Any]]:
    """Parse complete JSONL lines into metric rows, skipping blank or malformed lines."""
    rows = []
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=lambda _: None)
        except ValueError:
            continue
        if not isinstance(row, dict) or _num(row.get("step")) is None:
            continue
        rows.append({k: _finite_or_none(v) for k, v in row.items()})
    return rows


def _read_lines(f, offset: int, size: int, limit: int) -> tuple[bytes, int, bool]:
    """Read whole lines from byte `offset`, about `limit` bytes at most.

    Returns (data, end, truncated): `data` stops after the last newline (a partial trailing
    line is left for the next call), `end` is the offset to resume from, and `truncated` says
    the limit stopped the read before `size`.
    """
    f.seek(offset)
    chunk = f.read(min(limit, max(0, size - offset)))
    cut = chunk.rfind(b"\n")
    while cut < 0 and chunk and offset + len(chunk) < size:  # one line longer than `limit`
        extra = f.read(min(limit, size - offset - len(chunk)))
        if not extra:
            break
        hit = extra.find(b"\n")
        cut = len(chunk) + hit if hit >= 0 else -1
        chunk += extra
    data = chunk[: cut + 1]
    return data, offset + len(data), offset + len(chunk) < size


@dataclass
class _Scan:
    """Incremental summary of one run's metrics.jsonl (for the run list)."""

    inode: int = -1
    offset: int = 0
    last_step: float | None = None
    first_time: float | None = None
    last_time: float | None = None
    eval: dict[str, Any] = field(default_factory=dict)
    eval_step: float | None = None

    def update(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            step = _num(row.get("step"))
            if step is not None and (self.last_step is None or step > self.last_step):
                self.last_step = step
            t = _num(row.get("time"))
            if self.first_time is None:
                self.first_time = t
            if t is not None and (self.last_time is None or t > self.last_time):
                self.last_time = t
            evals = {k: v for k, v in row.items() if k.startswith("eval/")}
            if evals:
                self.eval.update(evals)
                self.eval_step = step


class RunIndex:
    """Reads run directories under `root`; caches parsed configs and metric scans."""

    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self._configs: dict[str, tuple[tuple[int, int], dict]] = {}
        self._scans: dict[str, _Scan] = {}
        self._complete: dict[str, tuple[int, int]] = {}  # png path -> (mtime_ns, size) seen whole

    def png_complete(self, path: str | Path, st: os.stat_result) -> bool:
        """True once the PNG has its trailer, i.e. it is not still being written."""
        sig = (st.st_mtime_ns, st.st_size)
        key = str(path)
        if self._complete.get(key) == sig:
            return True
        try:
            with open(path, "rb") as f:
                f.seek(max(0, st.st_size - 1024))
                whole = _PNG_IEND in f.read()
        except OSError:
            return False
        if whole:
            with self._lock:
                if len(self._complete) > 100_000:
                    self._complete.clear()
                self._complete[key] = sig
        return whole

    def run_dir(self, run_id: str) -> Path | None:
        if not _safe_name(run_id):
            return None
        path = self.root / run_id
        return path if path.is_dir() else None

    def config(self, run_dir: Path) -> dict | None:
        path = run_dir / "config.json"
        try:
            st = path.stat()
        except OSError:
            return None
        sig = (st.st_mtime_ns, st.st_size)
        with self._lock:
            hit = self._configs.get(run_dir.name)
        if hit and hit[0] == sig:
            return hit[1]
        data = _read_json(path)
        if not isinstance(data, dict):  # missing or half-written: retry on the next request
            return None
        data = _clean(data)
        with self._lock:
            self._configs[run_dir.name] = (sig, data)
        return data

    def scan(self, run_dir: Path) -> _Scan:
        path = run_dir / "metrics.jsonl"
        with self._lock:
            scan = self._scans.get(run_dir.name) or _Scan()
            try:
                with open(path, "rb") as f:
                    st = os.fstat(f.fileno())
                    if st.st_ino != scan.inode or st.st_size < scan.offset:  # new or rewritten
                        scan = _Scan(inode=st.st_ino)
                        start = max(0, st.st_size - SCAN_TAIL)
                        if start:  # landed mid-line: resume after the next newline
                            f.seek(start - 1)
                            skip = f.read(SCAN_TAIL).find(b"\n")
                            start = st.st_size if skip < 0 else start + skip
                        scan.offset = start
                    if st.st_size > scan.offset:
                        data, scan.offset, _ = _read_lines(f, scan.offset, st.st_size, st.st_size)
                        scan.update(_parse_rows(data))
            except OSError:
                scan = _Scan()
            self._scans[run_dir.name] = scan
            return scan

    def summary(self, run_dir: Path) -> dict[str, Any]:
        doc = self.config(run_dir) or {}
        config = doc.get("config") if isinstance(doc.get("config"), dict) else {}
        meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
        status = _read_json(run_dir / "status.json")
        status = status if isinstance(status, dict) else {}
        scan = self.scan(run_dir)
        git = meta.get("git") if isinstance(meta.get("git"), dict) else {}
        steps = [s for s in (_num(status.get("step")), scan.last_step) if s is not None]
        # Last activity from the files' own timestamps (mtimes lie after cp/rsync).
        times = [t for t in (_num(status.get("updated")), scan.last_time) if t is not None]
        created = _num(meta.get("created")) or scan.first_time or _mtime(run_dir)
        state = status.get("state")
        return {
            "id": run_dir.name,
            "name": str(meta.get("name") or config.get("name") or run_dir.name),
            "state": state if isinstance(state, str) and state else "unknown",
            "message": str(status.get("message") or ""),
            "step": int(max(steps)) if steps else None,
            "updated": max(times) if times else created,
            "created": created,
            "commit": git.get("commit"),
            "git": git,
            "host": meta.get("host"),
            "preset": meta.get("preset"),
            "headline": {key: _lookup(config, key) for key in HEADLINE},
            "eval": scan.eval,
            "eval_step": int(scan.eval_step) if scan.eval_step is not None else None,
        }

    def runs(self) -> list[dict[str, Any]]:
        try:
            dirs = [p for p in self.root.iterdir() if _safe_name(p.name) and p.is_dir()]
        except OSError:
            return []
        dirs = [d for d in dirs if any((d / f).exists() for f in _RUN_FILES)]
        out = [self.summary(d) for d in dirs]
        out.sort(key=lambda r: (r["created"] or 0.0, r["id"]), reverse=True)
        with self._lock:  # forget runs that were deleted
            live = {r["id"] for r in out}
            for cache in (self._configs, self._scans):
                for stale in set(cache) - live:
                    del cache[stale]
        return out

    def metrics(self, run_dir: Path, offset: int, limit: int) -> dict[str, Any]:
        try:
            f = open(run_dir / "metrics.jsonl", "rb")
        except OSError:
            return {"rows": [], "offset": 0, "more": False, "reset": offset > 0}
        with f:
            size = os.fstat(f.fileno()).st_size
            reset = offset > size  # the file shrank: it was rewritten, start over
            if reset:
                offset = 0
            data, end, truncated = _read_lines(f, offset, size, limit)
        return {"rows": _parse_rows(data), "offset": end, "more": truncated, "reset": reset}

    def figures(self, run_dir: Path) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        try:
            folders = sorted(p for p in (run_dir / "figures").iterdir() if p.is_dir())
            base = run_dir.resolve(strict=True)
        except OSError:
            return out
        for folder in folders:
            if not _safe_name(folder.name) or not folder.resolve().is_relative_to(base):
                continue
            steps = []
            try:
                for entry in os.scandir(folder):
                    m = _FIGURE_FILE.match(entry.name)
                    if not m or not entry.is_file():
                        continue
                    if entry.is_symlink() and not Path(entry.path).resolve().is_relative_to(base):
                        continue  # the image route refuses these, so don't list them
                    if not self.png_complete(entry.path, entry.stat()):
                        continue  # still being written
                    steps.append(int(m.group(1)))
            except OSError:
                continue
            if steps:
                out[folder.name] = sorted(set(steps))
        return out

    def figure_path(self, run_id: str, name: str, filename: str) -> Path | None:
        run_dir = self.run_dir(run_id)
        if run_dir is None or not _safe_name(name) or not _safe_name(filename):
            return None
        if not filename.endswith(".png"):
            return None
        folder = run_dir / "figures" / name
        path = folder / filename
        if not path.exists():  # tolerate other zero-paddings of the step number
            m = _FIGURE_FILE.match(filename)
            if not m:
                return None
            want = int(m.group(1))
            try:
                hits = [
                    e.name
                    for e in os.scandir(folder)
                    if (n := _FIGURE_FILE.match(e.name)) and int(n.group(1)) == want
                ]
            except OSError:
                return None
            if not hits:
                return None
            path = folder / hits[0]
        try:
            resolved = path.resolve(strict=True)
            base = run_dir.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if not resolved.is_relative_to(base) or not resolved.is_file():
            return None
        if not self.png_complete(resolved, resolved.stat()):
            return None  # half-written: the client retries once it is listed
        return resolved


class _Handler(BaseHTTPRequestHandler):
    server: DashboardServer
    protocol_version = "HTTP/1.1"
    timeout = 120  # drop idle keep-alive connections

    def version_string(self) -> str:
        return "jem-dashboard"

    def do_GET(self) -> None:
        self._dispatch(head=False)

    def do_HEAD(self) -> None:
        self._dispatch(head=True)

    def log_message(self, format: str, *args: Any) -> None:
        pass  # polling would flood the terminal; errors are reported in _dispatch

    def _dispatch(self, head: bool) -> None:
        self._head = head
        try:
            self._route()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            traceback.print_exc(file=sys.stderr)
            try:
                self._json({"error": "internal error"}, 500)
            except OSError:
                self.close_connection = True

    def _route(self) -> None:
        url = urlsplit(self.path)
        parts = url.path.split("/")[1:]
        if parts and parts[-1] == "":
            parts.pop()
        # Decode each segment separately, so an encoded "/" can never create new segments.
        parts = [unquote(p) for p in parts]
        index = self.server.index
        match parts:
            case [] | ["index.html"]:
                self._index()
            case ["api", "runs"]:
                self._json(index.runs())
            case ["api", "info"]:
                self._json({"root": str(index.root), "host": socket.gethostname()})
            case ["api", "runs", run_id, endpoint]:
                run_dir = index.run_dir(run_id)
                if run_dir is None:
                    self._json({"error": f"unknown run {run_id!r}"}, 404)
                elif endpoint == "config":
                    doc = index.config(run_dir)
                    self._json(doc if doc is not None else _NO_CONFIG)
                elif endpoint == "metrics":
                    self._metrics(run_dir, parse_qs(url.query))
                elif endpoint == "figures":
                    self._json(index.figures(run_dir))
                else:
                    self._json({"error": "not found"}, 404)
            case ["runs", run_id, "figures", name, filename]:
                path = index.figure_path(run_id, name, filename)
                if path is None:
                    self._plain("not found", 404)
                else:
                    self._file(path, "image/png")
            case _:
                if parts and parts[0] == "api":
                    self._json({"error": "not found"}, 404)
                else:
                    self._plain("not found", 404)

    def _metrics(self, run_dir: Path, query: dict[str, list[str]]) -> None:
        try:
            offset = int(query.get("offset", ["0"])[-1])
            limit = int(query.get("limit", [str(METRICS_LIMIT)])[-1])
        except ValueError:
            self._json({"error": "offset and limit must be integers"}, 400)
            return
        if offset < 0 or limit < 1:
            self._json({"error": "offset must be >= 0 and limit >= 1"}, 400)
            return
        self._json(self.server.index.metrics(run_dir, offset, min(limit, 8 * METRICS_LIMIT)))

    # -- responses -------------------------------------------------------------------------

    def _send(
        self, status: int, body: bytes, ctype: str, headers: dict[str, str], compress: bool = True
    ) -> None:
        accepts = "gzip" in self.headers.get("Accept-Encoding", "")
        if compress and accepts and len(body) > 1024:
            body = gzip.compress(body, compresslevel=5)
            headers = {**headers, "Content-Encoding": "gzip", "Vary": "Accept-Encoding"}
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if not self._head:
            self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        try:
            body = json.dumps(obj, separators=(",", ":"), allow_nan=False)
        except ValueError:  # a non-finite float slipped through: scrub and retry
            body = json.dumps(_clean(obj), separators=(",", ":"), allow_nan=False)
        self._send(
            status, body.encode(), "application/json; charset=utf-8", {"Cache-Control": "no-store"}
        )

    def _plain(self, text: str, status: int) -> None:
        headers = {"Cache-Control": "no-store"}
        self._send(status, text.encode(), "text/plain; charset=utf-8", headers)

    def _index(self) -> None:
        body = (STATIC / "index.html").read_bytes()
        self._send(
            200,
            body,
            "text/html; charset=utf-8",
            {"Cache-Control": "no-cache", "Content-Security-Policy": _CSP},
        )

    def _file(self, path: Path, ctype: str) -> None:
        st = path.stat()
        etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        if etag in self.headers.get("If-None-Match", ""):
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(st.st_size))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")  # revalidate: a live run may rewrite a file
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if not self._head:
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root: str | Path, host: str, port: int):
        self.index = RunIndex(Path(root).expanduser().resolve())
        if ":" in host:
            self.address_family = socket.AF_INET6
        super().__init__((host, port), _Handler)

    def server_bind(self) -> None:
        # HTTPServer.server_bind also calls socket.getfqdn(), which can stall on DNS.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    @property
    def root(self) -> Path:
        return self.index.root

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        if host in ("0.0.0.0", "::", ""):
            host = socket.gethostname()
        elif ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{port}/"


def make_server(root: str | Path, host: str = "127.0.0.1", port: int = 8765) -> DashboardServer:
    """Build (but don't start) the dashboard server; port=0 picks a free port."""
    return DashboardServer(root, host, port)


def serve(root: str | Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the dashboard for the runs under `root` until interrupted (blocking)."""
    server = make_server(root, host, port)
    note = "" if server.root.is_dir() else " (does not exist yet; runs appear once it does)"
    print(f"jem dashboard: {server.url}  root={server.root}{note}", flush=True)
    if host in ("0.0.0.0", "::", ""):
        print("warning: listening on all interfaces and there is no authentication", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
