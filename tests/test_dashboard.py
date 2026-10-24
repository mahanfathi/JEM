"""Dashboard server: API shapes, incremental metrics, figures, and path-traversal guards."""

import gzip
import http.client
import json
import re
import threading
import urllib.error
import urllib.request

import pytest
from matplotlib.figure import Figure

from jem.dashboard import make_server
from jem.runlog import RunLog

CONFIG = {
    "name": "alpha",
    "model": {"kind": "linear", "option_dim": 8},
    "objective": {"name": "jem"},
    "train": {"steps": 400, "lr": 0.001},
}
PARTIAL = "19700101-000000-partial"


def _figure():
    fig = Figure(figsize=(1, 1))
    fig.subplots().plot([0, 1], [1, 0])
    return fig


@pytest.fixture
def root(tmp_path):
    runs = tmp_path / "runs"
    a = RunLog.create(runs, "alpha", CONFIG, meta={"created": 1000.0})
    for step in range(0, 400, 50):
        a.log(step, {"train/nll": 1.0 / (step + 1), "diag/scale": float(step)})
        if step == 200:
            a.log(step, {"eval/nmi": 0.5})
    a.log(400, {"train/nll": float("nan"), "diag/scale": float("inf")})
    for step, name in [(200, "options"), (400, "options"), (400, "rollouts")]:
        a.figure(step, name, _figure())
    a.set_status("finished", 400, "done")
    a.close()

    b_config = {**CONFIG, "name": "beta", "model": {"kind": "mlp", "option_dim": 2}}
    b = RunLog.create(runs, "beta", b_config, meta={"created": 2000.0})
    b.log(0, {"train/nll": 3.0})
    b.log(50, {"train/nll": 2.0})
    b.close()
    with open(b.path / "metrics.jsonl", "a") as f:  # a live run mid-write
        f.write('{"step": 100, "time": 1.0, "train/nll": 1.')

    partial = runs / PARTIAL  # crashed before logging anything
    partial.mkdir()
    doc = {"config": {"name": "partial"}, "meta": {"name": "partial", "created": 500.0}}
    (partial / "config.json").write_text(json.dumps(doc))
    (tmp_path / "secret.png").write_bytes(b"\x89PNG\r\n\x1a\nsecret")  # outside the root
    return runs, a.id, b.id


def _start(root_dir):
    srv = make_server(root_dir, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address[:2]
    return srv, {"base": f"http://{host}:{port}", "host": host, "port": port}


@pytest.fixture
def server(root):
    runs, a, b = root
    srv, info = _start(runs)
    yield {**info, "runs": runs, "a": a, "b": b}
    srv.shutdown()
    srv.server_close()


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, res.headers, res.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def get_json(url):
    status, headers, body = get(url)
    assert status == 200, (url, status, body)
    assert headers["Content-Type"].startswith("application/json")
    assert headers["Cache-Control"] == "no-store"
    return json.loads(body)


def raw_status(server, path):
    """GET `path` byte for byte; urllib would normalise dot segments first."""
    conn = http.client.HTTPConnection(server["host"], server["port"], timeout=10)
    try:
        conn.request("GET", path)
        res = conn.getresponse()
        res.read()
        return res.status
    finally:
        conn.close()


def test_index_is_self_contained(server):
    status, headers, body = get(server["base"] + "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    html = body.decode()
    assert "<canvas" in html or "createElement" in html
    # Nothing may be fetched from another origin: the dashboard must work offline.
    assert not re.search(r"""(?:src|href)\s*=\s*["']?(?:https?:)?//""", html, re.I)
    assert not re.search(r"""url\(\s*["']?(?:https?:)?//""", html, re.I)
    assert not re.search(r"""fetch\(\s*["'`](?:https?:)?//""", html, re.I)
    assert "@import" not in html


def test_runs_list(server):
    runs = get_json(server["base"] + "/api/runs")
    assert [r["id"] for r in runs] == [server["b"], server["a"], PARTIAL]  # newest first
    keys = {"id", "name", "state", "step", "updated", "created", "commit", "headline", "eval"}
    for run in runs:
        assert keys <= set(run)
    beta, alpha, partial = runs
    assert alpha["name"] == "alpha"
    assert (alpha["state"], alpha["step"], alpha["message"]) == ("finished", 400, "done")
    assert alpha["created"] == 1000.0
    headline = {"model.kind": "linear", "model.option_dim": 8, "objective.name": "jem"}
    assert alpha["headline"] == headline
    assert alpha["eval"] == {"eval/nmi": 0.5}
    assert alpha["eval_step"] == 200
    assert (beta["state"], beta["step"]) == ("running", 50)  # the partial line is not counted
    assert beta["headline"]["model.kind"] == "mlp"
    assert (partial["name"], partial["state"], partial["step"]) == ("partial", "unknown", None)
    assert partial["eval"] == {}


def test_config(server):
    doc = get_json(f"{server['base']}/api/runs/{server['a']}/config")
    assert doc["config"] == CONFIG
    assert doc["meta"]["name"] == "alpha"
    assert {"created", "host", "argv", "git"} <= set(doc["meta"])
    assert get_json(f"{server['base']}/api/runs/{PARTIAL}/config")["config"] == {"name": "partial"}
    status, headers, body = get(f"{server['base']}/api/runs/no-such-run/config")
    assert status == 404
    assert "error" in json.loads(body)


def test_metrics_are_read_incrementally(server):
    base, b = server["base"], server["b"]
    path = server["runs"] / b / "metrics.jsonl"
    first = get_json(f"{base}/api/runs/{b}/metrics?offset=0")
    assert [row["step"] for row in first["rows"]] == [0, 50]
    assert first["offset"] == path.read_bytes().rindex(b"\n") + 1  # stops before the partial line
    assert (first["more"], first["reset"]) == (False, False)

    again = get_json(f"{base}/api/runs/{b}/metrics?offset={first['offset']}")
    assert again == {"rows": [], "offset": first["offset"], "more": False, "reset": False}

    with open(path, "a") as f:  # the writer finishes its line and keeps going
        f.write('5, "extra": 7}\n')
        f.write("not json\n")
        f.write('{"step": 150, "time": 2.0, "train/nll": NaN, "diag/z": Infinity}\n')
        f.write('{"step": 200, "time": 3.0, "eval/nmi": 0.9}\n')
        f.write('{"step": 250')
    new = get_json(f"{base}/api/runs/{b}/metrics?offset={first['offset']}")
    assert new["rows"] == [
        {"step": 100, "time": 1.0, "train/nll": 1.5, "extra": 7},
        {"step": 150, "time": 2.0, "train/nll": None, "diag/z": None},
        {"step": 200, "time": 3.0, "eval/nmi": 0.9},
    ]
    assert new["offset"] == path.stat().st_size - len('{"step": 250')

    beta = get_json(base + "/api/runs")[0]  # the run list picks up the new rows too
    assert (beta["step"], beta["eval"], beta["eval_step"]) == (200, {"eval/nmi": 0.9}, 200)


def test_metrics_nonfinite_paging_and_reset(server):
    base, a = server["base"], server["a"]
    rows = get_json(f"{base}/api/runs/{a}/metrics")["rows"]  # offset defaults to 0
    last = {"step": 400, "time": rows[-1]["time"], "train/nll": None, "diag/scale": None}
    assert rows[-1] == last
    assert sum("eval/nmi" in row for row in rows) == 1

    offset, paged = 0, []
    for _ in range(100):  # a tiny limit still returns whole lines, and flags what is left
        page = get_json(f"{base}/api/runs/{a}/metrics?offset={offset}&limit=64")
        assert page["rows"] or not page["more"]
        paged += page["rows"]
        offset = page["offset"]
        if not page["more"]:
            break
    assert paged == rows

    size = (server["runs"] / a / "metrics.jsonl").stat().st_size
    rewritten = get_json(f"{base}/api/runs/{a}/metrics?offset={size + 10}")
    assert rewritten["reset"] is True
    assert rewritten["rows"] == rows

    for bad in ("-1", "abc"):
        assert get(f"{base}/api/runs/{a}/metrics?offset={bad}")[0] == 400
    empty = get_json(f"{base}/api/runs/{PARTIAL}/metrics?offset=0")
    assert empty == {"rows": [], "offset": 0, "more": False, "reset": False}


def test_gzip_when_accepted(server):
    with open(server["runs"] / server["a"] / "metrics.jsonl", "a") as f:
        for step in range(500, 1500, 10):  # big enough to be worth compressing
            f.write(json.dumps({"step": step, "time": 0.0, "train/nll": 1 / step}) + "\n")
    url = f"{server['base']}/api/runs/{server['a']}/metrics"
    status, headers, body = get(url, {"Accept-Encoding": "gzip"})
    assert status == 200
    assert headers["Content-Encoding"] == "gzip"
    assert json.loads(gzip.decompress(body)) == get_json(url)


def test_figures(server):
    base, a, b = server["base"], server["a"], server["b"]
    assert get_json(f"{base}/api/runs/{a}/figures") == {"options": [200, 400], "rollouts": [400]}
    assert get_json(f"{base}/api/runs/{b}/figures") == {}
    assert get_json(f"{base}/api/runs/{PARTIAL}/figures") == {}

    url = f"{base}/runs/{a}/figures/options/00000200.png"
    status, headers, body = get(url)
    assert status == 200
    assert headers["Content-Type"] == "image/png"
    assert body == (server["runs"] / a / "figures" / "options" / "00000200.png").read_bytes()
    assert body.startswith(b"\x89PNG\r\n\x1a\n")
    assert get(url, {"If-None-Match": headers["ETag"]})[0] == 304
    assert get(f"{base}/runs/{a}/figures/options/200.png")[0] == 200  # other zero-padding
    assert get(f"{base}/runs/{a}/figures/options/00000300.png")[0] == 404
    assert get(f"{base}/runs/{a}/figures/nope/00000200.png")[0] == 404
    assert get(f"{base}/runs/no-such-run/figures/options/00000200.png")[0] == 404


def test_half_written_figures_are_hidden(server):
    base, a = server["base"], server["a"]
    folder = server["runs"] / a / "figures" / "options"
    png = (folder / "00000400.png").read_bytes()
    partial = folder / "00000600.png"
    partial.write_bytes(png[: len(png) // 2])  # matplotlib is still writing it
    assert get_json(f"{base}/api/runs/{a}/figures")["options"] == [200, 400]
    assert get(f"{base}/runs/{a}/figures/options/00000600.png")[0] == 404
    partial.write_bytes(png)
    assert get_json(f"{base}/api/runs/{a}/figures")["options"] == [200, 400, 600]
    assert get(f"{base}/runs/{a}/figures/options/00000600.png")[0] == 200


def test_path_traversal_is_rejected(server):
    a, runs = server["a"], server["runs"]
    # A symlink inside the run that points outside it is neither listed nor served.
    (runs / a / "figures" / "options" / "00000999.png").symlink_to(runs.parent / "secret.png")
    assert get_json(f"{server['base']}/api/runs/{a}/figures")["options"] == [200, 400]
    attempts = [
        f"/runs/{a}/figures/../../config.json",
        f"/runs/{a}/figures/options/../../config.json",
        f"/runs/{a}/figures/options/../../../secret.png",
        f"/runs/{a}/figures/options/..%2F..%2Fconfig.json",
        f"/runs/{a}/figures/options/..%2F..%2F..%2Fsecret.png",
        f"/runs/{a}/figures/options/%2e%2e%2f%2e%2e%2f%2e%2e%2fsecret.png",
        f"/runs/{a}/figures/%2e%2e/%2e%2e%2fconfig.json",
        f"/runs/{a}/figures/..%5c..%5cconfig.json/x.png",
        f"/runs/{a}/figures/%2E%2E/..%252F..%252Fsecret.png",
        f"/runs/..%2F{a}/figures/options/00000200.png",
        "/runs/%2e%2e/figures/options/00000200.png",
        f"/runs/{a}/figures/options/00000999.png",
        f"/runs/{a}/figures/options/00000200.png%00.txt",
        f"/runs/{a}/figures/options/.hidden.png",
        f"/runs/{a}/config.json",
        f"/runs/{a}/metrics.jsonl",
        f"/runs/{a}/checkpoints/last.pt",
        "/api/runs/..%2F..%2Fetc/config",
        "/api/runs/../config",
        "/api/runs/%2e%2e/metrics",
        "/static/../server.py",
        "/../../../etc/passwd",
        "/%2e%2e/%2e%2e/etc/passwd",
    ]
    for path in attempts:
        status = raw_status(server, path)
        assert 400 <= status < 500, (path, status)
    assert raw_status(server, f"/runs/{a}/figures/options/00000200.png") == 200


@pytest.mark.parametrize("exists", [True, False])
def test_empty_or_missing_root(tmp_path, exists):
    root = tmp_path / "runs"
    if exists:
        (root / "not-a-run").mkdir(parents=True)  # no run files: ignored
        (root / "notes.txt").write_text("hi")
    srv, info = _start(root)
    try:
        assert get_json(info["base"] + "/api/runs") == []
        assert get(info["base"] + "/api/runs/not-a-run/metrics")[0] in (200, 404)
        assert get(info["base"] + "/api/runs/nope/figures")[0] == 404
        assert get(info["base"] + "/")[0] == 200
        assert get(info["base"] + "/api/nope")[0] == 404
    finally:
        srv.shutdown()
        srv.server_close()
