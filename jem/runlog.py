"""Local run directories: the only logging backend (read live by `jem dash`).

    runs/<run_id>/
        config.json     {"config": {...nested Config...}, "meta": {...}}
        status.json     {"state": "running|finished|failed|diverged", "step": int,
                         "updated": unix_time, "message": str}
        metrics.jsonl   one JSON object per line:
                        {"step": int, "time": unix_time, "<group>/<name>": float | null, ...}
                        groups: train/ diag/ eval/ perf/ ; non-finite values are written as null
        figures/<name>/<step:08d>.png
        checkpoints/last.pt

Append-only and plain text, so a crashed run still leaves a readable record.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

STATES = ("running", "finished", "failed", "diverged")


def _git_info(cwd: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5)
        return out.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "--short", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        return {}


def _scalar(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        value = value.item()
    value = float(value)
    return value if math.isfinite(value) else None


class RunLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._metrics = None

    @classmethod
    def create(cls, root: str | Path, name: str, config: dict, meta: dict | None = None) -> RunLog:
        root = Path(root)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "run"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = root / f"{stamp}-{slug}"
        suffix = 1
        while path.exists():
            suffix += 1
            path = root / f"{stamp}-{slug}-{suffix}"
        (path / "figures").mkdir(parents=True)
        (path / "checkpoints").mkdir()
        meta = {
            "name": name,
            "created": time.time(),
            "host": platform.node(),
            "argv": sys.argv,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "git": _git_info(Path(__file__).resolve().parent),
            **(meta or {}),
        }
        (path / "config.json").write_text(json.dumps({"config": config, "meta": meta}, indent=2))
        run = cls(path)
        run.set_status("running", 0)
        return run

    @property
    def id(self) -> str:
        return self.path.name

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        row = {"step": int(step), "time": time.time()}
        row.update({key: _scalar(value) for key, value in metrics.items()})
        if self._metrics is None:
            self._metrics = open(self.path / "metrics.jsonl", "a", buffering=1)
        self._metrics.write(json.dumps(row) + "\n")

    def figure(self, step: int, name: str, fig) -> Path:
        folder = self.path / "figures" / name
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"{int(step):08d}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        return out

    def set_status(self, state: str, step: int, message: str = "") -> None:
        assert state in STATES, state
        status = {"state": state, "step": int(step), "updated": time.time(), "message": message}
        tmp = self.path / "status.json.tmp"
        tmp.write_text(json.dumps(status))
        os.replace(tmp, self.path / "status.json")

    def save_checkpoint(self, obj: Any, name: str = "last.pt") -> Path:
        out = self.path / "checkpoints" / name
        torch.save(obj, out.with_suffix(".tmp"))
        os.replace(out.with_suffix(".tmp"), out)
        return out

    def close(self) -> None:
        if self._metrics is not None:
            self._metrics.close()
            self._metrics = None
