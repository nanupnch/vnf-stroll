"""Shared helpers for the experiment scripts: output paths, run metadata and table formatting."""

from __future__ import annotations

import csv
import datetime
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def _run(cmd) -> str:
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def code_version() -> str:
    """Commit of the code that produced a result, with "-dirty" if files other than results/ differ from it."""
    head = _run(["git", "rev-parse", "--short", "HEAD"])
    if not head:
        return "no commit yet"
    changed = _run(["git", "status", "--porcelain", "--", ".", ":(exclude)results"])
    return head + ("-dirty" if changed else "")


def metadata(**extra) -> dict:
    cpu = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    return {
        "command": " ".join(["python", *sys.orig_argv[1:]]),
        "date (UTC)": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M"),
        "code version": code_version(),
        "python / numpy / scipy": f"{platform.python_version()} / {np.__version__} / {scipy.__version__}",
        "machine": f"{platform.system()} {platform.machine()}, {cpu}",
        **extra,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(header: list, rows: list[list]) -> str:
    lines = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def md_metadata(meta: dict) -> str:
    return "\n".join(f"- {k}: `{v}`" if k == "command" else f"- {k}: {v}" for k, v in meta.items())


def write_report(path: Path, title: str, meta: dict, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.strip()}\n\n## Run\n\n{md_metadata(meta)}\n")


def mean_ci(values) -> tuple[float, float]:
    """Mean and half-width of a 95% t confidence interval."""
    from scipy import stats

    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
