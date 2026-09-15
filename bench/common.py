"""Shared benchmark plumbing: system info, timing protocol, results, plots.

Protocol (see README): every measurement is 3 warmup runs + 5 timed runs and the
median is reported; tail latency is reported separately as p99 over per-query
timings. Numba JIT compilation happens during warmup and is never timed.
"""

from __future__ import annotations

import ctypes
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
CACHE = ROOT / "data" / "cache"

# Fixed identity -> colour map (validated categorical slots, light surface).
# Colour follows the engine everywhere, whatever subset a plot shows.
COLORS = {
    "siftdb-hnsw": "#2a78d6",
    "faiss-hnsw": "#eb6834",
    "siftdb-hnsw-pq": "#1baf7a",
    "faiss-ivfpq": "#eda100",
    "siftdb-hnsw-es": "#e87ba4",
    "faiss-ivfpq-refine": "#008300",
}
INK, INK2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def ram_gb() -> float:
    if sys.platform == "win32":

        class MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MemStatus()
        m.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return round(m.ullTotalPhys / 2**30, 1)
    import os

    return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)


def system_info(threads: int) -> dict[str, Any]:
    import numba

    info = {
        "cpu": cpu_name(),
        "ram_gb": ram_gb(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "numba_threading_layer": None,
        "threads": threads,
    }
    try:
        import faiss

        info["faiss"] = faiss.__version__
    except ImportError:
        pass
    return info


def timed(fn: Callable[[], Any], warmup: int, runs: int) -> tuple[list[float], Any]:
    """Run ``fn`` warmup + runs times; return the timed durations and the last result."""
    out = None
    for _ in range(warmup):
        out = fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return times, out


def median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def recall_at_k(found: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Mean |found[:k] ∩ gt[:k]| / k over queries."""
    f = found[:, :k]
    hit = (f[:, :, None] == gt[:, None, :k]).any(-1)
    return float(hit.sum(1).mean() / k)


def per_query_recall(found: np.ndarray, gt: np.ndarray, k: int) -> np.ndarray:
    f = found[:, :k]
    return (f[:, :, None] == gt[:, None, :k]).any(-1).sum(1) / k


def save_json(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def new_figure(width: float = 7.5, height: float = 4.6):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "siftdb",  # deterministic SVG ids -> clean diffs
        }
    )
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def save_figure(fig, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.svg"
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
