"""Shared benchmark plumbing: system info, timing protocol, results, plots.

Protocol (see README): every measurement is 3 warmup runs + 5 timed runs and the
median is reported; tail latency is reported separately as p99 over per-query
timings. Numba JIT compilation happens during warmup and is never timed.
"""

from __future__ import annotations

# Stdlib for system info, timing, JSON results and paths; NumPy for recall maths.
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

# Shared folders: results JSON/SVG output and the cache of prebuilt indexes.
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
CACHE = ROOT / "data" / "cache"

# Fixed identity -> colour map (validated categorical slots, light surface).
# Colour follows the engine everywhere, whatever subset a plot shows.
COLORS = {
    "vectra-hnsw": "#2a78d6",
    "faiss-hnsw": "#eb6834",
    "vectra-hnsw-pq": "#1baf7a",
    "faiss-ivfpq": "#eda100",
    "vectra-hnsw-es": "#e87ba4",
    "faiss-ivfpq-refine": "#008300",
}
# Palette for text, grid lines, axes and background used by every plot.
INK, INK2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)


# CPU model string for the results file, read per OS (registry, /proc/cpuinfo, or platform).
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


# Total RAM in GB for the results file; Windows needs a ctypes call, others use sysconf.
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


# Hardware and library versions saved with every result, so numbers can be compared fairly.
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
    # FAISS is optional; record its version only if it is installed.
    try:
        import faiss

        info["faiss"] = faiss.__version__
    except ImportError:
        pass
    return info


# Context manager that pins the current thread to one CPU while timing single-thread latency.
class pinned_to_cpu:
    """Pin the calling thread to one logical CPU for single-threaded timing.

    Hybrid Intel parts mix P- and E-cores (on the reference laptop, logical
    CPUs 0-11 are P-cores and 12-19 E-cores, ~1.6x slower); an unpinned thread
    migrates between them mid-run and turns latency numbers into noise.
    ``cpu=None`` leaves scheduling alone.
    """

    # Remember which CPU to pin to; the old affinity is saved on enter so exit can restore it.
    def __init__(self, cpu: int | None) -> None:
        self.cpu = cpu
        self._old = None

    # On enter: save the current affinity and pin to self.cpu (Windows API or Linux sched call).
    def __enter__(self):
        if self.cpu is None:
            return self
        if sys.platform == "win32":
            k32 = ctypes.windll.kernel32
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadAffinityMask.restype = ctypes.c_size_t
            k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self._old = k32.SetThreadAffinityMask(k32.GetCurrentThread(), 1 << self.cpu)
        else:
            import os

            self._old = os.sched_getaffinity(0)
            os.sched_setaffinity(0, {self.cpu})
        time.sleep(0.05)
        return self

    # On exit: put the old affinity back so later code is not stuck on one core.
    def __exit__(self, *exc):
        if self.cpu is None or self._old is None:
            return False
        if sys.platform == "win32":
            k32 = ctypes.windll.kernel32
            k32.SetThreadAffinityMask(k32.GetCurrentThread(), self._old)
        else:
            import os

            os.sched_setaffinity(0, self._old)
        return False


# Benchmark timing protocol: untimed warmup runs (JIT compile), then timed runs.
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


# Median of the timed runs, the number we report.
def median(xs: list[float]) -> float:
    return float(statistics.median(xs))


# Recall@k: average share of each query's true top-k that the engine found.
def recall_at_k(found: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Mean |found[:k] ∩ gt[:k]| / k over queries."""
    f = found[:, :k]
    hit = (f[:, :, None] == gt[:, None, :k]).any(-1)
    return float(hit.sum(1).mean() / k)


# Same as recall_at_k but one value per query (used for buckets and training labels).
def per_query_recall(found: np.ndarray, gt: np.ndarray, k: int) -> np.ndarray:
    f = found[:, :k]
    return (f[:, :, None] == gt[:, None, :k]).any(-1).sum(1) / k


# Write a results dict to bench/results/<name>.json; returns the path.
def save_json(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


# Apply the shared plot style: light background, grid, no top/right borders, muted ticks.
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


# Make a styled matplotlib figure using the file-only Agg backend (no window needed).
def new_figure(width: float = 7.5, height: float = 4.6):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "vectra",  # deterministic SVG ids -> clean diffs
        }
    )
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


# Put each line's label right next to it, spreading labels so they never overlap.
def direct_labels(ax, anchors: list[tuple[tuple[float, float], str, str]], min_gap: float = 13.0):
    """Label each line at its anchor point, nudged apart vertically (in pixels)
    so labels never overlap each other; a thin leader ties a nudged label back.

    ``anchors``: [((x, y), text, color)] in data coordinates.
    """
    # Convert anchors to pixel space and sort them bottom to top.
    if not anchors:
        return
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    pts = sorted(((to_px(xy), xy, t, c) for xy, t, c in anchors), key=lambda a: a[0][1])
    ys = [p[0][1] for p in pts]
    for i in range(1, len(ys)):  # push up until every pair is min_gap apart
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    shift = (sum(p[0][1] for p in pts) - sum(ys)) / len(ys)  # re-centre the stack
    # Draw each label at its spread-out position, with a thin leader line if it moved far.
    for (px, xy, text, color), y in zip(pts, ys, strict=True):
        dy = y + shift - px[1]
        ax.annotate(
            text, xy, xytext=(8, dy), textcoords="offset points", va="center", ha="left",
            fontsize=9, color=INK2,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8} if abs(dy) > 4 else None,
        )  # fmt: skip


# Save a figure as SVG next to the results JSON, then close it to free memory.
def save_figure(fig, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.svg"
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
