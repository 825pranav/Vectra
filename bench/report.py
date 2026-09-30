"""Render the README's result tables from bench/results/*.json.

    python -m bench.report

Numbers in the README are pasted from this script's output, never typed by hand.
"""

from __future__ import annotations

import json

from bench.ann import qps_at_recall
from bench.common import RESULTS


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def fmt(v, spec: str = ".0f", none: str = "n/a") -> str:
    return none if v is None else format(v, spec)


def pt_near(points: list[dict], key: str, value) -> dict:
    return next(p for p in points if p[key] == value)


def hnsw_table() -> str:
    d = load("sift1m_hnsw")
    rows = ["| engine | build (16 threads) | index RAM | QPS @ recall 0.95 | ef for 0.95 | "
            "1-thread p50 / p99 @ ef=64 |", "|---|---|---|---|---|---|"]  # fmt: skip
    for e in d["engines"]:
        q = qps_at_recall(e["points"], 0.95)
        first = next((p for p in sorted(e["points"], key=lambda p: p["ef"]) if p["recall"] >= 0.95),
                     None)  # fmt: skip
        p64 = pt_near(e["points"], "ef", 64)
        rows.append(
            f"| {e['name']} | {e['build_s']:.0f} s | {e['index_bytes'] / 2**20:.0f} MiB | "
            f"{q / 1e3:.1f}k | {first['ef'] if first else 'n/a'} | "
            f"{p64['lat_p50_ms'] * 1e3:.0f} / {p64['lat_p99_ms'] * 1e3:.0f} µs |"
        )
    return "\n".join(rows)


def pq_table() -> str:
    d = load("sift1m_pq")
    rows = ["| engine | index RAM | vectors held in RAM | QPS @ 0.90 | QPS @ 0.95 | best recall |",
            "|---|---|---|---|---|---|"]  # fmt: skip
    for e in d["engines"]:
        q90, q95 = qps_at_recall(e["points"], 0.90), qps_at_recall(e["points"], 0.95)
        best = max(p["recall"] for p in e["points"])
        if e["kind"] == "vectra_hnsw_pq":
            held = f"16-byte codes ({e['code_bytes'] / 2**20:.1f} MiB)"
        elif e["name"] == "faiss-ivfpq":
            held = "16-byte codes"
        else:
            held = "float32 (488 MiB)"
        rows.append(
            f"| {e['name']} | {e['index_bytes'] / 2**20:.0f} MiB | {held} | "
            f"{fmt(q90 and q90 / 1e3, '.1f')}k | {fmt(q95 and q95 / 1e3, '.1f')}k | {best:.3f} |"
        )
    return "\n".join(rows).replace("n/ak", "n/a")


def filters_table() -> str:
    d = load("sift1m_filters")
    rows = ["| strategy | mean latency | p99 | recall@10 |", "|---|---|---|---|"]
    for s, v in d["test"]["overall"].items():
        rows.append(f"| {s} | {v['mean_ms']:.2f} ms | {v['p99_ms']:.1f} ms | {v['recall']:.3f} |")
    buckets = ["| selectivity | queries | post_filter | bitmap | brute_force | planner |",
               "|---|---|---|---|---|---|"]  # fmt: skip
    for b in d["test"]["buckets"]:
        cells = [f"{b[s]['mean_ms']:.2f} ms ({b[s]['recall']:.2f})"
                 for s in ("post_filter", "bitmap", "brute_force", "planner")]  # fmt: skip
        buckets.append(f"| {b['lo']:g}–{b['hi']:g} | {b['queries']} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n\n" + "\n".join(buckets)


def early_stop_table() -> str:
    rows = ["| dataset | recall | fixed ef | upfront (Δ) | checkpoint (Δ) | "
            "Δ distance computations (upfront / checkpoint) |",
            "|---|---|---|---|---|---|"]  # fmt: skip

    def cell(v, reduction):
        return "n/a" if v is None else f"{v * 1e3:.0f} µs ({-(reduction or 0):+.1f}%)"

    def delta(reduction):
        return "n/a" if reduction is None else f"{-reduction:+.1f}%"

    for name, label in (("sift1m_early_stop", "SIFT1M"), ("msmarco_early_stop", "MS MARCO 300k")):
        for t, r in load(name)["latency_at_recall"].items():
            rows.append(
                f"| {label} | {t} | {fmt(r['fixed-ef_ms'] and r['fixed-ef_ms'] * 1e3)} µs | "
                f"{cell(r['upfront_ms'], r['upfront_reduction_pct'])} | "
                f"{cell(r['checkpoint_ms'], r['checkpoint_reduction_pct'])} | "
                f"{delta(r['upfront_ndist_reduction_pct'])} / "
                f"{delta(r['checkpoint_ndist_reduction_pct'])} |"
            )
    return "\n".join(rows)


def crash_line() -> str:
    d = load("crash")
    inflight = sum(1 for r in d["results"] if r.get("in_flight_delete"))
    return (f"{d['passed']}/{d['runs']} kill -9 runs recovered every acknowledged write "
            f"({d['total_acked_records']:,} acknowledged records verified bit-for-bit; "
            f"{inflight} kills landed mid-delete).")  # fmt: skip


def main() -> None:
    for title, fn in [("HNSW", hnsw_table), ("PQ", pq_table), ("Filters", filters_table),
                      ("Early stop", early_stop_table), ("Crash", crash_line)]:  # fmt: skip
        print(f"### {title}\n\n{fn()}\n")


if __name__ == "__main__":
    main()
