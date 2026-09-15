"""SIFT1M (INRIA TEXMEX): 1M base vectors, 10k queries, 100k learn, ground truth.

    python -m bench.datasets.sift1m            # download + extract into data/sift1m

The INRIA FTP server is the canonical source but is very slow from many
networks; a byte-identical copy of the same tarball on Hugging Face is tried
first. Integrity is checked against the published ground truth, not a hash:
brute-forcing a sample of queries must reproduce sift_groundtruth.ivecs.
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm

from index.flat import search_flat

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "sift1m"
URLS = [
    "https://huggingface.co/datasets/fzliu/sift1m/resolve/main/sift.tar.gz",
    "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
]


def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:].view(np.float32).copy()


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:].copy()


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0) or None
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=url) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def fetch(data_dir: Path = DATA) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "sift"
    if (out / "sift_base.fvecs").exists():
        return out
    tar = data_dir / "sift.tar.gz"
    if not tar.exists():
        for url in URLS:
            try:
                _download(url, tar)
                break
            except OSError as e:
                print(f"download from {url} failed: {e}")
        else:
            raise RuntimeError("could not download SIFT1M from any mirror")
    with tarfile.open(tar) as t:
        t.extractall(data_dir, filter="data")
    return out


def load(data_dir: Path = DATA) -> dict[str, np.ndarray]:
    d = fetch(data_dir)
    return {
        "base": read_fvecs(d / "sift_base.fvecs"),
        "query": read_fvecs(d / "sift_query.fvecs"),
        "learn": read_fvecs(d / "sift_learn.fvecs"),
        "gt": read_ivecs(d / "sift_groundtruth.ivecs"),
    }


def verify(ds: dict[str, np.ndarray], n_check: int = 20) -> None:
    base, query, gt = ds["base"], ds["query"], ds["gt"]
    assert base.shape == (1_000_000, 128), base.shape
    assert query.shape == (10_000, 128), query.shape
    assert gt.shape == (10_000, 100), gt.shape
    for i in range(n_check):
        ids, _ = search_flat(base, len(base), query[i], 10)
        # ties at the 10th position are possible; compare as sets on the first 9
        assert set(ids[:9].tolist()) <= set(gt[i, :10].tolist()), f"query {i} mismatch"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=DATA)
    args = ap.parse_args()
    ds = load(args.dir)
    verify(ds)
    print({k: v.shape for k, v in ds.items()}, "- ground truth verified")


if __name__ == "__main__":
    main()
