"""Neural dataset: 300k MS MARCO passages embedded with all-MiniLM-L6-v2 (384-d).

    python -m bench.datasets.msmarco          # download, embed (GPU), ground truth

* base   first 300,000 passages of the MS MARCO passage collection
* query  first 10,000 MS MARCO *dev* queries  (held-out test split)
* learn  first 20,000 MS MARCO *train* queries (only split used for training)
* gt     exact top-100 neighbours of each test query, brute-forced once

Only the first slice of the 1 GB collection archive is needed, so it is read
as a stream and the connection is dropped after 300k lines. Embeddings are
L2-normalised, so squared L2 ranks exactly like cosine similarity. The GPU is
used here, offline, and never at query time.
"""

from __future__ import annotations

# Stdlib for CLI, streaming tar download and paths; NumPy; brute-force ground truth helper.
import argparse
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from bench.datasets.gt import brute_force_topk

# Where the dataset lives on disk, where it is downloaded from, the embedding model and split sizes.
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "msmarco300k"
BASE_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_BASE, N_QUERY, N_LEARN = 300_000, 10_000, 20_000


# Read the first `limit` texts out of a remote .tar.gz without downloading the whole archive.
def _stream_tsv(url: str, member: str, limit: int) -> list[str]:
    """First ``limit`` second-column values of a TSV inside a remote .tar.gz."""
    texts: list[str] = []
    # Open the URL as a stream and walk tar members until we hit the TSV we want.
    with urllib.request.urlopen(url, timeout=60) as r:
        with tarfile.open(fileobj=r, mode="r|gz") as tar:
            for info in tar:
                if not info.name.endswith(member):
                    continue
                # stream-mode members are not seekable, so no TextIOWrapper here
                # Keep the text column of each line; stop and drop the connection once we have enough.
                for raw in tar.extractfile(info):
                    texts.append(raw.decode("utf-8").rstrip("\r\n").split("\t", 1)[1])
                    if len(texts) >= limit:
                        return texts
    return texts


# Turn texts into unit-length 384-d float32 vectors with MiniLM (GPU if there is one).
def _embed(texts: list[str], batch: int = 256) -> np.ndarray:
    # Imported here so loading cached .npy files never needs torch installed.
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=device)
    emb = model.encode(
        texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=True,
        convert_to_numpy=True,
    )  # fmt: skip
    return emb.astype(np.float32)


# One-time offline build: download and embed each split, then compute exact top-100 ground truth.
def build(data_dir: Path = DATA) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    # Each split: which archive, which TSV inside it, and how many rows to take.
    parts = {
        "base": (f"{BASE_URL}/collection.tar.gz", "collection.tsv", N_BASE),
        "query": (f"{BASE_URL}/queries.tar.gz", "queries.dev.tsv", N_QUERY),
        "learn": (f"{BASE_URL}/queries.tar.gz", "queries.train.tsv", N_LEARN),
    }
    # Skip splits already on disk; otherwise stream the texts, check the count, embed and save.
    for name, (url, member, n) in parts.items():
        out = data_dir / f"{name}.npy"
        if out.exists():
            continue
        texts = _stream_tsv(url, member, n)
        if len(texts) < n:
            raise RuntimeError(f"{member}: expected {n} rows, got {len(texts)}")
        np.save(out, _embed(texts))
        print(f"{name}: {n} x 384 saved")
    # Ground truth last, since it needs base and query embeddings on disk.
    gt_path = data_dir / "gt.npy"
    if not gt_path.exists():
        base, query = np.load(data_dir / "base.npy"), np.load(data_dir / "query.npy")
        np.save(gt_path, brute_force_topk(base, query, 100).astype(np.int32))


# Dataset loader used by bench: build on first use, then return the four arrays from .npy files.
def load(data_dir: Path = DATA) -> dict[str, np.ndarray]:
    if not (data_dir / "gt.npy").exists():
        build(data_dir)
    return {k: np.load(data_dir / f"{k}.npy") for k in ("base", "query", "learn", "gt")}


# CLI: build or load the dataset and print the array shapes.
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=DATA)
    args = ap.parse_args()
    ds = load(args.dir)
    print({k: v.shape for k, v in ds.items()})


if __name__ == "__main__":
    main()
