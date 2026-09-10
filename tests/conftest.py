import sys
from pathlib import Path

import numpy as np
import pytest

# Tests import the engine as top-level packages (index.*, storage.*, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def small_data(rng) -> np.ndarray:
    """2k x 32 float32, clustered so nearest-neighbour structure is non-trivial."""
    centers = rng.normal(size=(20, 32)).astype(np.float32) * 5
    assign = rng.integers(0, 20, size=2000)
    return (centers[assign] + rng.normal(size=(2000, 32)).astype(np.float32)).astype(np.float32)
