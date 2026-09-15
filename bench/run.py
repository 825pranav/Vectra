"""Single entry point: one config in, one results JSON + one plot out.

python -m bench.run configs/bench/<experiment>.yaml [--rebuild]
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from engine.config import load_yaml

EXPERIMENTS = {
    "ann_curves": "bench.ann",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--rebuild", action="store_true", help="ignore cached indexes")
    ap.add_argument("--plot-only", action="store_true", help="redraw from saved results")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.rebuild:
        cfg["rebuild"] = True
    module = importlib.import_module(EXPERIMENTS[cfg["experiment"]])
    if args.plot_only:
        module.replot(cfg, args.config.stem)
    else:
        module.run(cfg, args.config.stem)


if __name__ == "__main__":
    main()
