"""Single entry point: one config in, one results JSON + one plot out.

python -m bench.run configs/bench/<experiment>.yaml [--rebuild]
"""

from __future__ import annotations

# CLI parsing, dynamic import of the experiment module, and the YAML config loader.
import argparse
import importlib
from pathlib import Path

from engine.config import load_yaml

# Maps the "experiment" key in a config to the module that runs it.
EXPERIMENTS = {
    "ann_curves": "bench.ann",
    "crash": "bench.crash",
    "filters": "bench.filters",
    "early_stop": "bench.early_stop",
}


# Bench entry point: read a YAML config, pick the experiment module, run it or just redraw plots.
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--rebuild", action="store_true", help="ignore cached indexes")
    ap.add_argument("--plot-only", action="store_true", help="redraw from saved results")
    args = ap.parse_args()
    # Load the config and apply the --rebuild flag on top of it.
    cfg = load_yaml(args.config)
    if args.rebuild:
        cfg["rebuild"] = True
    # Every experiment module exposes run(cfg, name); most also have replot(cfg, name).
    module = importlib.import_module(EXPERIMENTS[cfg["experiment"]])
    if args.plot_only:
        module.replot(cfg, args.config.stem)
    else:
        module.run(cfg, args.config.stem)


if __name__ == "__main__":
    main()
