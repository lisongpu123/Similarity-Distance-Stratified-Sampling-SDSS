#!/usr/bin/env python
"""Reviewer-visible entry point for full SDSS + MLP retraining.

This script requires private de-identified Train/Test1/Test2 files specified in
its YAML configuration.  It is not needed for reconstructing the archived
manuscript table; use `build_final_sdss32_from_archived_grid.py` for that.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an SDSS + MLP experiment from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {cfg_path}. Edit configs/*.yaml to point to local de-identified data."
        )

    sys.argv = ["sdsskg.run_experiment", "--config", str(cfg_path)]
    runpy.run_module("sdsskg.run_experiment", run_name="__main__")


if __name__ == "__main__":
    main()
