#!/usr/bin/env python3
"""CLI entry point to run SNSPD design sweeps."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from fbqc_snspe.geometry import DeviceParams
from fbqc_snspe.materials import MaterialLibrary
from fbqc_snspe.modes_backend import create_mode_solver
from fbqc_snspe.sweep import SweepConfig, run_grid_sweep


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a polarization-flattened SNSPD sweep")
    parser.add_argument("config", type=Path, help="YAML configuration file")
    parser.add_argument("--output", type=Path, default=Path("outputs/sweep.csv"), help="CSV output path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging of sweep points")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_params_data = cfg["base_params"]
    if hasattr(DeviceParams, "model_validate"):
        base_params = DeviceParams.model_validate(base_params_data)  # Pydantic v2+
    else:  # pragma: no cover - fallback for Pydantic v1
        base_params = DeviceParams.parse_obj(base_params_data)
    sweep_cfg = SweepConfig(
        wavelengths_nm=cfg["sweep"]["wavelengths_nm"],
        sweep_space=cfg["sweep"]["sweep_space"],
        mean_threshold=cfg["sweep"].get("mean_threshold", 0.85),
        delta_db_limit=cfg["sweep"].get("delta_db_limit", 1.0),
    )

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("fbqc_snspe.sweep")
    material_library = MaterialLibrary()
    mode_solver = create_mode_solver(cfg.get("solver"))
    mesh_kwargs = cfg.get("mesh", {})
    df = run_grid_sweep(
        base_params=base_params,
        material_library=material_library,
        config=sweep_cfg,
        mode_solver=mode_solver,
        mesh_kwargs=mesh_kwargs,
        verbose=args.verbose,
        logger=logger,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved sweep results to {args.output}")


if __name__ == "__main__":
    main()
