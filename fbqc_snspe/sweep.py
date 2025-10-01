"""Parameter sweeps and optimisation scaffolding."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd
from tqdm.auto import tqdm

from .absorptance import AbsorptanceCalculator
from .geometry import DeviceParams
from .materials import MaterialLibrary
from .modes_backend import ModeSolver
from .objectives import BandMetrics, metrics, objective, pdde_proxy


@dataclass(slots=True)
class SweepConfig:
    wavelengths_nm: Sequence[float]
    sweep_space: Mapping[str, Sequence[float]]
    mean_threshold: float = 0.85
    delta_db_limit: float = 1.0


def apply_updates(params: DeviceParams, updates: Mapping[str, float]) -> DeviceParams:
    data = params.dict()
    for path, value in updates.items():
        keys = path.split(".")
        target = data
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return DeviceParams.parse_obj(data)


def run_grid_sweep(
    base_params: DeviceParams,
    material_library: MaterialLibrary,
    config: SweepConfig,
    mode_solver: ModeSolver | None = None,
    mesh_kwargs: Dict | None = None,
    show_progress: bool = True,
    verbose: bool = False,
    log_every: int = 1,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    keys = list(config.sweep_space.keys())
    values_lists = [config.sweep_space[key] for key in keys]
    rows = []
    iterator = product(*values_lists)
    total = 1
    for values in values_lists:
        total *= len(values)
    if show_progress:
        iterator = tqdm(iterator, total=total, desc="Grid sweep")
    for idx, combination in enumerate(iterator, start=1):
        updates = dict(zip(keys, combination))
        if verbose and (idx % log_every == 0 or log_every <= 1):
            logger.info("[%d/%d] Evaluating sweep point %s", idx, total, updates)
        params = apply_updates(base_params, updates)
        calc = AbsorptanceCalculator(
            params=params,
            material_library=material_library,
            mode_solver=mode_solver,
            mesh_kwargs=mesh_kwargs,
        )
        absorptance = calc.sweep(config.wavelengths_nm)
        band_metrics = metrics(
            absorptance,
            mean_threshold=config.mean_threshold,
            delta_db_limit=config.delta_db_limit,
        )
        rows.append(
            {
                **updates,
                "delta_db_max": band_metrics.delta_db_max,
                "mean_absorptance": band_metrics.mean_absorptance,
                "worst_absorptance": band_metrics.worst_case_absorptance,
                "band_ok": band_metrics.band_ok,
                "objective": objective(
                    absorptance,
                    mean_threshold=config.mean_threshold,
                    delta_db_limit=config.delta_db_limit,
                ),
                "pdde_proxy_db": pdde_proxy(absorptance),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["run_grid_sweep", "apply_updates", "SweepConfig"]
