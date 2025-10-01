"""Fabrication tolerance analysis via Monte Carlo sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .absorptance import AbsorptanceCalculator
from .geometry import DeviceParams
from .materials import MaterialLibrary
from .modes_backend import ModeSolver
from .objectives import metrics
from .sweep import apply_updates


@dataclass(slots=True)
class Perturbation:
    path: str
    sigma: float
    distribution: str = "normal"


@dataclass(slots=True)
class ToleranceConfig:
    wavelengths_nm: Sequence[float]
    perturbations: Sequence[Perturbation]
    num_samples: int = 100
    mean_threshold: float = 0.85
    delta_db_limit: float = 1.0
    random_seed: int | None = None


def run_monte_carlo(
    base_params: DeviceParams,
    material_library: MaterialLibrary,
    config: ToleranceConfig,
    mode_solver: ModeSolver | None = None,
    mesh_kwargs: dict | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    rows = []
    for idx in range(config.num_samples):
        updates = {}
        for perturb in config.perturbations:
            if perturb.distribution == "normal":
                delta = rng.normal(0.0, perturb.sigma)
            elif perturb.distribution == "uniform":
                delta = rng.uniform(-perturb.sigma, perturb.sigma)
            else:
                raise ValueError(f"Unsupported distribution {perturb.distribution}")
            updates[perturb.path] = delta
        perturbed_params = apply_relative_updates(base_params, updates)
        calc = AbsorptanceCalculator(
            params=perturbed_params,
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
                "sample": idx,
                "delta_db_max": band_metrics.delta_db_max,
                "mean_absorptance": band_metrics.mean_absorptance,
                "worst_absorptance": band_metrics.worst_case_absorptance,
                "band_ok": band_metrics.band_ok,
            }
        )
    return pd.DataFrame(rows)


def apply_relative_updates(params: DeviceParams, deltas: dict[str, float]) -> DeviceParams:
    base = params.dict()
    for path, delta in deltas.items():
        keys = path.split(".")
        target = base
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] += delta
    return DeviceParams.parse_obj(base)


__all__ = ["run_monte_carlo", "ToleranceConfig", "Perturbation"]
