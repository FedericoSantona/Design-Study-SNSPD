"""Objective functions and metrics for SNSPD design."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .absorptance import PolarizationAbsorptance


@dataclass(slots=True)
class BandMetrics:
    delta_db_max: float
    mean_absorptance: float
    worst_case_absorptance: float
    band_ok: bool


def metrics(absorptance: PolarizationAbsorptance, mean_threshold: float = 0.85, delta_db_limit: float = 1.0) -> BandMetrics:
    delta_db_max = absorptance.delta_db_max
    mean_abs = absorptance.mean_absorptance
    worst_case = absorptance.worst_case_absorptance
    band_ok = delta_db_max <= delta_db_limit and mean_abs >= mean_threshold
    return BandMetrics(
        delta_db_max=delta_db_max,
        mean_absorptance=mean_abs,
        worst_case_absorptance=worst_case,
        band_ok=band_ok,
    )


def objective(absorptance: PolarizationAbsorptance, mean_threshold: float = 0.85, delta_db_limit: float = 1.0, penalty: float = 10.0) -> float:
    m = metrics(absorptance, mean_threshold=mean_threshold, delta_db_limit=delta_db_limit)
    obj = m.delta_db_max
    if m.mean_absorptance < mean_threshold:
        obj += penalty * (mean_threshold - m.mean_absorptance)
    return obj


def pdde_proxy(absorptance: PolarizationAbsorptance) -> float:
    eta_max = np.max((absorptance.te + absorptance.tm) / 2)
    eta_min = np.min((absorptance.te + absorptance.tm) / 2)
    if eta_min == 0:
        return float("inf")
    return 10 * np.log10(eta_max / eta_min)


__all__ = ["metrics", "objective", "pdde_proxy", "BandMetrics"]
