"""Abstract interfaces for mode solver backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from ..mesh import RectilinearMesh


@dataclass(slots=True)
class Mode:
    wavelength_nm: float
    neff: complex
    polarization: str
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    hx: np.ndarray | None = None
    hy: np.ndarray | None = None
    hz: np.ndarray | None = None
    attenuation_neper_per_um: float | None = None

    def loss_db_per_cm(self) -> float | None:
        if self.attenuation_neper_per_um is None:
            return None
        # convert from nepers per micron to dB per centimetre
        return 8.686 * self.attenuation_neper_per_um * 1e4


@dataclass(slots=True)
class ModeSolverResult:
    modes: List[Mode]

    def sorted_by_neff(self, descending: bool = True) -> Sequence[Mode]:
        return tuple(sorted(self.modes, key=lambda m: m.neff.real, reverse=descending))

    def polarization_mode(self, polarization: str) -> Mode | None:
        for mode in self.modes:
            if mode.polarization.lower().startswith(polarization.lower()):
                return mode
        return None


class ModeSolver(ABC):
    """Abstract base class for transverse mode solvers."""

    def __init__(self, target_modes: int = 2) -> None:
        self.target_modes = target_modes

    @abstractmethod
    def solve(self, mesh: RectilinearMesh, wavelength_nm: float) -> ModeSolverResult:
        """Compute eigenmodes for the supplied cross-section."""
        raise NotImplementedError


__all__ = ["Mode", "ModeSolver", "ModeSolverResult"]
