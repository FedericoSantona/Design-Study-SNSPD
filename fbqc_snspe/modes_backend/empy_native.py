"""Wrapper around the original EMpy eigenmode solver.

This backend provides an interface that mirrors :class:`EmpyModeSolver` but
relies on the optional `EMpy` package. Users can enable it via configuration
(`backend: empy_native`) provided that EMpy is installed in their Python
environment. If EMpy is missing, a descriptive :class:`RuntimeError` is raised
at instantiation time.

The implementation translates the :class:`RectilinearMesh` produced by the
mesher into the data structures expected by ``EMpy.modesolvers.wgmsolver`` and
returns :class:`Mode` instances compatible with the rest of the toolkit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np

from .base import Mode, ModeSolver, ModeSolverResult
from ..mesh import RectilinearMesh

try:  # pragma: no cover - optional dependency
    from EMpy.modesolvers import wgmsolver
except Exception:  # pragma: no cover - keep broad to surface useful message later
    wgmsolver = None

NM_TO_UM = 1e-3


class EmpyNativeError(RuntimeError):
    """Raised when the EMpy backend encounters an error."""


@dataclass(slots=True)
class EmpyNativeOptions:
    """Configuration knobs for the native EMpy solver."""

    num_modes: int = 4
    boundary: str = "0000"
    log_progress: bool = False


class EmpyNativeModeSolver(ModeSolver):
    """Mode solver backed by the EMpy ``WGMSolver`` implementation."""

    def __init__(self, options: EmpyNativeOptions | None = None, target_modes: int = 2) -> None:
        if wgmsolver is None:
            raise EmpyNativeError(
                "EMpy is not installed. Install the 'EMpy' package to use the native backend."
            )
        super().__init__(target_modes=target_modes)
        self.options = options or EmpyNativeOptions()

    def solve(self, mesh: RectilinearMesh, wavelength_nm: float) -> ModeSolverResult:
        try:
            solver = self._build_solver(mesh, wavelength_nm)
            solver.solve(wavelength_nm * NM_TO_UM, self.options.num_modes)
            neffs = _collect_neffs(solver)
        except Exception as exc:  # pragma: no cover - propagate informative error
            raise EmpyNativeError(f"EMpy solve failed: {exc}") from exc

        modes: List[Mode] = []
        max_modes = min(self.target_modes, len(neffs))
        for idx in range(max_modes):
            fields = solver.get_field(idx)
            modes.append(self._mode_from_field(mesh, wavelength_nm, neffs[idx], fields))
        return ModeSolverResult(modes=modes)

    def _build_solver(self, mesh: RectilinearMesh, wavelength_nm: float):
        x_um = np.asarray(mesh.x_nm, dtype=float) * NM_TO_UM
        z_um = np.asarray(mesh.z_nm, dtype=float) * NM_TO_UM
        eps_map = np.asarray(mesh.epsilon(wavelength_nm), dtype=np.complex128)

        def eps_func(_: np.ndarray, __: np.ndarray) -> np.ndarray:
            return eps_map

        boundary = getattr(wgmsolver, self.options.boundary, self.options.boundary)
        return wgmsolver.WGMSolver(x_um, z_um, eps_func, boundary=boundary)

    def _mode_from_field(self, mesh: RectilinearMesh, wavelength_nm: float, neff: complex, field) -> Mode:
        zeros = lambda: np.zeros(mesh.region_indices.shape, dtype=np.complex128)
        ex = np.asarray(getattr(field, "Ex", zeros()), dtype=np.complex128)
        ey = np.asarray(getattr(field, "Ey", zeros()), dtype=np.complex128)
        ez = np.asarray(getattr(field, "Ez", zeros()), dtype=np.complex128)
        hx = np.asarray(getattr(field, "Hx", zeros()), dtype=np.complex128)
        hy = np.asarray(getattr(field, "Hy", zeros()), dtype=np.complex128)
        hz = np.asarray(getattr(field, "Hz", zeros()), dtype=np.complex128)

        polarization = "TE" if np.sum(np.abs(ey)) >= np.sum(np.abs(ex)) else "TM"
        return Mode(
            wavelength_nm=wavelength_nm,
            neff=neff,
            polarization=polarization,
            ex=ex,
            ey=ey,
            ez=ez,
            hx=hx,
            hy=hy,
            hz=hz,
            attenuation_neper_per_um=None,
        )


def _collect_neffs(solver) -> List[complex]:
    if hasattr(solver, "get_neff"):
        return list(solver.get_neff())
    if hasattr(solver, "n_eff"):
        return list(np.atleast_1d(solver.n_eff))
    raise EmpyNativeError("Could not retrieve effective indices from EMpy solver result.")


__all__ = ["EmpyNativeModeSolver", "EmpyNativeOptions", "EmpyNativeError"]
