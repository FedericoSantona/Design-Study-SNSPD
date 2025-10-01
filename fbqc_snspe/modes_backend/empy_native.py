"""Wrapper around EMpy's finite-difference eigenmode solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .base import Mode, ModeSolver, ModeSolverResult
from ..mesh import RectilinearMesh

try:  # pragma: no cover - optional dependency
    from EMpy.modesolvers import FD as em_fd
except Exception:  # pragma: no cover - graceful failure when EMpy missing
    em_fd = None

NM_TO_UM = 1e-3


class EmpyNativeError(RuntimeError):
    """Raised when the EMpy backend encounters an error."""


@dataclass(slots=True)
class EmpyNativeOptions:
    """Configuration knobs for the native EMpy solver."""

    num_modes: int = 4
    boundary: str = "0000"


class EmpyNativeModeSolver(ModeSolver):
    """Mode solver backed by EMpy's vector finite-difference engine."""

    def __init__(self, options: EmpyNativeOptions | None = None, target_modes: int = 2) -> None:
        if em_fd is None:
            raise EmpyNativeError(
                "EMpy is not installed. Install the 'EMpy' package to use the native backend."
            )
        super().__init__(target_modes=target_modes)
        self.options = options or EmpyNativeOptions()

    def solve(self, mesh: RectilinearMesh, wavelength_nm: float) -> ModeSolverResult:
        solver = self._build_solver(mesh, wavelength_nm)
        try:
            solver.solve(neigs=self.options.num_modes, tol=0)
        except Exception as exc:  # pragma: no cover - propagate informative error
            raise EmpyNativeError(f"EMpy solve failed: {exc}") from exc

        em_modes = getattr(solver, "modes", None)
        if not em_modes:
            raise EmpyNativeError("EMpy returned no modes for the supplied geometry.")

        x_um = np.asarray(mesh.x_nm, dtype=float) * NM_TO_UM
        z_um = np.asarray(mesh.z_nm, dtype=float) * NM_TO_UM

        modes: List[Mode] = []
        for em_mode in em_modes[: self.target_modes]:
            modes.append(self._convert_mode(mesh, wavelength_nm, em_mode, x_um, z_um))
        return ModeSolverResult(modes=modes)

    def _build_solver(self, mesh: RectilinearMesh, wavelength_nm: float):
        x_nm = np.asarray(mesh.x_nm, dtype=float)
        z_nm = np.asarray(mesh.z_nm, dtype=float)
        eps_nodes = np.asarray(mesh.epsilon(wavelength_nm), dtype=np.complex128)

        if x_nm.size < 2 or z_nm.size < 2:
            raise EmpyNativeError("Mesh must contain at least two samples along each axis.")

        x_centers_nm = 0.5 * (x_nm[:-1] + x_nm[1:])
        z_centers_nm = 0.5 * (z_nm[:-1] + z_nm[1:])
        eps_cells = 0.25 * (
            eps_nodes[:-1, :-1]
            + eps_nodes[1:, :-1]
            + eps_nodes[:-1, 1:]
            + eps_nodes[1:, 1:]
        )
        eps_cells = eps_cells.T  # -> (len(x_centers), len(z_centers))

        x_centers_um = x_centers_nm * NM_TO_UM
        z_centers_um = z_centers_nm * NM_TO_UM

        def nearest_indices_nd(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
            grid = np.asarray(grid, dtype=float)
            values = np.asarray(values, dtype=float)
            diff = np.abs(values[..., None] - grid)
            return diff.argmin(axis=-1)

        def nearest_indices_1d(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
            grid = np.asarray(grid, dtype=float)
            values = np.asarray(values, dtype=float)
            if values.ndim == 0:
                values = values.reshape(1)
            diff = np.abs(values[:, None] - grid)
            return diff.argmin(axis=1)

        def eps_func(xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
            xc_arr = np.asarray(xc, dtype=float)
            yc_arr = np.asarray(yc, dtype=float)
            if xc_arr.ndim <= 1 and yc_arr.ndim <= 1:
                ix = nearest_indices_1d(x_centers_um, xc_arr)
                iz = nearest_indices_1d(z_centers_um, yc_arr)
                return eps_cells[np.ix_(ix, iz)]
            ix = nearest_indices_nd(x_centers_um, xc_arr)
            iz = nearest_indices_nd(z_centers_um, yc_arr)
            return eps_cells[ix, iz]

        return em_fd.VFDModeSolver(
            wavelength_nm * NM_TO_UM,
            x_nm * NM_TO_UM,
            z_nm * NM_TO_UM,
            eps_func,
            boundary=self.options.boundary,
        )

    def _convert_mode(
        self,
        mesh: RectilinearMesh,
        wavelength_nm: float,
        em_mode,
        x_um: np.ndarray,
        z_um: np.ndarray,
    ) -> Mode:
        def field(name: str) -> np.ndarray:
            data = np.asarray(em_mode.get_field(name, x_um, z_um), dtype=np.complex128)
            return data.T  # EMpy returns (len(x), len(z))

        ex = field("Ex")
        ey = field("Ey")
        ez = field("Ez")
        hx = field("Hx")
        hy = field("Hy")
        hz = field("Hz")

        if ex.shape != mesh.region_indices.shape:
            raise EmpyNativeError(
                f"EMpy field shape {ex.shape} does not match mesh shape {mesh.region_indices.shape}."
            )

        energy_y = np.sum(np.abs(ey))
        energy_x = np.sum(np.abs(ex))
        polarization = "TE" if energy_y >= energy_x else "TM"

        beta = (2 * np.pi * em_mode.neff) / (wavelength_nm * NM_TO_UM)

        return Mode(
            wavelength_nm=wavelength_nm,
            neff=em_mode.neff,
            polarization=polarization,
            ex=ex,
            ey=ey,
            ez=ez,
            hx=hx,
            hy=hy,
            hz=hz,
            attenuation_neper_per_um=float(beta.imag),
        )


__all__ = ["EmpyNativeModeSolver", "EmpyNativeOptions", "EmpyNativeError"]
