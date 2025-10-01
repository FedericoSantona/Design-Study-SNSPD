"""Semi-analytic mode solver providing a lightweight fallback backend."""

from __future__ import annotations

import numpy as np

from .base import Mode, ModeSolver, ModeSolverResult


class AnalyticSlabModeSolver(ModeSolver):
    """Crude effective-index approximation for quick prototyping.

    This backend does not replace full-vector solvers (MPB/EMpy), but offers a
    fast way to evaluate design trends when those heavy dependencies are not
    available.
    """

    def __init__(self, lateral_decay_nm: float = 400.0, vertical_decay_nm: float = 200.0, target_modes: int = 2) -> None:
        super().__init__(target_modes=target_modes)
        self.lateral_decay_nm = lateral_decay_nm
        self.vertical_decay_nm = vertical_decay_nm

    def solve(self, mesh, wavelength_nm: float) -> ModeSolverResult:
        dx_nm = float(np.mean(np.diff(mesh.x_nm))) if mesh.x_nm.size > 1 else 1.0
        dz_nm = float(np.mean(np.diff(mesh.z_nm))) if mesh.z_nm.size > 1 else 1.0
        area_weights = np.ones_like(mesh.region_indices, dtype=float) * dx_nm * dz_nm
        eps = mesh.epsilon(wavelength_nm)
        n_map = np.sqrt(np.real(eps))
        n_inv_map = np.zeros_like(n_map)
        mask = n_map > 0
        n_inv_map[mask] = 1.0 / n_map[mask] ** 2
        te_weight = area_weights
        tm_weight = area_weights * n_inv_map
        n_eff_te = np.sqrt(np.sum(n_map**2 * te_weight) / np.sum(te_weight))
        n_eff_tm = np.sqrt(1.0 / np.sum(tm_weight) * np.sum(te_weight))

        X, Z = np.meshgrid(mesh.x_nm, mesh.z_nm)
        z_center_nm = np.average(Z, weights=np.real(eps))
        te_field = self._gaussian_field(X, Z, z_center_nm)
        tm_field = self._gaussian_field(X, Z, z_center_nm, orientation="y")

        te_mode = Mode(
            wavelength_nm=wavelength_nm,
            neff=n_eff_te,
            polarization="TE",
            ex=te_field,
            ey=np.zeros_like(te_field),
            ez=0.1 * te_field,
            attenuation_neper_per_um=None,
        )
        tm_mode = Mode(
            wavelength_nm=wavelength_nm,
            neff=n_eff_tm,
            polarization="TM",
            ex=np.zeros_like(tm_field),
            ey=tm_field,
            ez=0.1 * tm_field,
            attenuation_neper_per_um=None,
        )
        return ModeSolverResult(modes=[te_mode, tm_mode])

    def _gaussian_field(self, X: np.ndarray, Z: np.ndarray, z_center_nm: float, orientation: str = "x") -> np.ndarray:
        wx = self.lateral_decay_nm
        wz = self.vertical_decay_nm
        field = np.exp(-((X / wx) ** 2 + ((Z - z_center_nm) / wz) ** 2))
        field /= np.max(field) if np.max(field) else 1.0
        return field


__all__ = ["AnalyticSlabModeSolver"]
