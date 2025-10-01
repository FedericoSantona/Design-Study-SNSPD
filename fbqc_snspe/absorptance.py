"""Absorptance computations for SNSPD designs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable

import numpy as np

from .geometry import DeviceParams
from .materials import MaterialLibrary
from .mesh import CrossSectionMesher
from .modes_backend import EmpyModeSolver, Mode, ModeSolver


@dataclass(slots=True)
class PolarizationAbsorptance:
    wavelengths_nm: np.ndarray
    te: np.ndarray
    tm: np.ndarray

    @property
    def delta_db(self) -> np.ndarray:
        ratio = np.divide(self.te, self.tm, out=np.ones_like(self.te), where=self.tm > 0)
        return 10 * np.log10(np.clip(ratio, 1e-12, None))

    @property
    def delta_db_max(self) -> float:
        return float(np.max(np.abs(self.delta_db)))

    @property
    def mean_absorptance(self) -> float:
        return float(np.mean((self.te + self.tm) / 2))

    @property
    def worst_case_absorptance(self) -> float:
        return float(np.min(np.minimum(self.te, self.tm)))


def compute_overlap(mode: Mode, mesh, target_region: str, wavelength_nm: float) -> float:
    indices = mesh.region_indices
    intensity = np.abs(mode.ex) ** 2 + np.abs(mode.ey) ** 2 + np.abs(mode.ez) ** 2
    total_intensity = float(np.sum(intensity))
    if total_intensity == 0:
        return 0.0
    overlap_fraction = 0.0
    absorption_coeff = 0.0
    for idx, region in enumerate(mesh.regions):
        if region.layer.name != target_region:
            continue
        mask = indices == idx
        overlap_fraction += float(np.sum(intensity[mask])) / total_intensity
        n_complex = region.refractive_index_fn(wavelength_nm)
        k = np.imag(n_complex)
        wavelength_um = wavelength_nm / 1000.0
        absorption_coeff = 4 * np.pi * k / wavelength_um
    return overlap_fraction * absorption_coeff


def absorptance_single_pass(alpha: float, length_um: float) -> float:
    return float(1 - np.exp(-alpha * length_um))


def absorptance_dual_pass(alpha: float, length_um: float, reflectivity: float) -> float:
    single = absorptance_single_pass(alpha, length_um)
    remaining = np.exp(-alpha * length_um)
    dual = single + reflectivity * remaining * single
    return float(min(1.0, dual))


class AbsorptanceCalculator:
    def __init__(
        self,
        params: DeviceParams,
        material_library: MaterialLibrary | None = None,
        mode_solver: ModeSolver | None = None,
        mesh_kwargs: Dict | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.params = params
        self.material_library = material_library or MaterialLibrary()
        self.mode_solver = mode_solver or EmpyModeSolver()
        self.mesh_kwargs = mesh_kwargs or {}
        self.logger = logger or logging.getLogger(__name__)
        self._mesh = None

    @property
    def mesh(self):
        if self._mesh is None:
            cross_section = self.params.to_cross_section(self.material_library)
            mesher = CrossSectionMesher(cross_section, **self.mesh_kwargs)
            self._mesh = mesher.build()
            if self.logger.isEnabledFor(logging.INFO):
                nz, nx = self._mesh.region_indices.shape
                dx_nm = float(np.mean(np.diff(self._mesh.x_nm))) if self._mesh.x_nm.size > 1 else 0.0
                dz_nm = float(np.mean(np.diff(self._mesh.z_nm))) if self._mesh.z_nm.size > 1 else 0.0
                interior_dof = max(0, (nz - 2) * (nx - 2))
                self.logger.info(
                    "Built mesh: shape=%dx%d, dx≈%.2f nm, dz≈%.2f nm, interior DOF=%d",
                    nz,
                    nx,
                    dx_nm,
                    dz_nm,
                    interior_dof,
                )
        return self._mesh

    def sweep(
        self,
        wavelengths_nm: Iterable[float],
        nanowire_region: str = "nanowire",
    ) -> PolarizationAbsorptance:
        wavelengths = np.asarray(list(wavelengths_nm), dtype=float)
        te_abs = np.zeros_like(wavelengths)
        tm_abs = np.zeros_like(wavelengths)
        length_um = self.params.propagation_length_um
        reflectivity = 0.0
        if self.params.dual_pass.enable and self.params.dual_pass.reflectivity is not None:
            reflectivity = self.params.dual_pass.reflectivity
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                "Starting absorptance sweep: %d wavelength(s), solver=%s, propagation_length=%.2f µm",
                len(wavelengths),
                self.mode_solver.__class__.__name__,
                length_um,
            )
        for idx, wl in enumerate(wavelengths):
            solve_start = perf_counter()
            result = self.mode_solver.solve(self.mesh, wl)
            solve_elapsed = perf_counter() - solve_start
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    "Solved modes (%d/%d) at %.1f nm in %.2f s (modes=%d)",
                    idx + 1,
                    len(wavelengths),
                    wl,
                    solve_elapsed,
                    len(result.modes),
                )
            te_mode = result.polarization_mode("TE") or result.sorted_by_neff()[0]
            tm_mode = result.polarization_mode("TM") or result.sorted_by_neff()[-1]
            alpha_te = compute_overlap(te_mode, self.mesh, nanowire_region, wl)
            alpha_tm = compute_overlap(tm_mode, self.mesh, nanowire_region, wl)
            if self.params.dual_pass.enable:
                te_abs[idx] = absorptance_dual_pass(alpha_te, length_um, reflectivity)
                tm_abs[idx] = absorptance_dual_pass(alpha_tm, length_um, reflectivity)
            else:
                te_abs[idx] = absorptance_single_pass(alpha_te, length_um)
                tm_abs[idx] = absorptance_single_pass(alpha_tm, length_um)
        return PolarizationAbsorptance(wavelengths_nm=wavelengths, te=te_abs, tm=tm_abs)


__all__ = [
    "AbsorptanceCalculator",
    "PolarizationAbsorptance",
    "compute_overlap",
    "absorptance_single_pass",
    "absorptance_dual_pass",
]
