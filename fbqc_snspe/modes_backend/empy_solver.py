"""Finite-difference eigensolver backend using SciPy.

The original design called for wrapping the `EMpy` package. Because EMpy is
not widely available on all platforms, this module implements a self-contained
finite-difference eigenmode solver that only depends on ``numpy`` and
``scipy``. The class keeps the historical name ``EmpyModeSolver`` so that other
modules can import it without changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, List, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as spla

from .base import Mode, ModeSolver, ModeSolverResult
from ..mesh import RectilinearMesh

NM_TO_M = 1e-9
NM_TO_UM = 1e-3
EPS0 = 8.854187817e-12
C0 = 299_792_458.0

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EmpyOptions:
    """Configuration for the finite-difference eigensolver."""

    num_modes: int = 4
    which: str = "LR"
    shift_sigma: float | None = None


class EmpyModeSolver(ModeSolver):
    """Compute TE/TM eigenmodes on a :class:`RectilinearMesh` using SciPy."""

    def __init__(self, options: EmpyOptions | None = None, target_modes: int = 2) -> None:
        super().__init__(target_modes=target_modes)
        self.options = options or EmpyOptions()
        self._logged_shapes: set[tuple[int, int]] = set()

    def solve(self, mesh: RectilinearMesh, wavelength_nm: float) -> ModeSolverResult:
        nz, nx = mesh.region_indices.shape
        interior_dof = max(0, (nz - 2) * (nx - 2))
        shape_key = (nz, nx)
        if LOGGER.isEnabledFor(logging.INFO) and shape_key not in self._logged_shapes:
            dx_nm = float(np.mean(np.diff(mesh.x_nm))) if mesh.x_nm.size > 1 else 0.0
            dz_nm = float(np.mean(np.diff(mesh.z_nm))) if mesh.z_nm.size > 1 else 0.0
            LOGGER.info(
                "EmpyModeSolver mesh: shape=%dx%d, interior DOF=%d, dx≈%.2f nm, dz≈%.2f nm, options=%s",
                nz,
                nx,
                interior_dof,
                dx_nm,
                dz_nm,
                self.options,
            )
            self._logged_shapes.add(shape_key)

        te_start = perf_counter()
        te_modes = _solve_te_modes(mesh, wavelength_nm, self.options)
        te_elapsed = perf_counter() - te_start
        if LOGGER.isEnabledFor(logging.INFO):
            LOGGER.info(
                "EmpyModeSolver: TE solve produced %d mode(s) in %.2f s (λ=%.1f nm)",
                len(te_modes),
                te_elapsed,
                wavelength_nm,
            )

        tm_start = perf_counter()
        tm_modes = _solve_tm_modes(mesh, wavelength_nm, self.options)
        tm_elapsed = perf_counter() - tm_start
        if LOGGER.isEnabledFor(logging.INFO):
            LOGGER.info(
                "EmpyModeSolver: TM solve produced %d mode(s) in %.2f s (λ=%.1f nm)",
                len(tm_modes),
                tm_elapsed,
                wavelength_nm,
            )

        # Interleave TE/TM fundamentals so downstream code can grab them easily.
        modes: List[Mode] = []
        if te_modes:
            modes.append(te_modes[0])
        if tm_modes:
            modes.append(tm_modes[0])

        # Add higher-order solutions if requested.
        remaining = self.target_modes - len(modes)
        if remaining > 0:
            extra: Iterable[Mode] = list(te_modes[1:] + tm_modes[1:])  # type: ignore[arg-type]
            for mode in extra:
                modes.append(mode)
                if len(modes) >= self.target_modes:
                    break

        return ModeSolverResult(modes=modes or te_modes or tm_modes)


def _solve_te_modes(mesh: RectilinearMesh, wavelength_nm: float, options: EmpyOptions) -> List[Mode]:
    """Solve the scalar TE eigenproblem."""

    x_nm = np.asarray(mesh.x_nm, dtype=float)
    z_nm = np.asarray(mesh.z_nm, dtype=float)
    if x_nm.size < 3 or z_nm.size < 3:
        raise RuntimeError("Mesh must contain at least three points along each axis for TE solving.")

    dx_nm = float(np.mean(np.diff(x_nm)))
    dz_nm = float(np.mean(np.diff(z_nm)))
    if not np.allclose(np.diff(x_nm), dx_nm):
        raise RuntimeError("TE solver currently requires a uniform x-grid.")
    if not np.allclose(np.diff(z_nm), dz_nm):
        raise RuntimeError("TE solver currently requires a uniform z-grid.")

    nz, nx = mesh.region_indices.shape
    nz_i = nz - 2
    nx_i = nx - 2
    if nz_i <= 0 or nx_i <= 0:
        raise RuntimeError("Mesh is too small for finite-difference interior points.")

    eps_map = mesh.epsilon(wavelength_nm)
    eps_inner = eps_map[1:-1, 1:-1].reshape(-1)

    dx2 = dx_nm**2
    dz2 = dz_nm**2

    lx = sparse.diags(
        diagonals=[np.ones(nx_i - 1), -2 * np.ones(nx_i), np.ones(nx_i - 1)],
        offsets=[-1, 0, 1],
        dtype=np.complex128,
    ) / dx2
    lz = sparse.diags(
        diagonals=[np.ones(nz_i - 1), -2 * np.ones(nz_i), np.ones(nz_i - 1)],
        offsets=[-1, 0, 1],
        dtype=np.complex128,
    ) / dz2

    lap = sparse.kron(sparse.identity(nz_i, dtype=np.complex128), lx) + sparse.kron(lz, sparse.identity(nx_i, dtype=np.complex128))

    k0 = 2.0 * np.pi / wavelength_nm
    operator = -lap + sparse.diags((k0**2) * eps_inner, dtype=np.complex128)

    num = min(options.num_modes, operator.shape[0] - 1)
    if num <= 0:
        num = 1

    vals, vecs = spla.eigs(operator, k=num, which=options.which, sigma=options.shift_sigma)
    beta_vals = _sqrt_eigenvalues(vals)

    modes: List[Mode] = []
    for beta, vec in _sort_by_neff(beta_vals, vecs, k0):
        field = np.zeros((nz, nx), dtype=np.complex128)
        field[1:-1, 1:-1] = vec.reshape(nz_i, nx_i)
        field = _normalize_field(field, dx_nm, dz_nm)
        modes.append(
            Mode(
                wavelength_nm=wavelength_nm,
                neff=beta / k0,
                polarization="TE",
                ex=np.zeros_like(field),
                ey=field,
                ez=np.zeros_like(field),
                attenuation_neper_per_um=float(beta.imag * 1e3),
            )
        )
    return modes


def _solve_tm_modes(mesh: RectilinearMesh, wavelength_nm: float, options: EmpyOptions) -> List[Mode]:
    """Solve the scalar TM eigenproblem via a generalized eigenvalue formulation."""

    x_nm = np.asarray(mesh.x_nm, dtype=float)
    z_nm = np.asarray(mesh.z_nm, dtype=float)
    if x_nm.size < 3 or z_nm.size < 3:
        raise RuntimeError("Mesh must contain at least three points along each axis for TM solving.")

    dx_nm = float(np.mean(np.diff(x_nm)))
    dz_nm = float(np.mean(np.diff(z_nm)))
    if not np.allclose(np.diff(x_nm), dx_nm):
        raise RuntimeError("TM solver currently requires a uniform x-grid.")
    if not np.allclose(np.diff(z_nm), dz_nm):
        raise RuntimeError("TM solver currently requires a uniform z-grid.")

    nz, nx = mesh.region_indices.shape
    nz_i = nz - 2
    nx_i = nx - 2
    if nz_i <= 0 or nx_i <= 0:
        return []

    eps_map = mesh.epsilon(wavelength_nm)
    inv_eps = np.divide(1.0, eps_map, out=np.zeros_like(eps_map), where=eps_map != 0)
    inv_eps_inner = inv_eps[1:-1, 1:-1]

    dx2 = dx_nm**2
    dz2 = dz_nm**2

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    def linear_index(iz: int, ix: int) -> int:
        return iz * nx_i + ix

    for iz in range(nz_i):
        for ix in range(nx_i):
            zi = iz + 1
            xi = ix + 1
            center_idx = linear_index(iz, ix)
            inv_center = inv_eps[zi, xi]
            right = 0.5 * (inv_eps[zi, xi] + inv_eps[zi, xi + 1])
            left = 0.5 * (inv_eps[zi, xi] + inv_eps[zi, xi - 1])
            up = 0.5 * (inv_eps[zi, xi] + inv_eps[zi + 1, xi])
            down = 0.5 * (inv_eps[zi, xi] + inv_eps[zi - 1, xi])

            diag_val = -(right + left) / dx2 - (up + down) / dz2 + (2.0 * np.pi / wavelength_nm) ** 2
            rows.append(center_idx)
            cols.append(center_idx)
            data.append(diag_val)

            if ix + 1 < nx_i:
                neighbor = linear_index(iz, ix + 1)
                val = right / dx2
                rows.append(center_idx)
                cols.append(neighbor)
                data.append(val)
            if ix - 1 >= 0:
                neighbor = linear_index(iz, ix - 1)
                val = left / dx2
                rows.append(center_idx)
                cols.append(neighbor)
                data.append(val)
            if iz + 1 < nz_i:
                neighbor = linear_index(iz + 1, ix)
                val = up / dz2
                rows.append(center_idx)
                cols.append(neighbor)
                data.append(val)
            if iz - 1 >= 0:
                neighbor = linear_index(iz - 1, ix)
                val = down / dz2
                rows.append(center_idx)
                cols.append(neighbor)
                data.append(val)

    size = nx_i * nz_i
    operator = sparse.coo_matrix((data, (rows, cols)), shape=(size, size), dtype=np.complex128).tocsr()
    mass = sparse.diags(inv_eps_inner.reshape(-1), dtype=np.complex128)

    num = min(options.num_modes, size - 1)
    if num <= 0:
        num = 1

    vals, vecs = spla.eigs(operator, M=mass, k=num, which=options.which, sigma=options.shift_sigma)
    beta_vals = _sqrt_eigenvalues(vals)
    k0 = 2.0 * np.pi / wavelength_nm

    modes: List[Mode] = []
    for beta, vec in _sort_by_neff(beta_vals, vecs, k0):
        hy_field = np.zeros((nz, nx), dtype=np.complex128)
        hy_field[1:-1, 1:-1] = vec.reshape(nz_i, nx_i)
        hy_field = _normalize_field(hy_field, dx_nm, dz_nm)
        ex_field, ez_field = _tm_electric_fields_from_hy(hy_field, eps_map, beta, dx_nm, dz_nm, wavelength_nm)
        modes.append(
            Mode(
                wavelength_nm=wavelength_nm,
                neff=beta / k0,
                polarization="TM",
                ex=ex_field,
                ey=np.zeros_like(ex_field),
                ez=ez_field,
                attenuation_neper_per_um=float(beta.imag * 1e3),
            )
        )
    return modes


def _tm_electric_fields_from_hy(
    hy_field: NDArray[np.complex128],
    eps_map: NDArray[np.complex128],
    beta: complex,
    dx_nm: float,
    dz_nm: float,
    wavelength_nm: float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    dx_m = dx_nm * NM_TO_M
    dz_m = dz_nm * NM_TO_M
    dhy_dx = np.zeros_like(hy_field, dtype=np.complex128)
    dhy_dz = np.zeros_like(hy_field, dtype=np.complex128)
    dhy_dx[:, 1:-1] = (hy_field[:, 2:] - hy_field[:, :-2]) / (2.0 * dx_m)
    dhy_dz[1:-1, :] = (hy_field[2:, :] - hy_field[:-2, :]) / (2.0 * dz_m)

    wavelength_m = wavelength_nm * NM_TO_M
    omega = 2.0 * np.pi * C0 / wavelength_m
    epsilon = EPS0 * eps_map
    denom = omega * epsilon

    ex = np.zeros_like(hy_field, dtype=np.complex128)
    ez = np.zeros_like(hy_field, dtype=np.complex128)
    np.divide(1j * dhy_dz, denom, out=ex, where=denom != 0)
    np.divide(-1j * dhy_dx, denom, out=ez, where=denom != 0)

    return ex, ez


def _normalize_field(field: NDArray[np.complex128], dx_nm: float, dz_nm: float) -> NDArray[np.complex128]:
    weight = dx_nm * dz_nm
    norm = np.sqrt(np.sum(np.abs(field) ** 2) * weight)
    if norm == 0:
        return field
    return field / norm


def _sqrt_eigenvalues(vals: NDArray[np.complex128]) -> NDArray[np.complex128]:
    betas = np.lib.scimath.sqrt(vals)
    adjusted = []
    for beta in betas:
        if beta.real < 0:
            beta = -beta
        if beta.imag < 0:
            beta = beta.conjugate()
        adjusted.append(beta)
    return np.asarray(adjusted, dtype=np.complex128)


def _sort_by_neff(
    betas: NDArray[np.complex128],
    vecs: NDArray[np.complex128],
    k0: float,
) -> Sequence[tuple[complex, NDArray[np.complex128]]]:
    indices = np.argsort(-np.real(betas / k0))
    return [(betas[idx], vecs[:, idx]) for idx in indices]


__all__ = ["EmpyModeSolver", "EmpyOptions"]
