"""Material dispersion handling for fbqc_snspe.

This module provides utilities for loading tabulated complex refractive index
spectra (n + ik) and interpolating them across the wavelength range of
interest. Data files are expected to live under ``data/materials`` as CSV files
with the columns ``wavelength_nm``, ``n`` and ``k``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping

import numpy as np
import pandas as pd

DATA_SUBDIR = Path(__file__).resolve().parent.parent / "data" / "materials"


class DispersionError(RuntimeError):
    """Raised when dispersion data are missing or invalid."""


@dataclass(slots=True)
class DispersionTable:
    """Tabulated dispersion data for a single material."""

    wavelengths_nm: np.ndarray
    n: np.ndarray
    k: np.ndarray

    def __post_init__(self) -> None:
        if np.any(np.diff(self.wavelengths_nm) <= 0):
            raise DispersionError("Wavelength grid must be strictly increasing.")
        if self.wavelengths_nm.ndim != 1:
            raise DispersionError("Wavelength grid must be one-dimensional.")
        if not (self.n.shape == self.k.shape == self.wavelengths_nm.shape):
            raise DispersionError("n and k arrays must match wavelength array shape.")

    def interpolator(self) -> Callable[[float | np.ndarray], np.ndarray]:
        """Return an interpolator for complex refractive index."""

        wavelengths = self.wavelengths_nm
        n_real = self.n
        k_imag = self.k

        def _interp(values_nm: float | np.ndarray) -> np.ndarray:
            values = np.atleast_1d(values_nm).astype(float)
            if np.any((values < wavelengths[0]) | (values > wavelengths[-1])):
                raise DispersionError(
                    "Requested wavelength outside tabulated range."
                )
            n_val = np.interp(values, wavelengths, n_real)
            k_val = np.interp(values, wavelengths, k_imag)
            complex_n = n_val + 1j * k_val
            return complex_n if np.ndim(values_nm) else complex_n.item()

        return _interp


def _resolve_material_path(material: str, data_dir: Path | None = None) -> Path:
    base_dir = Path(data_dir) if data_dir is not None else DATA_SUBDIR
    candidate = base_dir / f"{material}.csv"
    if not candidate.exists():
        existing = sorted(p.name for p in base_dir.glob("*.csv"))
        raise DispersionError(
            f"No dispersion data for material '{material}'. Available: {existing}"
        )
    return candidate


@lru_cache(maxsize=64)
def load_dispersion(material: str, data_dir: Path | None = None) -> Callable[[float], complex]:
    """Load complex refractive index interpolator for a material."""

    path = _resolve_material_path(material, data_dir)
    frame = pd.read_csv(path)
    required = {"wavelength_nm", "n", "k"}
    if not required.issubset(frame.columns):
        raise DispersionError(
            f"Columns {required} required in {path}, found {set(frame.columns)}"
        )
    table = DispersionTable(
        wavelengths_nm=frame["wavelength_nm"].to_numpy(dtype=float),
        n=frame["n"].to_numpy(dtype=float),
        k=frame["k"].to_numpy(dtype=float),
    )
    return table.interpolator()


class MaterialLibrary:
    """Convenience registry for dispersion data."""

    def __init__(self, material_map: Mapping[str, str] | None = None, data_dir: Path | None = None) -> None:
        self._material_map: MutableMapping[str, str] = dict(material_map or {})
        self._data_dir = Path(data_dir) if data_dir is not None else DATA_SUBDIR

    def register(self, name: str, filename: str) -> None:
        self._material_map[name] = filename

    def get(self, name: str) -> Callable[[float], complex]:
        if name in self._material_map:
            filename = self._material_map[name]
            return load_dispersion(filename, self._data_dir)
        return load_dispersion(name, self._data_dir)

    def available(self) -> Iterable[str]:
        registered = set(self._material_map)
        disk = {p.stem for p in self._data_dir.glob("*.csv")}
        return sorted(registered | disk)


__all__ = ["load_dispersion", "MaterialLibrary", "DispersionError", "DispersionTable"]
