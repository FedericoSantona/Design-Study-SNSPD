"""Meshing utilities for the SNSPD design workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import CrossSection, CrossSectionRegion


@dataclass(slots=True)
class RectilinearMesh:
    x_nm: np.ndarray
    z_nm: np.ndarray
    region_indices: np.ndarray
    regions: Sequence[CrossSectionRegion]

    def epsilon(self, wavelength_nm: float) -> np.ndarray:
        eps = np.zeros_like(self.region_indices, dtype=np.complex128)
        for idx, region in enumerate(self.regions):
            mask = self.region_indices == idx
            if not np.any(mask):
                continue
            n_complex = region.refractive_index_fn(wavelength_nm)
            eps[mask] = n_complex**2
        return eps

    @property
    def shape(self) -> tuple[int, int]:
        return self.region_indices.shape


class MeshingError(RuntimeError):
    pass


class CrossSectionMesher:
    def __init__(
        self,
        cross_section: CrossSection,
        dx_nm: float = 10.0,
        dz_nm: float = 5.0,
        lateral_span_nm: float | None = None,
        bottom_padding_nm: float = 2000.0,
        top_padding_nm: float = 2000.0,
    ) -> None:
        if dx_nm <= 0 or dz_nm <= 0:
            raise MeshingError("Mesh steps must be positive.")
        self.cross_section = cross_section
        self.dx_nm = dx_nm
        self.dz_nm = dz_nm
        self.lateral_span_nm = lateral_span_nm
        self.bottom_padding_nm = bottom_padding_nm
        self.top_padding_nm = top_padding_nm

    def build(self) -> RectilinearMesh:
        regions = self.cross_section.regions
        finite_layers = [r.layer for r in regions if np.isfinite(r.layer.z_min_nm) and np.isfinite(r.layer.z_max_nm)]
        if not finite_layers:
            raise MeshingError("At least one finite layer is required for meshing.")
        min_z = min(layer.z_min_nm for layer in finite_layers)
        max_z = max(layer.z_max_nm for layer in finite_layers)
        z_start = min_z - self.bottom_padding_nm
        z_end = max_z + self.top_padding_nm
        dz_used = self._effective_dz(finite_layers)
        z_nm = np.arange(z_start, z_end + dz_used, dz_used)

        if self.lateral_span_nm is None:
            width_candidates = [layer.width_nm for layer in finite_layers if np.isfinite(layer.width_nm)]
            if width_candidates:
                x_half = 0.5 * max(width_candidates) + 1000.0
            else:
                x_half = 2000.0
        else:
            x_half = self.lateral_span_nm / 2
        x_nm = np.arange(-x_half, x_half + self.dx_nm, self.dx_nm)

        region_indices = np.zeros((len(z_nm), len(x_nm)), dtype=np.int16)
        for iz, z in enumerate(z_nm):
            for ix, x in enumerate(x_nm):
                region_indices[iz, ix] = self._select_region(x, z, regions)

        self._validate_layer_sampling(z_nm, region_indices, regions, dz_used)

        return RectilinearMesh(x_nm=x_nm, z_nm=z_nm, region_indices=region_indices, regions=regions)

    @staticmethod
    def _select_region(x_nm: float, z_nm: float, regions: Sequence[CrossSectionRegion]) -> int:
        for idx, region in enumerate(regions):
            layer = region.layer
            if not (layer.z_min_nm <= z_nm < layer.z_max_nm or (np.isinf(layer.z_max_nm) and z_nm >= layer.z_min_nm)):
                continue
            if np.isinf(layer.width_nm):
                return idx
            half_width = layer.width_nm / 2
            if abs(x_nm - layer.center_x_nm) <= half_width:
                return idx
        # default to last region (usually superstrate)
        return len(regions) - 1

    def _effective_dz(self, layers: Sequence) -> float:
        min_thickness = min(layer.thickness_nm for layer in layers)
        if min_thickness <= 0:
            raise MeshingError("Layers must have positive thickness.")
        dz_used = float(self.dz_nm)
        while dz_used > min_thickness / 2:
            dz_used /= 2
            if dz_used < 1e-3:
                break
        return dz_used

    def _validate_layer_sampling(
        self,
        z_nm: np.ndarray,
        region_indices: np.ndarray,
        regions: Sequence[CrossSectionRegion],
        dz_used: float,
    ) -> None:
        for idx, region in enumerate(regions):
            layer = region.layer
            if not (np.isfinite(layer.z_min_nm) and np.isfinite(layer.z_max_nm)):
                continue
            mask_rows = (z_nm >= layer.z_min_nm) & (z_nm < layer.z_max_nm)
            if not np.any(mask_rows):
                raise MeshingError(
                    f"Layer '{layer.name}' (thickness {layer.thickness_nm} nm) is not sampled by dz={dz_used:.3f} nm. "
                    "Reduce 'dz_nm' or adjust layer definition."
                )
            if not np.any(region_indices[mask_rows] == idx):
                raise MeshingError(
                    f"Layer '{layer.name}' is present in geometry but no mesh cells were assigned. "
                    "Refine the vertical mesh or check material widths."
                )


__all__ = ["RectilinearMesh", "CrossSectionMesher", "MeshingError"]
