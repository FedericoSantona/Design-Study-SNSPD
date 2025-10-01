"""Geometry parameterisation for the SNSPD design workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np
from pydantic import BaseModel, Field, __version__ as _pydantic_version, validator


_PYDANTIC_MAJOR = int(_pydantic_version.split(".")[0])
_FIELD_REGEX_KEY = "pattern" if _PYDANTIC_MAJOR >= 2 else "regex"

from .materials import MaterialLibrary


class WaveguideCore(BaseModel):
    material: str = Field(..., description="Core material identifier")
    width_nm: float = Field(..., gt=0)
    height_nm: float = Field(..., gt=0)
    sidewall_angle_deg: float = Field(90.0, gt=0, le=90)


class Overlay(BaseModel):
    material: str = Field(...)
    thickness_nm: float = Field(..., gt=0)
    width_nm: float = Field(..., gt=0)
    offset_nm: float = Field(0.0)


class Nanowire(BaseModel):
    material: str = Field(...)
    width_nm: float = Field(..., gt=0)
    thickness_nm: float = Field(..., gt=0)
    pitch_nm: float = Field(..., gt=0)
    fill_factor: float = Field(..., gt=0, le=1.0)
    setback_nm: float = Field(0.0)

    @property
    def effective_width_nm(self) -> float:
        return self.pitch_nm * self.fill_factor


class Stack(BaseModel):
    substrate: str = Field(...)
    bottom_cladding: str = Field("SiO2")
    bottom_cladding_thickness_nm: float = Field(3000.0, gt=0)
    top_cladding: str = Field("SiO2")
    top_cladding_thickness_nm: float = Field(2000.0, gt=0)
    superstrate: str | None = Field(None)


_SCHEME_FIELD_KWARGS = {_FIELD_REGEX_KEY: "^(none|top_mirror|bottom_dbr|u_turn)$"}


class DualPassConfig(BaseModel):
    enable: bool = False
    scheme: str = Field("none", **_SCHEME_FIELD_KWARGS)
    length_um: float | None = Field(None, gt=0)
    reflectivity: float | None = Field(None, ge=0, le=1)


class DeviceParams(BaseModel):
    core: WaveguideCore
    overlay: Overlay
    nanowire: Nanowire
    stack: Stack
    propagation_length_um: float = Field(..., gt=0)
    dual_pass: DualPassConfig = Field(default_factory=DualPassConfig)

    class Config:
        arbitrary_types_allowed = True

    def describe_layers(self) -> List["LayerSpec"]:
        """Return a layer stack description sorted from bottom to top."""

        layers: List[LayerSpec] = []
        current = 0.0
        # substrate assumed semi-infinite
        layers.append(
            LayerSpec(
                name="substrate",
                material=self.stack.substrate,
                z_min_nm=-np.inf,
                z_max_nm=0.0,
                width_nm=np.inf,
                center_x_nm=0.0,
            )
        )
        current += self.stack.bottom_cladding_thickness_nm
        layers.append(
            LayerSpec(
                name="bottom_cladding",
                material=self.stack.bottom_cladding,
                z_min_nm=0.0,
                z_max_nm=current,
                width_nm=np.inf,
                center_x_nm=0.0,
            )
        )
        wg_bottom = current
        wg_top = wg_bottom + self.core.height_nm
        layers.append(
            LayerSpec(
                name="waveguide_core",
                material=self.core.material,
                z_min_nm=wg_bottom,
                z_max_nm=wg_top,
                width_nm=self.core.width_nm,
                center_x_nm=0.0,
            )
        )
        overlay_bottom = wg_top
        overlay_top = overlay_bottom + self.overlay.thickness_nm
        layers.append(
            LayerSpec(
                name="overlay",
                material=self.overlay.material,
                z_min_nm=overlay_bottom,
                z_max_nm=overlay_top,
                width_nm=self.overlay.width_nm,
                center_x_nm=self.overlay.offset_nm,
            )
        )
        nanowire_bottom = overlay_top + self.nanowire.setback_nm
        nanowire_top = nanowire_bottom + self.nanowire.thickness_nm
        layers.append(
            LayerSpec(
                name="nanowire",
                material=self.nanowire.material,
                z_min_nm=nanowire_bottom,
                z_max_nm=nanowire_top,
                width_nm=self.nanowire.effective_width_nm,
                center_x_nm=0.0,
            )
        )
        if self.stack.top_cladding_thickness_nm:
            top_bottom = nanowire_top
            top_top = top_bottom + self.stack.top_cladding_thickness_nm
            layers.append(
                LayerSpec(
                    name="top_cladding",
                    material=self.stack.top_cladding,
                    z_min_nm=top_bottom,
                    z_max_nm=top_top,
                    width_nm=np.inf,
                    center_x_nm=0.0,
                )
            )
        if self.stack.superstrate:
            layers.append(
                LayerSpec(
                    name="superstrate",
                    material=self.stack.superstrate,
                    z_min_nm=layers[-1].z_max_nm,
                    z_max_nm=np.inf,
                    width_nm=np.inf,
                    center_x_nm=0.0,
                )
            )
        return layers

    def to_cross_section(self, material_library: MaterialLibrary) -> "CrossSection":
        return CrossSection.from_layers(self.describe_layers(), material_library)


@dataclass(slots=True)
class LayerSpec:
    name: str
    material: str
    z_min_nm: float
    z_max_nm: float
    width_nm: float
    center_x_nm: float

    @property
    def thickness_nm(self) -> float:
        return self.z_max_nm - self.z_min_nm


@dataclass(slots=True)
class CrossSectionRegion:
    layer: LayerSpec
    refractive_index_fn: Callable[[float], complex]


class CrossSection:
    """Cross-section describing materials as a function of (x, z)."""

    def __init__(self, regions: Sequence[CrossSectionRegion]) -> None:
        self._regions = list(regions)

    @classmethod
    def from_layers(cls, layers: Sequence[LayerSpec], material_library: MaterialLibrary) -> "CrossSection":
        regions: List[CrossSectionRegion] = []
        for layer in layers:
            n_fn = material_library.get(layer.material)
            regions.append(CrossSectionRegion(layer=layer, refractive_index_fn=n_fn))
        return cls(regions)

    @property
    def regions(self) -> Sequence[CrossSectionRegion]:
        return tuple(self._regions)


__all__ = [
    "WaveguideCore",
    "Overlay",
    "Nanowire",
    "Stack",
    "DualPassConfig",
    "DeviceParams",
    "LayerSpec",
    "CrossSection",
    "CrossSectionRegion",
]
