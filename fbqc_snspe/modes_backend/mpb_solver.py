"""Wrapper around MPB (MIT Photonic Bands) for eigenmode extraction."""

from __future__ import annotations

from dataclasses import dataclass

from .base import ModeSolver

try:  # pragma: no cover - optional dependency
    import meep as mp
    from meep import mpb
except Exception:  # broad to handle partial installs
    mp = None
    mpb = None


@dataclass
class MPBOptions:
    resolution: int = 32
    supercell_size_um: float = 6.0
    num_bands: int = 4


class MPBModeSolver(ModeSolver):
    def __init__(self, options: MPBOptions | None = None, target_modes: int = 2) -> None:
        if mpb is None:
            raise RuntimeError(
                "MPB (meep.mpb) is not available. Install meep with MPB support to use this backend."
            )
        super().__init__(target_modes=target_modes)
        self.options = options or MPBOptions()

    def solve(self, mesh, wavelength_nm: float):  # pragma: no cover - relies on MPB
        raise NotImplementedError(
            "MPBModeSolver.solve requires full MPB geometry translation which is not yet implemented."
        )


__all__ = ["MPBModeSolver", "MPBOptions"]
