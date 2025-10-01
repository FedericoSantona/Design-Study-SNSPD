"""Tests for the finite-difference eigenmode solver backend."""

from __future__ import annotations

import numpy as np

from fbqc_snspe.geometry import DeviceParams, Nanowire, Overlay, Stack, WaveguideCore
from fbqc_snspe.materials import MaterialLibrary
from fbqc_snspe.mesh import CrossSectionMesher
from fbqc_snspe.modes_backend.empy_solver import EmpyModeSolver, EmpyOptions


def _build_test_mesh():
    params = DeviceParams(
        core=WaveguideCore(material="SiN", width_nm=800.0, height_nm=400.0),
        overlay=Overlay(material="TiO2", thickness_nm=80.0, width_nm=900.0),
        nanowire=Nanowire(
            material="NbTiN",
            width_nm=80.0,
            thickness_nm=6.0,
            pitch_nm=160.0,
            fill_factor=0.5,
            setback_nm=10.0,
        ),
        stack=Stack(substrate="SiO2", bottom_cladding="SiO2", top_cladding="SiO2"),
        propagation_length_um=50.0,
    )
    cross_section = params.to_cross_section(MaterialLibrary())
    mesher = CrossSectionMesher(cross_section, dx_nm=20.0, dz_nm=10.0, lateral_span_nm=3000.0)
    return mesher.build()


def test_empy_solver_returns_modes():
    mesh = _build_test_mesh()
    solver = EmpyModeSolver(EmpyOptions(num_modes=3))
    result = solver.solve(mesh, wavelength_nm=1550.0)

    te_mode = result.polarization_mode("TE")
    tm_mode = result.polarization_mode("TM")

    assert te_mode is not None, "TE mode should be present"
    assert tm_mode is not None, "TM mode should be present"

    for mode in (te_mode, tm_mode):
        assert mode.ex.shape == mesh.region_indices.shape
        assert mode.ey.shape == mesh.region_indices.shape
        assert mode.ez.shape == mesh.region_indices.shape
        assert 1.0 < np.real(mode.neff) < 4.0
        assert mode.attenuation_neper_per_um is not None
        assert mode.attenuation_neper_per_um >= 0.0
