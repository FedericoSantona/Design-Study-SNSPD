import numpy as np

from fbqc_snspe.absorptance import AbsorptanceCalculator
from fbqc_snspe.geometry import DeviceParams, DualPassConfig, Nanowire, Overlay, Stack, WaveguideCore
from fbqc_snspe.materials import MaterialLibrary


def make_params() -> DeviceParams:
    return DeviceParams(
        core=WaveguideCore(material="SiN", width_nm=1200, height_nm=400),
        overlay=Overlay(material="TiO2", thickness_nm=50, width_nm=1200),
        nanowire=Nanowire(
            material="NbTiN",
            width_nm=80,
            thickness_nm=7,
            pitch_nm=160,
            fill_factor=0.5,
            setback_nm=5,
        ),
        stack=Stack(substrate="SiO2"),
        propagation_length_um=50,
        dual_pass=DualPassConfig(enable=False),
    )


def test_absorptance_pipeline_runs():
    params = make_params()
    library = MaterialLibrary()
    calc = AbsorptanceCalculator(params=params, material_library=library)
    result = calc.sweep([1530, 1550, 1570])
    assert result.te.shape == (3,)
    assert result.tm.shape == (3,)
    assert np.all((result.te >= 0) & (result.te <= 1))
    assert np.all((result.tm >= 0) & (result.tm <= 1))
