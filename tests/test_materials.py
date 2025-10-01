import numpy as np

from fbqc_snspe.materials import MaterialLibrary


def test_material_loader_returns_callable():
    library = MaterialLibrary()
    interp = library.get("SiN")
    n = interp(1550)
    assert isinstance(n, complex)
    assert np.isclose(n.real, 1.99, atol=0.05)
