"""Tests for native EMpy mode solver integration via the factory."""

from __future__ import annotations

import importlib.util

import pytest

from fbqc_snspe.modes_backend import ModeSolverFactoryError, create_mode_solver


def test_native_backend_requires_empy():
    if importlib.util.find_spec("EMpy") is not None:
        pytest.skip("EMpy is available; skipping missing-dependency check.")
    with pytest.raises(ModeSolverFactoryError):
        create_mode_solver({"backend": "empy_native"})
