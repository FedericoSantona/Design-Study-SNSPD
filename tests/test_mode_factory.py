"""Tests for the mode solver factory utilities."""

from __future__ import annotations

from fbqc_snspe.modes_backend import (
    AnalyticSlabModeSolver,
    EmpyModeSolver,
    EmpyOptions,
    ModeSolverFactoryError,
    create_mode_solver,
)


def test_factory_default_backend_is_empy():
    solver = create_mode_solver()
    assert isinstance(solver, EmpyModeSolver)
    assert solver.target_modes == 2


def test_factory_allows_target_modes_and_options():
    solver = create_mode_solver(
        {
            "backend": "empy",
            "target_modes": 3,
            "options": {"num_modes": 5, "which": "LR"},
        }
    )
    assert isinstance(solver, EmpyModeSolver)
    assert solver.target_modes == 3
    assert isinstance(solver.options, EmpyOptions)
    assert solver.options.num_modes == 5


def test_factory_analytic_backend():
    solver = create_mode_solver(
        {
            "backend": "analytic",
            "target_modes": 1,
            "lateral_decay_nm": 500.0,
        }
    )
    assert isinstance(solver, AnalyticSlabModeSolver)
    assert solver.target_modes == 1


def test_factory_invalid_backend_raises():
    try:
        create_mode_solver({"backend": "invalid"})
    except ModeSolverFactoryError as exc:
        assert "Unknown mode solver backend" in str(exc)
    else:  # pragma: no cover - ensures failure surfaces
        raise AssertionError("Expected ModeSolverFactoryError")


def test_factory_rejects_unknown_empy_arguments():
    try:
        create_mode_solver({"backend": "empy", "foo": "bar"})
    except ModeSolverFactoryError as exc:
        assert "Unsupported Empy solver parameters" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ModeSolverFactoryError")
