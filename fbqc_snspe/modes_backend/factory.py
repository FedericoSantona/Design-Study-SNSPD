"""Factory utilities for instantiating mode solver backends."""

from __future__ import annotations

from typing import Any, Mapping

from .analytic_solver import AnalyticSlabModeSolver
from .base import ModeSolver
from .empy_solver import EmpyModeSolver, EmpyOptions
from .empy_native import EmpyNativeModeSolver, EmpyNativeOptions, EmpyNativeError


class ModeSolverFactoryError(ValueError):
    """Raised when a solver configuration is invalid."""


def create_mode_solver(config: Mapping[str, Any] | None = None) -> ModeSolver:
    """Create a :class:`ModeSolver` from a user configuration mapping.

    Parameters
    ----------
    config:
        Mapping describing the solver. Supported keys:

        ``backend`` (str, optional):
            Name of the backend to instantiate. Supported values are
            ``"empy"`` (default) and ``"analytic"``.
        ``target_modes`` (int, optional):
            Number of modes to keep in the returned solver. Defaults to 2.
        ``options`` (mapping, optional):
            Backend-specific options. For ``"empy"`` they are passed to
            :class:`EmpyOptions`.
        Additional keys are forwarded to backend constructors when supported
        (e.g. ``lateral_decay_nm`` for the analytic solver).

    Returns
    -------
    ModeSolver
        Instantiated solver ready to use.
    """

    cfg = dict(config or {})
    backend = str(cfg.pop("backend", "empy")).lower()
    target_modes = int(cfg.pop("target_modes", 2))

    if backend == "analytic":
        return AnalyticSlabModeSolver(target_modes=target_modes, **cfg)

    if backend in {"empy", "empy_fd"}:
        options_cfg = cfg.pop("options", {})
        if cfg:
            unknown = ", ".join(sorted(cfg.keys()))
            raise ModeSolverFactoryError(
                f"Unsupported Empy solver parameters: {unknown}"
            )
        options = EmpyOptions(**options_cfg)
        return EmpyModeSolver(options=options, target_modes=target_modes)

    if backend in {"empy_native", "empy_original"}:
        options_cfg = cfg.pop("options", {})
        if cfg:
            unknown = ", ".join(sorted(cfg.keys()))
            raise ModeSolverFactoryError(
                f"Unsupported Empy native solver parameters: {unknown}"
            )
        options = EmpyNativeOptions(**options_cfg)
        try:
            return EmpyNativeModeSolver(options=options, target_modes=target_modes)
        except EmpyNativeError as exc:
            raise ModeSolverFactoryError(str(exc)) from exc

    raise ModeSolverFactoryError(f"Unknown mode solver backend '{backend}'.")


__all__ = ["create_mode_solver", "ModeSolverFactoryError"]
