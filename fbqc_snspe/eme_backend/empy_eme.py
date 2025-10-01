"""EMpy-based Eigenmode Expansion backend."""

from __future__ import annotations

from .base import EMEBackend, EMEResult

try:  # pragma: no cover - optional dependency
    import EMpy
except Exception:
    EMpy = None


class EmpyEMEBackend(EMEBackend):
    def __init__(self) -> None:
        if EMpy is None:
            raise RuntimeError("EMpy is not installed. Install 'empy' to enable this backend.")

    def propagate(self, modes, length_um: float, reflection: float) -> EMEResult:  # pragma: no cover
        raise NotImplementedError("EmpyEMEBackend propagation is not yet implemented.")


__all__ = ["EmpyEMEBackend"]
