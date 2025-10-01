"""Abstract interfaces for Eigenmode Expansion (EME) backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..modes_backend.base import Mode


@dataclass(slots=True)
class EMEResult:
    transmission: complex
    reflection: complex
    fields: dict[str, np.ndarray] | None = None

    @property
    def insertion_loss_db(self) -> float:
        return -10 * np.log10(abs(self.transmission) ** 2)


class EMEBackend(ABC):
    """Abstract EigenMode Expansion interface."""

    @abstractmethod
    def propagate(self, modes: Sequence[Mode], length_um: float, reflection: float) -> EMEResult:
        """Propagate supplied modes through a section of length ``length_um``."""
        raise NotImplementedError


__all__ = ["EMEBackend", "EMEResult"]
