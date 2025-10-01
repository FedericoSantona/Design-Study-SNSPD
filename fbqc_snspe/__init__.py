"""Toolkit for designing polarization-flattened waveguide-integrated SNSPDs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fbqc_snspe")
except PackageNotFoundError:  # pragma: no cover - local editable install
    __version__ = "0.0.0"

__all__ = ["__version__"]
