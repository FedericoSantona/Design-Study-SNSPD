#!/usr/bin/env python3
"""Visualise the layered cross-section and permittivity map for a configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

if not matplotlib.rcParams.get("backend"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fbqc_snspe.geometry import DeviceParams
from fbqc_snspe.materials import MaterialLibrary
from fbqc_snspe.mesh import CrossSectionMesher


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def plot_regions(mesh, ax: plt.Axes) -> None:
    x = mesh.x_nm
    z = mesh.z_nm
    X, Z = np.meshgrid(x, z)
    region_map = mesh.region_indices
    cmap = plt.get_cmap("tab20")

    ax.pcolormesh(X, Z, region_map, cmap=cmap, shading="auto")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("z (nm)")
    ax.set_title("Layer regions")
    ax.set_aspect("equal", adjustable="box")


def plot_permittivity(mesh, wavelength_nm: float, ax: plt.Axes) -> None:
    eps = mesh.epsilon(wavelength_nm)
    x = mesh.x_nm
    z = mesh.z_nm
    X, Z = np.meshgrid(x, z)
    img = ax.pcolormesh(X, Z, np.real(eps), shading="auto", cmap="viridis")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("z (nm)")
    ax.set_title(f"Re(epsilon) at {wavelength_nm} nm")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(img, ax=ax)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cross-section mesh and permittivity")
    parser.add_argument("config", type=Path, help="YAML configuration file")
    parser.add_argument("--wavelength", type=float, default=1550.0, help="Wavelength (nm)")
    parser.add_argument("--output", type=Path, help="Optional path to save figure instead of showing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_params_data = cfg["base_params"]
    if hasattr(DeviceParams, "model_validate"):
        params = DeviceParams.model_validate(base_params_data)
    else:  # pragma: no cover
        params = DeviceParams.parse_obj(base_params_data)
    material_library = MaterialLibrary()
    cross_section = params.to_cross_section(material_library)
    mesher = CrossSectionMesher(
        cross_section,
        dx_nm=cfg.get("mesh", {}).get("dx_nm", 20),
        dz_nm=cfg.get("mesh", {}).get("dz_nm", 10),
        lateral_span_nm=cfg.get("mesh", {}).get("lateral_span_nm"),
        bottom_padding_nm=cfg.get("mesh", {}).get("bottom_padding_nm", 2000.0),
        top_padding_nm=cfg.get("mesh", {}).get("top_padding_nm", 2000.0),
    )
    mesh = mesher.build()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_regions(mesh, ax1)
    plot_permittivity(mesh, args.wavelength, ax2)
    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300)
    else:  # pragma: no cover - requires interactive backend
        plt.show()


if __name__ == "__main__":
    main()
