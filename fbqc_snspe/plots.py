"""Plotting helpers for SNSPD design results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .absorptance import PolarizationAbsorptance


def plot_band(absorptance: PolarizationAbsorptance, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    ax.plot(absorptance.wavelengths_nm, absorptance.te, label="TE")
    ax.plot(absorptance.wavelengths_nm, absorptance.tm, label="TM")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorptance")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    return ax


def plot_delta(absorptance: PolarizationAbsorptance, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    ax.plot(absorptance.wavelengths_nm, np.abs(absorptance.delta_db))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$|\Delta A|$ (dB)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    return ax


def heatmap(df: pd.DataFrame, x: str, y: str, value: str, ax: plt.Axes | None = None) -> plt.Axes:
    pivot = df.pivot_table(index=y, columns=x, values=value)
    ax = ax or plt.gca()
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.colorbar(im, ax=ax, label=value)
    return ax


def pareto(df: pd.DataFrame, x: str = "delta_db_max", y: str = "mean_absorptance", ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    ax.scatter(df[x], df[y], c=df.get("band_ok", False), cmap="coolwarm", edgecolor="k")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, linestyle=":", linewidth=0.5)
    return ax


__all__ = ["plot_band", "plot_delta", "heatmap", "pareto"]
