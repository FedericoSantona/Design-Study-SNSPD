# fbqc_snspe – Full-Stack SNSPD Design Toolkit

A comprehensive Python package for designing waveguide-integrated superconducting nanowire single-photon detectors (SNSPDs) tailored to fusion-based photonic quantum computing (FBQC). The toolkit builds a complete workflow: configure material dispersion and geometry, discretize the cross-section, solve for guided modes with multiple backend options (finite-difference or native EMpy), compute absorptance metrics across telecom bands, and explore the design space via automated sweeps, plots, and reports.

> **Target:** Flatten TE/TM absorptance (|ΔA| < 1 dB) across the C-band while maintaining high average absorptance (≥0.8–0.9) by co-engineering waveguide, overlay, nanowire, and optional dual-pass structures.

---

## 1. Prerequisites

- macOS or Linux (Windows via WSL should work; the finite-difference solver is pure Python, native EMpy requires compatible binaries).
- Python **3.10 or 3.11**. (The code uses `str | None` unions and Pydantic v2 support—they require Python ≥ 3.10.)
- A C/Fortran toolchain is required if you plan to install Meep/MPB; otherwise the finite-difference solver works without compilation.

---

## 2. Dependencies

Install everything in one go (all components are treated as required for simplicity):

```bash
python3.11 -m venv myenv
source myenv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes:

- Numerical/analysis stack: `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`
- Configuration/dataclasses: `pydantic`, `pyyaml`
- Rendering/templating: `jinja2`
- Solvers: `EMpy`, `meep>=1.25` (MPB + Meep; Meep install may require HDF5, MPI, and other prerequisites—follow Meep docs if needed).

If Meep or EMpy are unavailable, you can still run with the finite-difference backend (`empy_fd`); the factory will raise a descriptive error if you select `empy_native` without EMpy installed.

---

## 3. Repository Layout

```
config/
  example_sweep.yaml      # Annotated sweep configuration
  ...
data/
  materials/*.csv         # n(λ)+ik dispersion tables
fbqc_snspe/
  __init__.py
  materials.py            # Dispersion parsing & interpolation
  geometry.py             # Pydantic device definitions & cross-section builder
  mesh.py                 # Rectilinear meshing utilities
  modes_backend/
    base.py               # Mode, ModeSolver interfaces
    analytic_solver.py    # Lightweight Gaussian EIM fallback
    empy_solver.py        # Finite-difference TE/TM eigen solver (SciPy)
    empy_native.py        # Native EMpy WGMSolver wrapper
    factory.py            # Backend selection by config
  eme_backend/            # Eigenmode expansion interfaces
  absorptance.py          # Mode overlap → attenuation → absorptance
  objectives.py           # ΔA metrics, composite objective, PDDE proxy
  sweep.py                # Grid sweeps with tqdm/logging
  tolerance.py            # Monte Carlo tolerance analysis
  plots.py                # Band/delta/heatmap/pareto plotting helpers
  report.py               # Jinja HTML report generator
scripts/
  run_sweep.py            # CLI entry to execute sweeps from YAML
  make_report.py          # Build HTML summary from CSV results
tests/
  ...                     # pytest suite (geometry/materials/modes/factory)
requirements.py           # Legacy list of required packages (kept for reference)
requirements.txt          # Full dependency list (install from here)
```

---

## 4. Data and Configuration

### 4.1 Material Dispersion

CSV files in `data/materials/` must contain headers `wavelength_nm`, `n`, `k` (real and imaginary parts). `MaterialLibrary` loads/interpolates these tables and caches them. Add new materials by dropping more CSVs or calling `MaterialLibrary.register`.

### 4.2 Device Definition (`config/example_sweep.yaml`)

Annotated YAML file defines:

- `base_params`: Nominal geometry—waveguide core, overlay, nanowire, dielectric stack, propagation length, dual-pass config.
  - `dual_pass.scheme`: `none`, `top_mirror`, `bottom_dbr`, `u_turn` (currently mirror modeled by effective reflectivity).
- `sweep`: Wavelength grid, parameter sweeps (cartesian product), target absorptance thresholds.
  - Example uses 8-point grid (2×2×2) and a **trimmed 3-point wavelength set** (1530, 1550, 1570 nm) for quick iterations.
- `solver`: Backend choice (`empy_fd`, `empy_native`, `analytic`), target modes, solver-specific options.
- `mesh`: Discretization parameters (step sizes, lateral span, top/bottom padding).

You can clone this YAML to explore other geometries or modify it directly. Comments explain each parameter and acceptable choices.

---

## 5. Workflow Overview

1. **Configuration Parsing:** `run_sweep.py` loads YAML via PyYAML, validates `DeviceParams` with Pydantic (`model_validate` under v2).
2. **Cross-Section & Meshing:** `DeviceParams.to_cross_section` enumerates layers → `CrossSectionMesher` constructs a uniform x–z grid with region indices and complex permittivity maps.
3. **Mode Solving:** For each wavelength and sweep point, `ModeSolverFactory` instantiates the requested backend:
   - `empy_fd`: Finite-difference TE/TM scalar eigenproblems with SciPy (`eigs`). Reconstructs fields (E/H) on the mesh.
   - `empy_native`: Calls EMpy’s `WGMSolver`; requires EMpy installation. Fields come directly from the native solver.
   - `analytic`: Simple Gaussian field approximations (fast but inaccurate) for trend exploration.
4. **Absorptance Calculation:** `AbsorptanceCalculator` computes overlap between mode fields and the lossy nanowire region, converts to attenuation (`alpha`), then to absorptance (single-pass or dual-pass reflectivity model).
5. **Metric Aggregation:** `objectives.metrics` returns max |ΔA| in dB, mean/worst absorptance; `objective` adds penalty if mean < threshold; `pdde_proxy` approximates polarization-dependent detection efficiency spread.
6. **Sweep Loop:** `sweep.run_grid_sweep` iterates the cartesian product; tqdm tracks overall progress, optionally log each point with `-v` CLI flag. Results accumulate in a pandas DataFrame and write to CSV.
7. **Reporting:** Use `scripts/make_report.py` to turn a CSV into an HTML summary (top candidates table + metrics). `plots.py` exposes Matplotlib routines for band plots, deltas, heatmaps, Pareto charts.

---

## 6. Running the Toolkit

Assuming the virtualenv is active and dependencies installed:

```bash
# optional: install the package in editable mode
python -m pip install --no-build-isolation -e .

# fast sanity sweep (finite-difference backend)
python scripts/run_sweep.py config/example_sweep.yaml --output outputs/sweep.csv -v

# review numerical results
head -n 5 outputs/sweep.csv

# generate HTML summary (top 5 candidates)
python scripts/make_report.py outputs/sweep.csv --report outputs/report.html --top 5
open outputs/report.html  # use your OS equivalent
```

### Switching Solvers

Edit `config/example_sweep.yaml`:

- `solver.backend: empy_fd` → finite-difference solver (no extra dependencies).
- `solver.backend: empy_native` → original EMpy solver. Ensure `EMpy` is importable; otherwise the CLI will raise `ModeSolverFactoryError` explaining the missing dependency.
- `solver.backend: analytic` → quick approximation for exploratory runs.
- `solver.options.num_modes` controls the number of modes returned. Extra keys like `which` apply only to the finite-difference backend; remove them when switching to `empy_native`.

### Custom Sweeps

- Adjust `sweep.wavelengths_nm` for different bands (e.g., 1500–1600 nm).
- Add more parameters under `sweep_space` (e.g., `overlay.width_nm`, `nanowire.setback_nm`). Syntax is dotted path into the `DeviceParams` model.
- Change `show_progress`/`verbose` behaviours via CLI flags or modifying `run_grid_sweep` arguments.

### Performance Tuning

- **Mesh resolution:** The example YAML now uses 30 nm × 5 nm spacing. Keep `dz_nm` at or below your thinnest layer (the nanowire is 7 nm); a coarser vertical step will literally skip that layer and yield zero absorptance. Refine (`dx_nm`, padding) once candidates emerge.
- **Wavelength samples:** Start with the provided three-point set. Add back intermediate wavelengths once the design narrows.
- **Eigenmodes:** Keeping `num_modes=2` targets fundamental TE/TM modes. Increase only if you expect higher-order modes to contribute.
- **Backend choice:** The finite-difference backend (`empy_fd`) is fastest without extra dependencies. Switch to `empy_native` only when you need EMpy’s higher fidelity and have it installed.

### Tolerance Analysis & Advanced Workflows

- `fbqc_snspe/tolerance.py`: Monte Carlo perturbations of geometry/material; returns DataFrame with yield (band_ok) statistics.
- `fbqc_snspe/plots.py`: Use within notebooks or scripts to visualize absorptance spectra, delta plots, parameter heatmaps, Pareto fronts.
- `fbqc_snspe/report.py`: `render_report` takes metrics, candidate table, and optional figure paths to produce an HTML file.

---

## 7. Testing & Validation

Run the pytest suite (install pytest first if not included):

```bash
python -m pip install pytest
python -m pytest
```

Tests cover: material interpolation, absorptance pipeline sanity, finite-difference solver (structure/outputs), factory error handling. Extend with additional tests as you add features (e.g., verifying native EMpy output once EMpy is available).

---

## 8. Extending the Toolkit

- **New Materials:** add CSV files in `data/materials`, update YAML to reference them, or register programmatically.
- **Additional Solvers:** implement a new backend under `modes_backend/` (e.g., MPB wrapper), expose it in `factory.create_mode_solver`, and document a new backend keyword.
- **Dual-Pass Models:** currently uses a lumped reflectivity. For explicit U-turn waveguide modelling, implement in `eme_backend` and extend `AbsorptanceCalculator` to invoke `EMEBackend`.
- **Optimization:** the current sweep is a grid; plug in a Bayesian or evolutionary optimizer by evaluating `objective(params)` and integrating with `sweep.run_grid_sweep` or a new routine.
- **Tolerance/Monte Carlo:** configure `ToleranceConfig` with parameter perturbations to estimate yield vs fabrication variations.

---

## 9. Troubleshooting

- **`ModuleNotFoundError: fbqc_snspe`** → ensure you’re running from repo root and either set `export PYTHONPATH=$PWD` or install via `pip install -e .`.
- **Solver errors (EMpy)** → backend `empy_native` requires `EMpy`; install `EMpy` or switch to `empy_fd`.
- **Pydantic warnings** → the code handles both v1 and v2. Ensure `pydantic>=1.10,<3` per requirements.
- **Slow sweeps** → shrink `sweep_space`, widen mesh steps, or switch to `analytic` backend for first-pass scanning.
- **Meep installation issues** → refer to official Meep docs; HDF5/MPI dependencies may be necessary.

---

## 10. References & Next Steps

- Populate `REFERENCES.md` with relevant SNSPD and FBQC literature.
- Use `scripts/run_sweep.py` as the primary automation entry point—integrate into CI or parameter study pipelines.
- Export best designs via CSV/JSON and feed into layout tools or further optical/electrothermal analysis.

The codebase is structured to let you scale from quick what-if experiments to full parameter scans with richer solvers. Enjoy detecting photons!
