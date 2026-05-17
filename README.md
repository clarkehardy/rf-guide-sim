# RF Guide Simulation

Two particle-in-trap simulations of the loading chain for an optical Paul trap, built on [`trapsim`](https://github.com/clarkehardy/trapsim).

| Simulation | Directory | Particle | Drag regime | RF | Purpose |
|---|---|---|---|---|---|
| RF guide | `rf_guide/` | 166 nm SiO₂ nanosphere, 100e | Epstein (free molecular, Kn ~ 800) | 2 kHz, ±10 V | Guide sphere ~400 mm through a linear Paul trap RF guide into an optical trap |
| Paul trap | `paul_trap/` | 20 µm ethanol droplet, 10⁶e | Continuum (Stokes / Schiller-Naumann) | 1 kHz, 200 V | Drop falls under gravity; RF switches on mid-fall to catch it |

Both simulations share the STL library in `stl/`.

---

## Installation

```
git clone https://github.com/clarkehardy/rf-guide-sim
cd rf-guide-sim
pip install -r requirements.txt
```

This installs `trapsim[all]` from GitHub, bringing in numpy, pyyaml, trimesh, matplotlib, and pyvista. The C++ Laplace solver compiles automatically on first use (needs Xcode CLT on macOS or `build-essential` on Linux).

Tested with Python 3.12 on macOS (M-series).

---

## Running a simulation

Each simulation is self-contained in its subdirectory. `run.py` is a three-line shim that delegates to `trapsim.run`:

```
cd rf_guide
python run.py                             # refine → fly → animate → visualize
python run.py --no-animate --no-visualize # fly only
python run.py --run 2                     # second run → trajectories_2.csv
python run.py --refine                    # force re-solve (needed after geometry changes)
```

```
cd paul_trap
python run.py
```

Output files are written into the simulation folder alongside `geometry.yaml`:

| File | Contents |
|---|---|
| `field.pa<N>` | Unit-potential array for electrode N (SIMION-compatible binary) |
| `trajectories_<N>.csv` | Particle trajectories in Fusion-world mm |
| `schedule_<N>.json` | Voltage schedule snapshot used for this run |
| `solver/` | Compiled C++ solver binary and voxel masks (auto-created; gitignored) |

For full CLI flags and output format details see the [trapsim README](https://github.com/clarkehardy/trapsim).

---

## RF guide simulation (`rf_guide/`)

**Particle:** 166 nm fused-silica nanosphere (ρ = 2200 kg/m³, m ≈ 5.3×10⁻¹⁸ kg), 100 elementary charges. 20 particles launched from z = −98 mm (loading endcap region) with a 0.1 mm axial position spread.

**Physics:** Free-molecular (Epstein) drag at 0.1 Pa N₂ gas, T = 293 K. Langevin thermal noise via the fluctuation-dissipation theorem. Pressure ramps from 0.1 Pa → 100 Pa over 0.5 s once the `catch_sphere` trigger fires, damping residual kinetic energy after trapping.

**Geometry:** 10 electrodes, 3 dielectrics, grid 0.25 mm over x ∈ [−25, 40], y ∈ [−8, 37], z ∈ [−132, 295] mm.

| Electrode | Wired bodies | Role |
|---|---|---|
| `rf_loading` | rod sets 1 & 2 TL/BR | +RF loading guide |
| `rf_loading_inv` | rod sets 1 & 2 TR/BL | −RF loading guide |
| `endcap_load_U/D` | upstream endcaps | axial confinement in loading region |
| `rod_3_TL/TR/BL/BR` | rod set 3 (individual) | DC trims + RF for optical Paul trap |
| `endcap_optical_U/D` | optical trap endcaps | axial confinement in trapping region |

| Dielectric | ε_r | Role |
|---|---|---|
| `trapping_lens` | 3.0 | lens in optical trap region |
| `collection_lens` | 3.0 | collection-side lens |
| `lens_holder` | 3.0 | lens mount |

**Voltage scheme:** 2 kHz RF at ±10 V on `rf_loading`/`rf_loading_inv`; DC endcap bias ±10 V in loading region; rod set 3 and optical endcaps initially grounded.

**Triggers:**

| Trigger | Axis | Threshold | Action |
|---|---|---|---|
| `throw_sphere` | z | −83.52 mm | Drops `endcap_load_D` voltage to 0 V over 0.1 µs, releasing the sphere across the gate-valve gap |
| `catch_sphere` | z | 272 mm | Ramps DC trims on rod set 3 to ±15 V over 0.1 s; pulses optical endcaps to 20 V; begins pressure ramp |

---

## Paul trap simulation (`paul_trap/`)

**Particle:** 20 µm ethanol droplet (ρ = 789 kg/m³, m ≈ 3.3×10⁻¹¹ kg), 10⁶ elementary charges. 20 particles start at y = 31 mm (inkjet tip location) with 0.5 mm position spread, falling under gravity.

**Physics:** Continuum drag at atmospheric pressure (`ContinuumDrag`: Schiller-Naumann at Re > 1, Stokes at Re ≤ 1) with air at ρ = 1.225 kg/m³, η = 1.81×10⁻⁵ Pa·s. Langevin thermal noise at 293 K — dormant during free fall (Re ≫ 1, `damping_rate` = 0) and activates automatically once the droplet slows to Re ≤ 1 after trapping.

**Geometry:** 4 electrodes, 1 dielectric, grid 0.25 mm over x ∈ [−25, 40], y ∈ [−8, 37], z ∈ [−132, −70] mm.

| Electrode | Wired bodies | Role |
|---|---|---|
| `rf_loading` | `PT_rod_TL`, `PT_rod_BR` | +RF Paul trap rods |
| `rf_loading_inv` | `PT_rod_TR`, `PT_rod_BL` | −RF Paul trap rods |
| `endcap_load_U/D` | Paul trap endcaps | axial confinement (currently unbiased) |

| Dielectric | ε_r | Role |
|---|---|---|
| `electrode_holder` | 3.0 | rod mount structure |

**Voltage scheme:** RF starts at zero amplitude (trap off). Endcaps unbiased.

**Trigger:**

| Trigger | Axis | Threshold | Action |
|---|---|---|---|
| `turn_on_rf` | −y | 19 mm | Droplet falls through y = 19 mm; RF amplitude steps to 200 V at 1 kHz on both rod pairs |

The −y axis means the trigger fires on a downward-going (negative y) crossing: the droplet must be falling through y = 19 mm, not launched upward past it.

---

## Repository layout

```
rf_guide/
  geometry.yaml     10-electrode RF guide + optical Paul trap geometry
  experiment.py     nanosphere params, Epstein drag, loading triggers
  run.py            trapsim.run shim
  solver/           compiled solver + voxel masks (gitignored, auto-created)

paul_trap/
  geometry.yaml     4-electrode Paul trap geometry
  experiment.py     ethanol droplet params, continuum drag, RF-on trigger
  run.py            trapsim.run shim
  solver/           compiled solver + voxel masks (gitignored, auto-created)

stl/                all STL meshes shared by both simulations
legacy/             original SIMION files and early Python pipeline (reference only)
requirements.txt    pip dependency: trapsim[all] from GitHub
```

The `trapsim` package ([github.com/clarkehardy/trapsim](https://github.com/clarkehardy/trapsim)) provides the Laplace solver, integrator, schedule/trigger engine, and all visualization tools. This repo contains only the geometry and experiment definitions.

---

## Performance

Reference: M-series Mac, all CPUs, 20 particles.

| Step | RF guide | Paul trap |
|---|---|---|
| Refine (once per geometry edit) | ~3 min (10 electrodes) | ~1 min (4 electrodes) |
| Fly | ~5 s | ~30 s |
| Animate + visualize | a few seconds | a few seconds |

Refine is the bottleneck; fly is fast because the Dormand-Prince adaptive stepper takes large steps in smooth field regions.
