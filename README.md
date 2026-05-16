# trapsim

Geometry-agnostic particle integrator for arbitrary electrode trap geometries. Replaces the SIMION 2024 workflow with a pure-Python pipeline (C++ Laplace solver + Python integrator) driven by a single `geometry.yaml`.

The original Paul-trap-loading-via-RF-guide simulation that motivated this code ships as the default example: 166 nm silica nanospheres travelling 400 mm down an RF guide through a gate valve into an optical Paul trap. Any other electrode geometry works by editing two files — `geometry.yaml` and `experiment.py`.

---

## Installation

```
brew install libomp                # optional, parallelises the solver
~/.venvs/mesh/bin/pip install pyyaml trimesh numpy matplotlib pyvista
make -C solver                     # compiles solver/laplace once
```

Tested with Python 3.12 on macOS (M-series and Intel). The C++ solver uses only Xcode Command Line Tools — no external dependencies.

---

## Quickstart

```
git clone <this repo>
cd <repo>
python run.py                      # refine if needed → fly → animate → visualize
```

That runs the included Paul-trap example: 20 particles, ~5 s wall on an M-series Mac.

Output files (alongside the source):

- `paulTrap.pa<N>` — unit-potential array per electrode (SIMION-compatible)
- `trajectories_1.csv` — recorded trajectories in Fusion-world mm
- `schedule_1.json` — snapshot of the voltage schedule used (for animate)

---

## Defining a new geometry

Two files: **`geometry.yaml`** (what exists) and **`experiment.py`** (what happens).

### `geometry.yaml`

```yaml
grid:
  dx_mm: 0.5
  bounds_mm:
    x: [-25.0,  40.0]    # simulation volume in Fusion-world coords
    y: [ -8.0,  37.0]
    z: [-132.0, 295.0]

electrodes:
  - name: rf_loading           # used by the voltage schedule and physics
    stls:
      - stl/rod_1_TL.stl       # all listed STLs are wired together
      - stl/rod_1_BR.stl
    color: [0.85, 0.20, 0.15]  # optional, RGB 0..1 for visualizations
  - name: endcap_load_U
    stls: [stl/endcap_load_U.stl]
  # … one entry per independent voltage source ...

dielectrics:
  - name: trapping_lens
    stl: stl/trapping_lens.stl
    epsilon_r: 3.0

decoration:                    # bodies drawn but with no field contribution
  - name: ring_brake
    stl: stl/ring_brake.stl
```

Each electrode is assigned an integer `electrode_id` (1, 2, …) in declaration order — that's the suffix on `paulTrap.pa<id>`. Re-ordering electrodes re-numbers them.

STL paths are resolved against the YAML's directory, then the repo root, then `stl/` — so `rod_1_TL.stl` and `stl/rod_1_TL.stl` both work.

### `experiment.py`

Plain Python.  Edit the four blocks below to change a run.

```python
import numpy as np
from trapsim.physics import Electrostatic, Gravity, EpsteinDrag, Langevin

# (1) Particle and (2) starting conditions
particle  = {"radius_m": 83e-9, "density_kgm3": 2200, "charge_e": 100}
particles = {"n": 20,
             "starts": [{"position_mm": [0.0, 19.0, -98.12],
                         "ke_ev": 0.0, "direction": [0, 0, 1],
                         "sigma_mm": [0.0, 0.0, 0.1]}]}

# (3) Physics list — pluggable, see below
physics = [Electrostatic(),
           Gravity(),
           EpsteinDrag(pressure_pa=0.1, temperature_k=293, gas_mass_amu=28.0),
           Langevin(temperature_k=293)]

# (4) Integrator
integrator = {"dt_init_us": 1.0, "dt_min_us": 0.01, "dt_max_us": 25.0,
              "atol": 1e-3, "rtol": 1e-4,
              "v_stop_mm_us": 1e-6, "record_stride": 20}

# (5) Main voltage schedule
t = np.linspace(0, 2e5, 1000)
main_schedule = {
    "time_us": t,
    "dc": {"endcap_load_U":  10*np.ones_like(t),
           "endcap_load_D": -10*np.ones_like(t)},
    "rf": {"rf_loading":     {"amplitude": 10*np.ones_like(t),
                              "frequency_hz": 2000, "phase_deg":   0},
           "rf_loading_inv": {"amplitude": 10*np.ones_like(t),
                              "frequency_hz": 2000, "phase_deg": 180}},
}

# (6) Triggers — each fires when pos[axis] >= threshold; its schedule then
# overrides ONLY the listed electrodes from t_fire onward.  Each trigger has
# its OWN time array (relative to fire time).
triggers = [
    {"name": "drop_load_endcap",
     "axis": "z", "threshold_mm": -83.52,
     "schedule": {"time_us": np.array([0, 200, 200.1, 1e6]),
                  "dc": {"endcap_load_D": np.array([-10, -10, 0, 0])}}},
]
```

Every voltage uses **electrode names**, not numbers. The names must match `geometry.yaml`.

---

## Writing custom physics modules

A physics module overrides any subset of three hooks:

```python
from trapsim.physics import Physics
import numpy as np

class HarmonicAxialTrap(Physics):
    def __init__(self, omega_us, z0_mm):
        self.omega2 = omega_us ** 2; self.z0 = z0_mm
    def accel(self, t_us, pos_mm, vel_mm_us, env):
        return np.array([0, 0, -self.omega2 * (pos_mm[2] - self.z0)])
```

Drop it into `experiment.py`'s `physics = [...]` list. No registration needed.

The hooks:

| Hook | What it returns | When |
|------|-----------------|------|
| `accel(t, pos, vel, env)`         | 3-vec acceleration [mm/µs²] | every RK4/5 stage |
| `damping_rate(t, pos, vel, env)`  | scalar γ [1/µs] | once per accepted step |
| `kick(dt, t, pos, vel, env)`      | 3-vec Δv [mm/µs]            | once per accepted step |

`env` exposes:
- `env.particle` — dict with `mass_kg`, `charge_C`, `radius_m`, `charge_e`
- `env.voltages` — `{electrode_name: V}` at the current time
- `env.field(pos_mm)` — total `(Ex, Ey, Ez)` in V/mm at `pos_mm` (Fusion world)
- `env.trigger_state` — `{trigger_name: t_fire_µs or None}` for this particle
- `env.total_damping_rate` — γ summed across all physics (used by Langevin)
- `env.rng` — `numpy.random.Generator` seeded per particle

The integrator special-cases `damping_rate`: it sums all contributions and applies them via the exact factor `v ← exp(−γ·dt)·v` after each accepted step (better than `accel = −γv` for large dt).

Built-in physics:
- `Electrostatic()`  — `q·E/m` from `env.field`
- `Gravity(g_mm_us2=9.81e-9, axis="-y")`
- `EpsteinDrag(pressure_pa, temperature_k, gas_mass_amu, pressure_ramp=None, scale=1.0)` — `pressure_ramp = {"trigger": "release", "p_final_pa": 100.0, "duration_us": 5e5}` triggers a linear pressure ramp starting at the named trigger's fire time
- `Langevin(temperature_k)` — FDT noise scaled to `env.total_damping_rate`

---

## CLI reference

```
python run.py                  # full pipeline: refine → fly → animate → visualize
  --run N                      # writes trajectories_N.csv, schedule_N.json
  --workers N                  # default = all CPUs
  --refine                     # force re-refine (voxelize + Laplace solve)
  --no-fly                     # reuse an existing trajectories_N.csv
  --no-animate                 # skip the matplotlib animation
  --no-visualize               # skip the PyVista 3D viewer

python -m trapsim.refine [--force-voxelize]            # refine only
python -m trapsim.fly --run 2 --workers 4              # fly only
python -m trapsim.viz.animate --traj trajectories_1.csv --schedule schedule_1.json
python -m trapsim.viz.visualize --animation flythrough.mp4
python -m trapsim.viz.plot_field --slice y=19 --time 50000 --quantity E
```

---

## Output file formats

### `paulTrap.pa<N>` — SIMION potential array (binary)

56-byte header (`flags`, `scale_ref`, `NX`, `NY`, `NZ`, `dx`) followed by `NX·NY·NZ` float64 in `[k][j][i]` order (z slowest, x fastest). Electrode-surface voxels encoded with sign-bit or `>1.5·scale_ref`. Free-space voxels store φ/scale_ref where φ is the unit-drive potential. Read via `trapsim.io.pa.read_pa(path)`.

### `trajectories_<N>.csv`

```
ion_id,t_us,x_mm,y_mm,z_mm
1,0.0000,0.00000,19.00000,-98.04495
1,456.0000,-0.00463,19.02678,-98.04464
…
```

Coordinates are in Fusion-world mm. Recorded once per `record_stride` accepted steps, plus the start and end points.

### `schedule_<N>.json`

```json
{
  "main": {
    "time_us": [...],
    "dc": {"endcap_load_U": [...], ...},
    "rf": {"rf_loading": {"amplitude": [...], "frequency_hz": 2000, "phase_deg": 0}}
  },
  "triggers": [
    {"name": "drop_load_endcap", "axis": "z", "threshold_mm": -83.52,
     "schedule": {"time_us": [...], "dc": {...}}}
  ]
}
```

A serialised copy of the schedule used for this run. Used by `animate.py` to plot the voltage timeline; useful for reproducing or analysing the run later.

---

## Geometry workflow (Autodesk Fusion → STL)

Each rigid body that you want as an independent voltage source or dielectric needs its own binary-STL export:

1. In the canvas, right-click the component → **Find in Browser**.
2. Expand to the **Body**, right-click → **Isolate**.
3. Right-click the top-level assembly → **Save As Mesh**.
4. **Format:** STL (Binary), **Unit Type:** Millimeter, **Structure:** One File, **Refinement:** High.
5. Save to `stl/<body_name>.stl`.

Bodies wired together (e.g. four rods on the same RF supply) get listed under the same electrode `name` in `geometry.yaml`. The voxelizer takes the union of their meshes.

The simulation volume (`grid.bounds_mm`) must enclose every body. The grid spacing (`grid.dx_mm`) sets the trade-off between accuracy and memory: at 0.5 mm a 131×91×855 grid uses ~80 MB per electrode.

---

## Repository layout

```
geometry.yaml          object inventory + grid (edit for a new geometry)
experiment.py          particles, schedule, triggers, physics  (edit for a new run)
run.py                 pipeline entry point
stl/                   STL bodies

trapsim/               the geometry-agnostic package
  config.py            geometry.yaml loader + validation
  voxelize.py          STL → solver masks
  refine.py            orchestrates voxelize + C++ solve
  fly.py               particle integrator
  schedule.py          schedule + trigger resolution
  physics/             pluggable physics modules
    base.py electrostatic.py gravity.py epstein_drag.py langevin.py
  io/                  PA / trajectory / schedule readers and writers
  viz/                 animate, visualize, plot_field
  run.py               full-pipeline orchestrator

solver/                C++ Laplace solver (laplace.cpp + Makefile)
legacy/                pre-refactor scripts (SIMION + early Python pipeline)
```

---

## Performance

Reference: M-series Mac, 14 worker processes, 20 particles, 2·10⁵ µs simulated.

| Step | Wall time |
|------|-----------|
| Refine (10 electrodes, once per geometry edit) | ~3 minutes |
| Fly (20 particles) | ~5 seconds |
| Animate + visualize | a few seconds |

Refine is the bottleneck; fly is fast because the Dormand-Prince adaptive stepper takes ~10× larger steps in the smooth field regions of the trap centre than a fixed dt would allow.
