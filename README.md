# RF Guide SIMION Simulation

SIMION 8.2 particle-trajectory simulation of 166 nm silica nanospheres through a linear Paul trap RF guide, with a parallel-axis Paul trap at the downstream end that surrounds the optical-trap lenses. The geometry is designed for loading a tightly-focused 1064 nm optical trap from a remote loading region across a gate-valve gap.

## Repository layout

```
paulTrap.gem                 SIMION geometry definition (10 electrodes + PA dimensions)
paulTrap.lua                 User program: RF fast-adjust, Langevin dynamics (Epstein
                             drag + thermal noise), gravity, trajectory recording
trap_config.lua              Gas, particle, drag, trigger, and start-position config
generate_voltages.py         Write voltage schedule CSV for all 10 electrodes
generate_dielectric_pa.py    Build the dielectric permittivity array for the lenses
                             and the single uniform lens holder
refine_with_dielectric.lua   Re-refine electrode PAs with dielectric effects (run in SIMION)
animate.py                   2-panel animation: trajectory + voltage timeline
visualize.py                 Interactive 3-D view of geometry + trajectories (PyVista)
plot_field.py                Plot electric field cross-sections from PA binary files
sanity_check.py              Verify STL seed points and geometry bounds before refining
```

STL files for every electrode and dielectric body must be present in this directory. They are not tracked by git (listed in `.gitignore`).

---

## 1. Coordinate system

All STL files are exported in **Fusion world coordinates** (mm). The GEM file uses a `locate(tx, ty, tz)` block to convert Fusion coordinates to SIMION GEM coordinates:

```
GEM x = Fusion X + tx
GEM y = Fusion Y + ty
GEM z = Fusion Z + tz
```

The Python scripts use the same convention via `GEM_OFF = (-tx, -ty, -tz)`.

### Labelling convention

Looking down the RF guide axis:

| Letter | Meaning |
|--------|---------|
| L      | −x      |
| R      | +x      |
| T      | +y      |
| B      | −y      |
| U      | +z (upstream)   |
| D      | −z (downstream) |

Rods are named with a set number (1–3) and a TL/TR/BL/BR suffix. Endcaps are named with `load`/`optical` and a U/D suffix.

---

## 2. Exporting geometry from Autodesk Fusion

Each body must be exported as a separate binary STL file in millimetres. Repeat the following for every body:

1. In the canvas, right-click the component and select **Find in Browser**. This highlights the component in the browser panel.
2. In the browser, expand the component until the **Body** is visible. Right-click the body and select **Isolate** so that only this body is shown.
3. Right-click the **top-level assembly** item at the top of the browser and select **Save As Mesh**.
4. In the dialog, set the following options:
   - **Format:** STL (Binary)
   - **Unit Type:** Millimeter
   - **Structure:** One File
   - **Refinement:** High
5. Click **OK** and save the file to this project directory using the filename referenced in `paulTrap.gem` (e.g. `rod_1_TL.stl`).
6. Turn off Isolate before exporting the next body.

> The STL origin is the Fusion world origin, so no manual coordinate adjustments are needed — SIMION reads Fusion coordinates directly via the `locate` block in the GEM.

### STL filename inventory

Rod sets (12 total):

```
rod_1_TL.stl  rod_1_TR.stl  rod_1_BL.stl  rod_1_BR.stl    (set 1, loading Paul trap)
rod_2_TL.stl  rod_2_TR.stl  rod_2_BL.stl  rod_2_BR.stl    (set 2, RF guide after gate valve)
rod_3_TL.stl  rod_3_TR.stl  rod_3_BL.stl  rod_3_BR.stl    (set 3, optical Paul trap, wider)
```

Endcaps (4 total):

```
endcap_load_U.stl     endcap_load_D.stl
endcap_optical_U.stl  endcap_optical_D.stl
```

Dielectric volumes (3 total):

```
trapping_lens.stl    collection_lens.stl    lens_holder.stl
```

---

## 3. Defining geometry in the GEM file

`paulTrap.gem` defines the potential-array dimensions and all conductive electrodes. Open it in a text editor to make changes.

### Potential array

```lua
pa_define{Nx*mm, Ny*mm, Nz*mm, 'planar', dx=0.5,
          filename="paulTrap.pa0", surface="fractional"}
```

- Dimensions must enclose all electrode geometry with at least ~5 mm clearance on every side.
- `dx=0.5` is used for fast testing. Change to `dx=0.15` for production runs.
- `surface='fractional'` (sub-grid surface enhancement) is **incompatible with SIMION 8.2's dielectric solver** and must be omitted when running `refine_with_dielectric.lua` afterwards.
- Update the three dimensions and the `locate` offsets whenever the geometry changes significantly.

### locate block

```lua
locate(tx, ty, tz) {
  e(1) {
    stl(D.."rod_1_TL.stl",  x_seed, y_seed, z_seed)
    ...
  }
}
```

Each `stl()` call takes the filename and a **seed point** — any point guaranteed to lie inside the solid body. The centroid of the mesh is a reliable choice. To compute centroids from the command line:

```bash
~/.venvs/mesh/bin/python3 - <<'EOF'
import numpy as np, struct, sys
def centroid(path):
    with open(path,'rb') as f:
        f.read(80); n=struct.unpack('<I',f.read(4))[0]
        v=[]
        for _ in range(n):
            f.read(12); v+=[struct.unpack('<fff',f.read(12)) for _ in range(3)]; f.read(2)
    return np.array(v).mean(axis=0)
for p in sys.argv[1:]: print(p, centroid(p))
EOF
rod_1_TL.stl rod_3_BR.stl endcap_load_U.stl   # replace with the files you need
```

### Electrode numbering

| # | Name | Type | Drive |
|---|------|------|-------|
| 1 | Sets 1+2 +RF  (rod_1_TL, rod_1_BR, rod_2_TL, rod_2_BR) | RF  | +V_RF·cos(ω_RF t) |
| 2 | Sets 1+2 −RF  (rod_1_TR, rod_1_BL, rod_2_TR, rod_2_BL) | RF  | −V_RF·cos(ω_RF t) |
| 3 | endcap_load_U                                          | DC  | `V_endcap_load_U` |
| 4 | endcap_load_D                                          | DC  | `V_endcap_load_D` |
| 5 | rod_3_TL                                               | RF+DC | +V_RF3·cos(ω_RF3 t) + `V_dc_3_TL` |
| 6 | rod_3_TR                                               | RF+DC | −V_RF3·cos(ω_RF3 t) + `V_dc_3_TR` |
| 7 | rod_3_BL                                               | RF+DC | −V_RF3·cos(ω_RF3 t) + `V_dc_3_BL` |
| 8 | rod_3_BR                                               | RF+DC | +V_RF3·cos(ω_RF3 t) + `V_dc_3_BR` |
| 9 | endcap_optical_U                                       | DC  | `V_endcap_optical_U` (triggered) |
| 10| endcap_optical_D                                       | DC  | `V_endcap_optical_D` (triggered) |

`trapping_lens.stl`, `collection_lens.stl`, and `lens_holder.stl` are **not defined in the GEM**. They are dielectric, not conductive — see Section 5.

After any geometry change, run the sanity check before refining:

```bash
~/.venvs/mesh/bin/python3 sanity_check.py
```

---

## 4. Standard refinement (no dielectrics)

Open SIMION, load `paulTrap.gem`, and click **Refine**. SIMION creates one potential array per electrode: `paulTrap.pa0` (geometry) and `paulTrap.pa1` through `paulTrap.pa10`.

If the dielectric volumes are not needed for a particular run, stop here and proceed to Section 6.

---

## 5. Dielectric refinement (lenses + lens holder)

### How SIMION handles dielectrics

SIMION's dielectric solver requires a **separate permittivity array** alongside the standard electric PAs. The dielectric bodies must be absent from the electric PA (they are already omitted from `paulTrap.gem`) and encoded only in this permittivity array.

### Why the standard refine workflow does not work here

SIMION 8.2's dielectric solver is incompatible with `surface='fractional'`. Without fractional surfaces, SIMION's GEM-to-`pa#` voxelizer marks surface voxels then flood-fills from a seed point. For meshes whose surfaces graze the grid at shallow angles, sub-grid gaps in the marked surface let the flood-fill leak into surrounding free space, marking large regions of the volume as electrode. Repairing the STLs does not fix this — the issue is purely geometric.

`rasterize_pa.py` bypasses SIMION's voxelizer entirely. For every grid node it runs `trimesh.contains()` (embreex-backed) against each electrode's STL and stamps the appropriate `pa#` marker. The result has no surface tracking, no flood-fill, and no leak paths. Mesh quality stops mattering as long as `contains()` returns the right answer. The resulting `pa#` can then be loaded back into SIMION, which runs only its relaxation solver on it — no re-voxelization.

Use this workflow only when you need dielectric effects. For everything else, the standard fractional-surface workflow (Section 4) is faster to iterate on and produces sharper electrode surfaces.

### Workflow

1. **In SIMION:** load `paulTrap.gem`, click **Refine**. Pick any surface mode; we're about to overwrite the result. This step exists only to set up `paulTrap.pa#` at the grid `dx` you want.
2. **In SIMION:** `File → Save PA` (`paulTrap.pa#`) and then `File → Close PA` (or `Unload`). SIMION caches PA contents in memory; if you don't unload it, the next step's overwrite on disk won't be picked up.
3. **Shell:** run the rasterizer.
   ```bash
   ~/.venvs/mesh/bin/python3 rasterize_pa.py
   ```
   This writes `paulTrap_rasterized.pa#` (the original `paulTrap.pa#` is untouched).
4. **Shell:** swap the rasterized file in for the SIMION-generated one.
   ```bash
   cp paulTrap.pa#               paulTrap.pa#.simion-backup
   cp paulTrap_rasterized.pa#    paulTrap.pa#
   ```
5. **In SIMION:** `File → Load PA → paulTrap.pa#` (reload). Then click **Refine** again. Because `paulTrap.pa#` already contains electrode markers, SIMION skips voxelization and just runs the relaxation solver, producing leak-free `paulTrap.pa1` through `paulTrap.pa10`.
6. **Shell:** build the dielectric permittivity array.
   ```bash
   ~/.venvs/mesh/bin/python3 generate_dielectric_pa.py
   ```
   This creates `paulTrap-dielectric.pa`, assigning ε_r = `EPSILON_GLASS` (defined at the top of the script — currently 3.0, a fused-silica-and-PEEK ballpark) to every grid cell whose centre falls inside any of `trapping_lens.stl`, `collection_lens.stl`, or `lens_holder.stl`. All other cells are ε_r = 1.0. Re-run this whenever `EPSILON_GLASS` changes or the dielectric STL files are updated.
7. **In SIMION:** run `refine_with_dielectric.lua` (File → Run Lua Script, or from the SIMION command line with `--nogui lua`). This re-refines `pa1` through `pa10` incorporating the dielectric permittivity. Fast-adjust then works as normal. Re-run this after every standard Refine and after every regeneration of `paulTrap-dielectric.pa`.

### Verifying

```bash
~/.venvs/mesh/bin/python3 rasterize_pa.py --verify
```

`--verify` re-reads the source pa# and reports per-electrode cell counts for both. The rasterized counts should be **pair-symmetric** by construction (electrodes 1↔2, 3↔4, 9↔10 should match exactly; rods 5–8 should be close). Large positive diffs in the source column or broken pair-symmetry indicate the source pa# is leaking — that's the situation this workflow exists to fix.

### Going back to the standard workflow

```bash
cp paulTrap.pa#.simion-backup paulTrap.pa#
```

Then in SIMION: unload the PA, reload it from disk, and Refine normally (the fractional-surface workflow from Section 4).

---

## 6. Creating a voltage schedule

Edit `generate_voltages.py` to define the time-varying voltages for all 10 electrodes, then run it:

```bash
~/.venvs/nano/bin/python3 generate_voltages.py
```

This produces `voltages_1.csv` (or `voltages_N.csv` — pass `--out N`) and opens a preview plot showing all channels. The CSV contains two interleaved row types sharing one header:

```
# f_RF_Hz=<value>
# f_RF3_Hz=<value>
time_us, V_RF, V_RF3,
         V_endcap_load_U, V_endcap_load_D,
         V_dc_3_TL, V_dc_3_TR, V_dc_3_BL, V_dc_3_BR,
         V_endcap_optical_U, V_endcap_optical_D,
         time_trig_us, V_endcap_optical_U_trig, V_endcap_optical_D_trig
```

**Main-schedule rows** (`time_us` present, `time_trig_us` empty): drive all 10 electrodes on the coarse simulation time axis. The `V_endcap_optical_U/D` columns are a fallback; if the trig columns are loaded they take priority.

**Post-trigger rows** (`time_trig_us` present, `time_us` empty): define what electrodes 9 and 10 do after the trigger fires, as a function of *time since trigger fire*. Use a finer time step here (`dt_trig` in `generate_voltages.py`) to capture short pulses accurately. Once the schedule runs past its last row it clamps to the final value.

SIMION interpolates voltages linearly between rows within each time axis. For a sharp step, place two rows at the same time (or 0.1 µs apart).

### Key parameters to set

| Parameter | Description |
|-----------|-------------|
| `f_RF`     | Sets 1+2 RF carrier frequency (Hz) |
| `f_RF3`    | Set 3 RF carrier frequency (Hz) — **PLACEHOLDER** |
| `V_RF`     | Sets 1+2 RF zero-to-peak amplitude vs time |
| `V_RF3`    | Set 3 RF amplitude — **PLACEHOLDER** (currently 0 V) |
| `V_dc_3_TL/TR/BL/BR` | Per-rod DC trims on set 3 (shift equilibrium toward optical focus) |
| `V_endcap_load_U/D`  | Load Paul trap endcap DC biases |
| `V_endcap_optical_U/D` | Optical Paul trap endcap DC biases (gated by trigger) |

Set 3 has 4 fully-independent SIMION electrodes; equal-and-opposite DC trims on opposite rods shift the equilibrium laterally, while common-mode trims raise the mean rod potential.

---

## 7. Running a simulation in SIMION

### Workbench (`paulTrap.iob`) setup — one-time, must be done in the GUI

The `.iob` file is SIMION-specific and can only be edited inside the SIMION GUI. Everything else in this project lives in plain text and the workbench is the only handoff point. You set it up once and then never touch it.

The workbench must reference:

- The PA file (`paulTrap.pa#`, and the refined `pa1..pa10` it produces). The PA instance's position/orientation in the workbench should be the identity transform — `locate(...)` in the GEM already shifts everything into the right place. The workbench must have **10 electrode slots** matching the new GEM.
- The FLY2 file (`paulTrap.fly2`).
- The user program (`paulTrap.lua`).
- TOF cutoff and trajectory quality. The lua does not set these; if the TOF cutoff is shorter than you need, trajectories are truncated.
- Initial values for the adjustables `voltage_file_number` and `run_number` (overridden at runtime by `SIMION_VOL_FILE` / `SIMION_RUN_NUM` env vars in `run_simulation.sh`).

### How particle properties are set

Everything about the particles — mass, charge, start position, initial velocity, count, gas, drag — is in `trap_config.lua`. `paulTrap.lua:325-360` overrides the workbench/fly2 values for `ion_mass`, `ion_charge`, `ion_px_mm/py_mm/pz_mm`, and `ion_vx_mm/vy_mm/vz_mm` per-ion before flight starts. Any ions beyond `particles.n` are splatted immediately (`paulTrap.lua:332-334, 450-452`) at no cost.

The **only** field in `paulTrap.fly2` that's actually meaningful is the ion count `n`, which has to be ≥ `particles.n` from `trap_config.lua`. The mass and charge are placeholders.

> **Gotcha** — `paulTrap.fly2` `position = vector(...)` must be a valid free-space point inside the PA volume, even though `segment.initialize()` will overwrite it immediately. SIMION pre-validates the ion against the PA *before* calling `initialize()`, so a placeholder like `vector(0, 0, 0)` either lands outside the grid or inside an electrode marker and the headless fly silently fails. The current value `vector(25, 27.2, 52.5)` corresponds to the loading-trap centre in workbench coords and works fine; if you change it, pick a point you know is in free space at the current `pa_define`/`locate` offsets.

### Running

Headless (recommended) — `run_simulation.sh` regenerates voltages, sets the env vars, launches a `--nogui fly` through CrossOver, and animates the result.

```bash
./run_simulation.sh --vol 1 --run 1
```

Interactive (GUI) — load `paulTrap.iob`, set `voltage_file_number` and `run_number` in the Variables panel, edit `trap_config.lua` for the particle/gas/trigger setup, click **Fly'm**.

Trajectories are written to `trajectories_N.csv` in the project directory (Fusion world coordinates). Voltages are read from `voltages_N.csv`.

### Physics implemented in `paulTrap.lua`

- **RF fast-adjust**: electrodes 1, 2 driven analytically at `f_RF`; electrodes 5–8 at `f_RF3` (diagonal-pair phasing: TL+BR get +V_RF3, TR+BL get −V_RF3).
- **DC schedule**: electrodes 3, 4, 5–8 (DC trim part), 9, 10 interpolated from the voltage CSV.
- **Triggers**: only electrodes 9, 10 (optical endcaps) are gated; they sit at 0 V until the ion's Fusion-z crosses the threshold in `trap_config.lua`, then follow the schedule from t=0 of trigger-fire.
- **Langevin dynamics** (full, not just drag): `accel_adjust` applies the deterministic half of the Ornstein-Uhlenbeck propagator — Epstein drag with finite-timestep correction factor `(1 − exp(−γ·dt))/(γ·dt)`. `other_actions` adds the matching stochastic velocity kick each time step, drawn from σ² = (k_B T/m)·(1 − exp(−2γ·dt)). Together these satisfy the fluctuation-dissipation theorem, so particles equilibrate to a Maxwell-Boltzmann distribution at the gas temperature. Setting `langevin_noise = false` in `trap_config.lua` reverts to deterministic drag only. Setting `drag_scale = 0` disables both halves.
  - **Epstein drag** (free-molecular regime, Kn ≈ 800 at 1 mbar): `β = (8π/3) r² P / c̄`; damping rate `γ = β / m`. Linear in pressure, so γ is recomputed each timestep from the current pressure.
  - For our 166 nm silica sphere at 293 K, equilibrium per-axis `v_rms = √(k_B T/m) ≈ 2.8 × 10⁻⁵ mm/µs`, well above the `v_stop_mm_us = 1e-5` floor.
- **Triggered pressure ramp**: optional `pressure_ramp` block in `trap_config.lua` models a solenoid valve opening when a named trigger fires (per-ion). Pressure ramps linearly from `pressure_pa` to `P_final_pa` over `duration_us`; before the trigger fires the pressure stays at the baseline.
- **Gravity** in −Y (9.81 × 10⁻⁹ mm/µs²).

---

## 8. Viewing results

### Trajectory animation

```bash
~/.venvs/nano/bin/python3 animate.py
~/.venvs/nano/bin/python3 animate.py --traj trajectories_2.csv --volt voltages_2.csv
~/.venvs/nano/bin/python3 animate.py --speed 500   # µs of sim time per wall-second
~/.venvs/nano/bin/python3 animate.py --save out.mp4
```

Shows a side view of ion trails (Z horizontal, Y vertical) alongside the voltage timeline, with a cursor tracking the current frame.

### 3-D geometry and trajectories

```bash
~/.venvs/nano/bin/python3 visualize.py
~/.venvs/nano/bin/python3 visualize.py --traj trajectories_2.csv
~/.venvs/nano/bin/python3 visualize.py --screenshot out.png
```

Loads all STL files and overlays trajectory paths coloured by time-of-flight. Requires PyVista.

### Electric field cross-sections

```bash
~/.venvs/nano/bin/python3 plot_field.py --elec 3   # load endcap U
~/.venvs/nano/bin/python3 plot_field.py --elec 5   # set-3 TL rod
~/.venvs/nano/bin/python3 plot_field.py            # all DC endcaps (3, 4, 9, 10)
```

Parses the SIMION PA binary files and produces a two-panel figure: an X-Z cross-section of the potential with field vectors, and an axial E_z profile. Output saved as `field_N.png`.

---

## 9. Python environments

| Environment | Purpose |
|-------------|---------|
| `~/.venvs/nano` | Main simulation scripts (numpy, matplotlib, PyVista) |
| `~/.venvs/mesh` | Mesh processing — `trimesh`, `rtree`, `embreex` (point-in-mesh for dielectric PA generation) |

> **embreex is important.** Without it, `trimesh.contains()` falls back to a numpy/rtree path that builds an `(n_rays × n_triangles)` dense scratch matrix per call, which can use tens of GB for the lens holder at production resolution. With `embreex` installed, peak memory drops to essentially just the epsilon array itself. Install with `~/.venvs/mesh/bin/pip install embreex`.

---

## 10. Open placeholders

The following values are set to approximate defaults and should be updated before production runs:

- **`pa_define` extents and `locate(tx, ty, tz)` offsets** in `paulTrap.gem` — must reflect the new Fusion bounding box after re-export.
- **Every `stl()` seed point** in `paulTrap.gem` — currently `(0, 0, 0)` placeholders; replace with each STL's actual Fusion centroid.
- **`EPSILON_GLASS`** in `generate_dielectric_pa.py` (currently 3.0). Typical values: fused silica ≈ 3.82 (DC), PEEK ≈ 3.2.
- **`f_RF3`** and **`V_RF3`** in `generate_voltages.py` — RF frequency and amplitude for the optical Paul trap (set 3). Compute from set 3 r₀ and a target stability parameter q < 0.908.
- **Trigger z_mm** in `trap_config.lua` — Fusion-Z threshold at which the optical endcaps fire.
- **Particle start position** in `trap_config.lua` — should be inside the loading Paul trap between endcap_load_U and endcap_load_D.
- **`dx` in `paulTrap.gem`** — change from 0.5 mm to 0.15 mm for production-quality field calculations (requires re-running all refinement steps).
- **Ion start position** in the SIMION `.iob` file — GEM coordinates will need updating after any geometry change that shifts the `locate` offsets.
- **`MAIN_IX`, `MAIN_IY`, `GEM_OFF`, `NX`, `NY`, `NZ`** in `plot_field.py` — update to match new PA geometry.
- **`GEO` dict** in `animate.py` — Z extents of each rod set and endcap positions are placeholders until the Fusion bounding box is known.
- **`PA_X/Y/Z` and seed coords** in `sanity_check.py` — same situation.
- **`stability_map.py`** — designed for the old perpendicular-trap geometry; needs an analysis-layer rewrite to sweep along Z and reflect the new optical-trap physics (PA loading has been updated to the new electrode numbers; visualisation is still legacy).
