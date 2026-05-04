# RF Guide SIMION Simulation

SIMION 8.2 particle-trajectory simulation of 166 nm silica nanospheres through a linear Paul trap RF guide, with a perpendicular Paul trap at the exit for retrapping. The geometry is designed for loading an optical trap across a gate-valve gap.

## Repository layout

```
paulTrap.gem                 SIMION geometry definition (electrodes + PA dimensions)
paulTrap.lua                 User program: RF fast-adjust, Epstein drag, gravity,
                             trajectory recording
generate_voltages.py         Write voltage schedule CSV for both traps
generate_dielectric_pa.py    Build the dielectric permittivity array for the glass lenses
refine_with_dielectric.lua   Re-refine electrode PAs with dielectric effects (run in SIMION)
animate.py                   2-panel animation: trajectory + voltage timeline
visualize.py                 Interactive 3-D view of geometry + trajectories (PyVista)
plot_field.py                Plot electric field cross-sections from PA binary files
sanity_check.py              Verify STL seed points and geometry bounds before refining
```

STL files for every electrode and dielectric body must be present in this directory. They are not tracked by git (listed in `.gitignore`).

---

## 1. Coordinate system

All STL files are exported in **Fusion world coordinates** (mm). The GEM file uses a `locate(tx, ty, tz)` block that converts Fusion coordinates to SIMION GEM coordinates:

```
GEM x = Fusion X + 25
GEM y = Fusion Y +  8
GEM z = Fusion Z + 132
```

Seed points inside the GEM file are therefore written directly as Fusion coordinates. The Python scripts use the same convention (`GEM_OFF = [-25, -8, -132]`).

---

## 2. Exporting geometry from Autodesk Fusion

Each electrode body must be exported as a separate binary STL file in millimetres. Repeat the following for every body:

1. In the canvas, right-click the component and select **Find in Browser**. This highlights the component in the browser panel.
2. In the browser, expand the component until the **Body** is visible. Right-click the body and select **Isolate** so that only this body is shown.
3. Right-click the **top-level assembly** item at the top of the browser and select **Save As Mesh**.
4. In the dialog, set the following options:
   - **Format:** STL (Binary)
   - **Unit Type:** Millimeter
   - **Structure:** One File
   - **Refinement:** High
5. Click **OK** and save the file to this project directory using the filename referenced in `paulTrap.gem` (e.g. `rod_P1_L1.stl`).
6. Turn off Isolate before exporting the next body (right-click the top-level assembly → **Isolate** again to toggle it off).

> The STL origin is the Fusion world origin, so no manual coordinate adjustments are needed — SIMION reads the Fusion coordinates directly via the `locate` block in the GEM.

---

## 3. Defining geometry in the GEM file

`paulTrap.gem` defines the potential-array dimensions and all conductive electrodes. Open it in a text editor to make changes.

### Potential array

```lua
pa_define{59*mm, 45*mm, 427*mm, 'planar', dx=0.5, surface='fractional',
          filename="paulTrap.pa0"}
```

- Dimensions must enclose all electrode geometry with at least ~5 mm clearance on every side.
- `dx=0.5` is used for fast testing. Change to `dx=0.15` for production runs (increases file size ~37×).
- `surface='fractional'` (sub-grid surface enhancement) is **incompatible with SIMION 8.2's dielectric solver** and is therefore omitted. If you need higher electrode-surface accuracy without dielectrics, you can add `surface='fractional'` back, but you will not be able to run `refine_with_dielectric.lua` afterwards.
- Update the three dimensions and the `locate` offsets whenever the geometry changes significantly.

### locate block

```lua
locate(25, 8, 132) {
  -- seed points in Fusion world coordinates (mm)
  e(1) {
    stl(D.."rod_P1_L1.stl",  -2.1082, 21.1582, -20.6470)
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
rod_P1_L1.stl endcap_L.stl   # replace with the files you need
```

### Electrode numbering

| # | Name | Type | Drive |
|---|------|------|-------|
| 1 | Rod pair 1, left (P1_L1 + P1_L2) | RF | +RF |
| 2 | Rod pair 2, left (P2_L1 + P2_L2) | RF | −RF |
| 3 | Left end cap | DC | `V_endcap` |
| 4 | Rod pair 1, right (P1_R1 + P1_R2) | RF | +RF |
| 5 | Rod pair 2, right (P2_R1 + P2_R2) | RF | −RF |
| 6 | Ring electrode, left | DC | `V_ring_L` |
| 7 | Ring electrode, right | DC | `V_ring_R` |
| 8 | Right end cap | DC | `V_endcap_R` |
| 9 | Perp-trap rod pair 1 (TL + BR) | RF | +RF2 |
| 10 | Perp-trap rod pair 2 (TR + BL) | RF | −RF2 |
| 11 | Trapping lens holder | DC | `V_trap_lens` |
| 12 | Collection lens holder | DC | `V_coll_lens` |

Electrodes 13 and 14 (glass lenses) are **not defined in the GEM**. They are dielectric, not conductive — see Section 5.

After any geometry change, run the sanity check before refining:

```bash
~/.venvs/mesh/bin/python3 sanity_check.py
```

---

## 4. Standard refinement (no dielectrics)

Open SIMION, load `paulTrap.gem`, and click **Refine**. SIMION creates one potential array per electrode: `paulTrap.pa0` (geometry) and `paulTrap.pa1` through `paulTrap.pa12`.

If the dielectric lenses are not needed for a particular run, stop here and proceed to Section 6.

---

## 5. Dielectric refinement (glass lenses)

SIMION's dielectric solver requires a **separate permittivity array** alongside the standard electric PAs. The lenses must be absent from the electric PA (they are already omitted from `paulTrap.gem`) and encoded only in this permittivity array.

### Step 1 — Generate the dielectric PA

```bash
~/.venvs/mesh/bin/python3 generate_dielectric_pa.py
```

This creates `paulTrap-dielectric.pa` (≈ 69 MB). The file assigns ε_r = `EPSILON_GLASS` (defined at the top of the script, currently 3.0 — update it to the correct value for your glass) to every grid cell whose centre falls inside a lens mesh, and ε_r = 1.0 elsewhere.

**Run this step whenever `EPSILON_GLASS` changes or when the lens STL files are updated.**

### Step 2 — Re-refine electrode PAs with the dielectric

From within SIMION, run `refine_with_dielectric.lua` (File → Run Lua Script, or from the SIMION command line with `--nogui lua`). This re-refines `pa1` through `pa12` incorporating the lens permittivity. The fast-adjust system will then work normally.

**Run this step after every standard Refine (Step 1 of Section 4) and after every dielectric PA regeneration.**

---

## 6. Creating a voltage schedule

Edit `generate_voltages.py` to define the time-varying voltages for both traps, then run it:

```bash
~/.venvs/nano/bin/python3 generate_voltages.py
```

This produces `voltages_1.csv` (or `voltages_N.csv` — set `OUT_NUMBER` at the top of the script) and opens a preview plot showing all channels. The CSV format is:

```
# f_RF_Hz=<value>
# f_RF2_Hz=<value>
time_us, V_endcap, V_endcap_R, V_ring_L, V_ring_R, V_RF, V_RF2, V_trap_lens, V_coll_lens
```

SIMION interpolates voltages linearly between rows. For a sharp step, place two rows at the same time (or 0.1 µs apart).

### Key parameters to set

| Parameter | Location | Description |
|-----------|----------|-------------|
| `f_RF` | top of script | Main trap RF carrier frequency (Hz) |
| `f_RF2` | top of script | Perp-trap RF carrier frequency (Hz) — **PLACEHOLDER** |
| `V_RF` | body of script | Main trap RF zero-to-peak amplitude vs time |
| `V_RF2` | body of script | Perp-trap RF amplitude — **PLACEHOLDER** (currently 0 V) |
| `V_trap_lens` | body of script | Trapping lens holder DC bias — **PLACEHOLDER** |
| `V_coll_lens` | body of script | Collection lens holder DC bias — **PLACEHOLDER** |

---

## 7. Running a simulation in SIMION

1. Load the workbench (`.iob` file) in SIMION.
2. In the **Variables** panel, set:
   - `voltage_file_number` — integer N matching the `voltages_N.csv` file to use
   - `run_number` — integer appended to the output trajectory filename
   - `drag_scale` — set to 0 to disable Epstein drag (useful for field testing); 1 for full physics
   - `v_stop` — ion is terminated when its speed falls below this value (mm/µs); set to 0 to disable
   - `record_stride` — write one trajectory row every N time steps; larger = smaller output file
3. Define ions in the **Particles** panel: mass ≈ 3.19 × 10⁶ u (for 166 nm silica sphere, 100 electron charges), charge = 100.
4. Click **Fly'm**.

Trajectories are written to `trajectories_N.csv` in the project directory (Fusion world coordinates). Voltages are read from `voltages_N.csv`.

### Physics implemented in `paulTrap.lua`

- **RF fast-adjust**: electrodes 1, 2, 4, 5 driven analytically at `f_RF`; electrodes 9, 10 at `f_RF2`.
- **DC schedule**: electrodes 3, 6, 7, 8, 11, 12 interpolated from the voltage CSV.
- **Epstein drag** (free-molecular regime, Kn ≈ 800 at 1 mbar): `β = (8π/3) r² P / c̄`; damping rate `γ = β / m`.
- **Gravity** in −Y (9.81 × 10⁻⁹ mm/µs²).
- **Finite-timestep drag correction** to avoid underestimating drag over long steps.

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
~/.venvs/nano/bin/python3 plot_field.py --elec 8   # right end cap
~/.venvs/nano/bin/python3 plot_field.py             # all DC electrodes (3, 6, 7, 8)
```

Parses the SIMION PA binary files and produces a two-panel figure: an X-Z cross-section of the potential with field vectors, and an axial E_z profile showing field screening by the RF rods. Output saved as `field_N.png`.

---

## 9. Python environments

| Environment | Purpose |
|-------------|---------|
| `~/.venvs/nano` | Main simulation scripts (numpy, matplotlib, PyVista) |
| `~/.venvs/mesh` | Mesh processing — trimesh, rtree (point-in-mesh for dielectric PA generation) |

---

## 10. Open placeholders

The following values are set to approximate defaults and should be updated before production runs:

- **`EPSILON_GLASS`** in `generate_dielectric_pa.py` (currently 3.0). Typical values: fused silica ≈ 3.82 (DC), N-BK7 ≈ 7.1 (DC). For optical-frequency fields use n² instead (fused silica at 1064 nm: n ≈ 1.46, ε_r ≈ 2.13).
- **`f_RF2`** in `generate_voltages.py` — RF frequency for the perpendicular trap. Compute from the trap r₀ and the target stability parameter q < 0.908.
- **`V_RF2`** in `generate_voltages.py` — RF amplitude for the perpendicular trap (currently 0 V; the trap is inactive).
- **`V_trap_lens`** and **`V_coll_lens`** in `generate_voltages.py` — DC biases for the perp-trap axial confinement (currently 0 V).
- **`dx` in `paulTrap.gem`** — change from 0.5 mm to 0.15 mm for production-quality field calculations (requires re-running all refinement steps).
- **Ion start position** in the SIMION `.iob` file — GEM coordinates will need updating after any geometry change that shifts the `locate` offsets.
