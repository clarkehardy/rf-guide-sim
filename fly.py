"""
fly.py  –  Pure-Python particle integrator for the linear Paul trap simulation.

Replaces SIMION's fly step and paulTrap.lua.  Reads pre-computed unit-potential
PA arrays (paulTrap.pa1–pa10), a voltage schedule CSV, and the trap_config.py
configuration module, then integrates trajectories with adaptive RK4/5
(Dormand-Prince) plus a split Langevin step.

Usage
-----
    python fly.py [--vol N] [--run N] [--workers N]

Defaults: --vol 1, --run 1, --workers = os.cpu_count().

Output: trajectories_N.csv with columns  ion,time_us,x_mm,y_mm,z_mm
        (coordinates in Fusion world; every record_stride accepted steps).
"""

import argparse
import importlib.util
import math
import multiprocessing
import multiprocessing.shared_memory
import os
import re
import struct
import sys
import time

import numpy as np

# ── Directory containing PA files, CSVs, and trap_config.py ──────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# ── Globals shared across worker processes ────────────────────────────────────
# Set by _shm_initializer (multiprocessing) or directly in single-process mode.
# _worker_phi_stack shape: (N_ELEC, NZ, NY, NX) — all 10 unit-potential arrays stacked.
_worker_phi_stack = None
_worker_grid      = None

# ── Physical constants ────────────────────────────────────────────────────────
KB_J        = 1.38065e-23       # J / K
AMU_KG      = 1.66054e-27       # kg per amu
E_C         = 1.602176634e-19   # Coulombs per elementary charge
G_MM_US2    = 9.81e-9           # mm / µs²  (standard gravity)

# ── PA grid constants (read from header; these are the expected values) ───────
HEADER_BYTES = 56
N_ELEC       = 10               # electrodes 1..10

# ── Dormand-Prince Butcher tableau ────────────────────────────────────────────
# Coefficients taken from the standard DP45 tableau.
_DP_A = [
    [],                                                             # k1
    [1/5],                                                          # k2
    [3/40,        9/40],                                            # k3
    [44/45,      -56/15,       32/9],                               # k4
    [19372/6561, -25360/2187,  64448/6561,  -212/729],              # k5
    [9017/3168,  -355/33,      46732/5247,   49/176,  -5103/18656], # k6
    [35/384,      0,           500/1113,     125/192, -2187/6784,   11/84],  # k7
]
# 5th-order propagation weights (same as k7 row → FSAL)
_DP_B5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
# Error estimate weights  (difference 5th − 4th order)
_DP_E  = np.array([71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40])


# ─────────────────────────────────────────────────────────────────────────────
# PA loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_pa_unit(path):
    """Load a SIMION PA file and return the unit-potential array.

    Returns
    -------
    phi : ndarray, shape (NZ, NY, NX), float64
        Potential in volts when 1 V is applied to this electrode.
    NX, NY, NZ : int
    DX : float   (mm per grid step — assumed isotropic)
    """
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        hdr = f.read(HEADER_BYTES)
        raw_bytes = f.read()

    scale_ref = struct.unpack_from("<d", hdr, 8)[0]   # typically 100000.0
    NX = struct.unpack_from("<i", hdr, 16)[0]
    NY = struct.unpack_from("<i", hdr, 20)[0]
    NZ = struct.unpack_from("<i", hdr, 24)[0]
    DX = struct.unpack_from("<d", hdr, 32)[0]         # mm

    n_pts = NX * NY * NZ
    expected = HEADER_BYTES + n_pts * 8
    if fsize != expected:
        print(f"  WARNING {os.path.basename(path)}: "
              f"size {fsize} != expected {expected} ({NX}×{NY}×{NZ})")

    raw = np.frombuffer(raw_bytes, dtype="<f8", count=n_pts).copy()

    # Decode electrode surface encoding
    elec_other = np.signbit(raw)                    # other-electrode voxels
    elec_this  = raw > 1.5 * scale_ref              # this-electrode voxels

    phi = np.abs(raw) / scale_ref
    phi[elec_this]  = 1.0
    phi[elec_other] = 0.0

    # Reshape: data layout is [NZ][NY][NX] (z slowest, x fastest)
    phi = phi.reshape(NZ, NY, NX)
    return phi, NX, NY, NZ, DX


def load_all_phi(base_dir):
    """Load paulTrap.pa1–pa10 and return a stacked unit-potential array.

    E = −∇φ is evaluated analytically from the trilinear interpolant gradient
    at each query point, keeping peak memory at ~780 MB (10 × 78 MB) rather
    than ~2.3 GB (30 pre-computed gradient arrays).

    Returns
    -------
    phi_stack : ndarray, shape (N_ELEC, NZ, NY, NX), float64
                Stacked unit-potential arrays; axis 0 = electrode index (0-based).
                Vectorised access (phi_stack[:, k, j, i]) evaluates all 10
                electrodes at once — much faster than 10 separate scalar indexing
                operations in a Python loop.
    grid      : dict with keys NX, NY, NZ, DX
    """
    phi_list = []
    grid     = None

    for en in range(1, N_ELEC + 1):
        pa_path = os.path.join(base_dir, f"paulTrap.pa{en}")
        if not os.path.exists(pa_path):
            raise FileNotFoundError(f"PA file not found: {pa_path}")

        print(f"  Loading paulTrap.pa{en} …", end="", flush=True)
        t0 = time.perf_counter()
        phi, NX, NY, NZ, DX = _load_pa_unit(pa_path)
        print(f" {NX}×{NY}×{NZ},  DX={DX:.3g} mm", end="", flush=True)

        if grid is None:
            grid = dict(NX=NX, NY=NY, NZ=NZ, DX=DX)
        else:
            if (NX, NY, NZ) != (grid["NX"], grid["NY"], grid["NZ"]):
                raise ValueError(
                    f"paulTrap.pa{en} has different grid size "
                    f"({NX},{NY},{NZ}) vs pa1 ({grid['NX']},{grid['NY']},{grid['NZ']})")

        phi_list.append(phi)
        elapsed = time.perf_counter() - t0
        print(f"  ({elapsed:.1f}s)", flush=True)

    phi_stack = np.stack(phi_list, axis=0)   # (N_ELEC, NZ, NY, NX)
    return phi_stack, grid


# ─────────────────────────────────────────────────────────────────────────────
# Voltage schedule loading
# ─────────────────────────────────────────────────────────────────────────────

def load_voltage_schedule(csv_path):
    """Parse voltages_N.csv (as produced by generate_voltages.py).

    Returns
    -------
    sched : dict with numpy arrays for each column name found in the CSV,
            plus scalar metadata keys:
              "f_RF_Hz"   – carrier frequency, sets 1+2 (float or None)
              "f_RF3_Hz"  – carrier frequency, set 3    (float or None)
            Main schedule arrays (indexed by time_us rows):
              "vt"            – time axis (µs)
              "v_rf"          – V_RF amplitude envelope
              "v_rf3"         – V_RF3 amplitude envelope
              "v_ec_load_U"   – electrode 3
              "v_ec_load_D"   – electrode 4
              "v_dc_TL"       – electrode 5 DC trim
              "v_dc_TR"       – electrode 6
              "v_dc_BL"       – electrode 7
              "v_dc_BR"       – electrode 8
              "v_ec_opt_U"    – electrode 9
              "v_ec_opt_D"    – electrode 10
            Post-trigger arrays (indexed by time_trig_us rows):
              "vt_trig"       – trigger time axis (µs since trigger)
              "v_trig"        – dict: electrode_num → ndarray (may be empty)
    """
    f_RF_Hz  = None
    f_RF3_Hz = None

    header = None
    main_rows  = []   # rows with time_us present
    trig_rows  = []   # rows with time_trig_us present

    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = re.match(r"#\s*f_RF_Hz\s*=\s*([0-9.eE+\-]+)", line)
                if m:
                    f_RF_Hz = float(m.group(1))
                m = re.match(r"#\s*f_RF3_Hz\s*=\s*([0-9.eE+\-]+)", line)
                if m:
                    f_RF3_Hz = float(m.group(1))
                continue
            if header is None:
                header = [h.strip() for h in line.split(",")]
                continue
            # Data row — split on commas, treating empty fields as NaN
            parts = line.split(",")
            vals = []
            for p in parts:
                p = p.strip()
                vals.append(float(p) if p else float("nan"))
            # Pad to header length if needed
            while len(vals) < len(header):
                vals.append(float("nan"))

            row = dict(zip(header, vals))
            has_main = "time_us" in row and not math.isnan(row.get("time_us", float("nan")))
            has_trig = "time_trig_us" in row and not math.isnan(row.get("time_trig_us", float("nan")))

            if has_main:
                main_rows.append(row)
            if has_trig:
                trig_rows.append(row)

    def _col(rows, name):
        """Extract a named column from a list of row dicts; return empty array if absent."""
        if not rows or name not in rows[0]:
            return np.array([])
        vals = [r.get(name, float("nan")) for r in rows]
        arr = np.array(vals, dtype=float)
        # Drop rows where this particular column is NaN
        ok = ~np.isnan(arr)
        return arr[ok]

    def _col_keep_all(rows, name):
        """Extract column, keeping NaN rows (for time alignment)."""
        if not rows:
            return np.array([])
        return np.array([r.get(name, float("nan")) for r in rows], dtype=float)

    vt = _col(main_rows, "time_us")

    # For columns with possible missing values, we need time-aligned extraction
    def _aligned_col(rows, time_name, val_name):
        if not rows:
            return np.array([]), np.array([])
        t_all = np.array([r.get(time_name, float("nan")) for r in rows])
        v_all = np.array([r.get(val_name, float("nan")) for r in rows])
        ok = ~np.isnan(t_all) & ~np.isnan(v_all)
        return t_all[ok], v_all[ok]

    # Main schedule columns (all share the same time axis vt)
    def _main_col(name):
        if not main_rows:
            return np.array([])
        vals = np.array([r.get(name, float("nan")) for r in main_rows])
        ok = ~np.isnan(vals)
        if not ok.all():
            # Some rows missing — fall back to rows that have both time and value
            t_ok = ~np.isnan(np.array([r.get("time_us", float("nan")) for r in main_rows]))
            ok2 = ok & t_ok
            return vals[ok2]
        return vals

    sched = {
        "f_RF_Hz":    f_RF_Hz,
        "f_RF3_Hz":   f_RF3_Hz,
        "vt":         vt,
        "v_rf":       _main_col("V_RF"),
        "v_rf3":      _main_col("V_RF3"),
        "v_ec_load_U": _main_col("V_endcap_load_U"),
        "v_ec_load_D": _main_col("V_endcap_load_D"),
        "v_dc_TL":    _main_col("V_dc_3_TL"),
        "v_dc_TR":    _main_col("V_dc_3_TR"),
        "v_dc_BL":    _main_col("V_dc_3_BL"),
        "v_dc_BR":    _main_col("V_dc_3_BR"),
        "v_ec_opt_U": _main_col("V_endcap_optical_U"),
        "v_ec_opt_D": _main_col("V_endcap_optical_D"),
    }

    # Trigger schedule
    if trig_rows:
        vt_trig = _col(trig_rows, "time_trig_us")
        sched["vt_trig"] = vt_trig
        v_trig = {}
        for en in range(1, N_ELEC + 1):
            col_name = f"V_e{en}_trig"
            if col_name in (trig_rows[0] if trig_rows else {}):
                arr = np.array([r.get(col_name, float("nan")) for r in trig_rows])
                ok = ~np.isnan(arr)
                v_trig[en] = arr[ok]
        sched["v_trig"] = v_trig
    else:
        sched["vt_trig"] = np.array([])
        sched["v_trig"]  = {}

    n_main = len(vt)
    n_trig = len(sched["vt_trig"])
    print(f"  Loaded {n_main} main rows + {n_trig} trigger rows from "
          f"{os.path.basename(csv_path)}")
    if f_RF_Hz is not None:
        print(f"  f_RF = {f_RF_Hz:g} Hz,  f_RF3 = {f_RF3_Hz:g} Hz")
    for name in ("v_rf", "v_rf3", "v_ec_load_U", "v_ec_load_D",
                 "v_dc_TL", "v_dc_TR", "v_dc_BL", "v_dc_BR",
                 "v_ec_opt_U", "v_ec_opt_D"):
        arr = sched[name]
        if len(arr):
            print(f"    {name:<16}: {len(arr):4d} rows,  "
                  f"t=0: {arr[0]:+.1f} V,  t_end: {arr[-1]:+.1f} V")
        else:
            print(f"    {name:<16}: NOT LOADED (column missing)")

    return sched


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(base_dir):
    """Import trap_config.py from base_dir and return its 'config' dict."""
    cfg_path = os.path.join(base_dir, "trap_config.py")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"trap_config.py not found at {cfg_path}")
    spec   = importlib.util.spec_from_file_location("trap_config", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _interp(t_arr, v_arr, t):
    """Linear interpolation of v_arr at t, clamped to endpoints."""
    if len(t_arr) == 0:
        return 0.0
    if t <= t_arr[0]:
        return float(v_arr[0])
    if t >= t_arr[-1]:
        return float(v_arr[-1])
    idx = np.searchsorted(t_arr, t, side="right") - 1
    idx = max(0, min(idx, len(t_arr) - 2))
    frac = (t - t_arr[idx]) / (t_arr[idx + 1] - t_arr[idx])
    return float(v_arr[idx] + frac * (v_arr[idx + 1] - v_arr[idx]))


# ─────────────────────────────────────────────────────────────────────────────
# Trilinear E-field interpolation
# ─────────────────────────────────────────────────────────────────────────────

def _phi_grad_trilinear(phi_arr, NX, NY, NZ, DX, px, py, pz):
    """Trilinear interpolation of φ and its gradient from the φ array alone.

    Computes E = −∇φ analytically from the gradient of the trilinear
    interpolant using the same 8 corner φ values.  This is equivalent to
    interpolating separately pre-computed E-field arrays but uses 3× less
    memory (one φ array instead of three E-field arrays per electrode).

    Returns
    -------
    (ez, ey, ex) : E-field components at the query point (V/mm per unit volt)
    """
    fx = max(0.0, min(px / DX, NX - 1.0001))
    fy = max(0.0, min(py / DX, NY - 1.0001))
    fz = max(0.0, min(pz / DX, NZ - 1.0001))

    i0 = int(fx);  wx = fx - i0;  i0 = min(i0, NX - 2)
    j0 = int(fy);  wy = fy - j0;  j0 = min(j0, NY - 2)
    k0 = int(fz);  wz = fz - k0;  k0 = min(k0, NZ - 2)
    ox, oy, oz = 1.0 - wx, 1.0 - wy, 1.0 - wz

    # 8 corner φ values  [k][j][i]
    c000 = phi_arr[k0,   j0,   i0  ]
    c100 = phi_arr[k0,   j0,   i0+1]
    c010 = phi_arr[k0,   j0+1, i0  ]
    c110 = phi_arr[k0,   j0+1, i0+1]
    c001 = phi_arr[k0+1, j0,   i0  ]
    c101 = phi_arr[k0+1, j0,   i0+1]
    c011 = phi_arr[k0+1, j0+1, i0  ]
    c111 = phi_arr[k0+1, j0+1, i0+1]

    # Gradient of the trilinear interpolant (per grid spacing → divide by DX)
    # ∂φ/∂x: differentiate weights w.r.t. wx, treating wy, wz as constants
    dphi_dx = (oy*oz*(c100 - c000) + wy*oz*(c110 - c010) +
               oy*wz*(c101 - c001) + wy*wz*(c111 - c011)) / DX
    dphi_dy = (ox*oz*(c010 - c000) + wx*oz*(c110 - c100) +
               ox*wz*(c011 - c001) + wx*wz*(c111 - c101)) / DX
    dphi_dz = (ox*oy*(c001 - c000) + wx*oy*(c101 - c100) +
               ox*wy*(c011 - c010) + wx*wy*(c111 - c110)) / DX

    # E = −∇φ
    return -dphi_dz, -dphi_dy, -dphi_dx


def _phi_grad_all(phi_stack, NX, NY, NZ, DX, px, py, pz):
    """Vectorised trilinear interpolation across all N_ELEC electrodes at once.

    phi_stack shape: (N_ELEC, NZ, NY, NX).

    Uses numpy slice ``phi_stack[:, k, j, i]`` to fetch all 10 electrode values
    in a single C-level operation — 8 such slices replace 80 Python-level scalar
    array accesses, giving a ~5× speedup on the field evaluation hot-path.

    Returns
    -------
    (ez_arr, ey_arr, ex_arr) : ndarrays of shape (N_ELEC,)
        E = −∇φ components per electrode (V/mm per unit volt).
        Dot-product with the voltage vector gives the total field component.
    """
    fx = max(0.0, min(px / DX, NX - 1.0001))
    fy = max(0.0, min(py / DX, NY - 1.0001))
    fz = max(0.0, min(pz / DX, NZ - 1.0001))

    i0 = int(fx);  wx = fx - i0;  i0 = min(i0, NX - 2)
    j0 = int(fy);  wy = fy - j0;  j0 = min(j0, NY - 2)
    k0 = int(fz);  wz = fz - k0;  k0 = min(k0, NZ - 2)
    ox, oy, oz = 1.0 - wx, 1.0 - wy, 1.0 - wz

    # 8 corner vectors, each shape (N_ELEC,)
    c000 = phi_stack[:, k0,   j0,   i0  ]
    c100 = phi_stack[:, k0,   j0,   i0+1]
    c010 = phi_stack[:, k0,   j0+1, i0  ]
    c110 = phi_stack[:, k0,   j0+1, i0+1]
    c001 = phi_stack[:, k0+1, j0,   i0  ]
    c101 = phi_stack[:, k0+1, j0,   i0+1]
    c011 = phi_stack[:, k0+1, j0+1, i0  ]
    c111 = phi_stack[:, k0+1, j0+1, i0+1]

    dphi_dx = (oy*oz*(c100-c000) + wy*oz*(c110-c010) +
               oy*wz*(c101-c001) + wy*wz*(c111-c011)) / DX
    dphi_dy = (ox*oz*(c010-c000) + wx*oz*(c110-c100) +
               ox*wz*(c011-c001) + wx*wz*(c111-c101)) / DX
    dphi_dz = (ox*oy*(c001-c000) + wx*oy*(c101-c100) +
               ox*wy*(c011-c010) + wx*wy*(c111-c110)) / DX

    return -dphi_dz, -dphi_dy, -dphi_dx


# ─────────────────────────────────────────────────────────────────────────────
# Shared-memory worker initializer
# ─────────────────────────────────────────────────────────────────────────────

def _shm_initializer(shm_name, shm_shape, grid):
    """Attach to the shared phi_stack block in each worker process.

    Called once per worker at pool startup.  Sets the module globals
    _worker_phi_stack and _worker_grid so that _worker() can access the
    potential arrays without any per-task data transfer (no pickle overhead).
    """
    global _worker_phi_stack, _worker_grid
    shm = multiprocessing.shared_memory.SharedMemory(name=shm_name)
    _worker_phi_stack = np.ndarray(shm_shape, dtype=np.float64, buffer=shm.buf)
    # shm handle is intentionally kept open for the worker's lifetime
    _worker_grid = grid


# ─────────────────────────────────────────────────────────────────────────────
# Voltage computation (fast_adjust equivalent)
# ─────────────────────────────────────────────────────────────────────────────

def compute_voltages(t_abs, sched, trig_fired, trig_fire_time,
                     omega_rf, omega_rf3, V0_default, V0_3_default,
                     trig_for_electrode):
    """Compute the 10 electrode voltages at absolute time t_abs.

    Parameters
    ----------
    t_abs            : float  (µs, absolute time of flight for this ion)
    sched            : dict returned by load_voltage_schedule()
    trig_fired       : list[bool]  (1-indexed; trig_fired[i] for trigger i)
    trig_fire_time   : list[float] (µs; fire time for each trigger)
    omega_rf         : float  (rad/µs, sets 1+2)
    omega_rf3        : float  (rad/µs, set 3)
    V0_default       : float  (V, fallback amplitude sets 1+2)
    V0_3_default     : float  (V, fallback amplitude set 3)
    trig_for_electrode : dict {electrode_int → trigger_index_1based}

    Returns
    -------
    adj : dict {1..10 → float} of electrode voltages in volts
    """
    vt          = sched["vt"]
    v_rf        = sched["v_rf"]
    v_rf3       = sched["v_rf3"]
    v_ec_load_U = sched["v_ec_load_U"]
    v_ec_load_D = sched["v_ec_load_D"]
    v_dc_TL     = sched["v_dc_TL"]
    v_dc_TR     = sched["v_dc_TR"]
    v_dc_BL     = sched["v_dc_BL"]
    v_dc_BR     = sched["v_dc_BR"]
    v_ec_opt_U  = sched["v_ec_opt_U"]
    v_ec_opt_D  = sched["v_ec_opt_D"]
    vt_trig     = sched["vt_trig"]
    v_trig      = sched["v_trig"]

    def volt_dc(en, t_abs_val, v_main):
        """DC voltage for electrode en, respecting trigger state."""
        trig_idx = trig_for_electrode.get(en)
        if trig_idx is None:
            # Not a triggered electrode — follow main schedule absolutely
            return _interp(vt, v_main, t_abs_val)
        # Triggered electrode
        if not trig_fired[trig_idx]:
            # Trigger not yet fired — use main schedule with absolute time
            return _interp(vt, v_main, t_abs_val)
        # Trigger fired — use post-trigger schedule (time since fire)
        t_rel = t_abs_val - trig_fire_time[trig_idx]
        vt_trig_arr = vt_trig
        if len(vt_trig_arr) > 0 and en in v_trig and len(v_trig[en]) > 0:
            return _interp(vt_trig_arr, v_trig[en], t_rel)
        # Fallback: main schedule using time-since-fire
        return _interp(vt, v_main, t_rel)

    # Sets 1+2 RF (electrodes 1, 2) — always on, absolute time
    amp_rf = _interp(vt, v_rf, t_abs) if len(v_rf) else V0_default
    V_RF   = amp_rf * math.cos(omega_rf * t_abs)

    # Set 3 RF (electrodes 5–8) — always on, absolute time
    amp_rf3 = _interp(vt, v_rf3, t_abs) if len(v_rf3) else V0_3_default
    V_RF3   = amp_rf3 * math.cos(omega_rf3 * t_abs)

    adj = {
        1:  V_RF,
        2: -V_RF,
        3:  volt_dc(3, t_abs, v_ec_load_U),
        4:  volt_dc(4, t_abs, v_ec_load_D),
        5:  V_RF3 + volt_dc(5, t_abs, v_dc_TL),
        6: -V_RF3 + volt_dc(6, t_abs, v_dc_TR),
        7: -V_RF3 + volt_dc(7, t_abs, v_dc_BL),
        8:  V_RF3 + volt_dc(8, t_abs, v_dc_BR),
        9:  volt_dc(9,  t_abs, v_ec_opt_U),
        10: volt_dc(10, t_abs, v_ec_opt_D),
    }
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics function
# ─────────────────────────────────────────────────────────────────────────────

def _dynamics(t, y, adj, phi_stack, NX, NY, NZ, DX, q_C, m_kg):
    """Compute dy/dt = [vx, vy, vz, ax, ay, az] for the Dormand-Prince stages.

    Parameters
    ----------
    y         : ndarray [px, py, pz, vx, vy, vz]  (GEM mm, mm/µs)
    adj       : dict {1..10 → float} electrode voltages
    phi_stack : ndarray, shape (N_ELEC, NZ, NY, NX)

    Returns
    -------
    dydt : ndarray [vx, vy, vz, ax, ay, az]  (mm/µs, mm/µs²)
    """
    px, py, pz, vx, vy, vz = y

    # Voltage vector for all 10 electrodes
    v_arr = np.empty(N_ELEC)
    for en in range(1, N_ELEC + 1):
        v_arr[en - 1] = adj[en]

    # Vectorised field: 8 numpy slices across N_ELEC then 3 dot products
    ez_arr, ey_arr, ex_arr = _phi_grad_all(phi_stack, NX, NY, NZ, DX, px, py, pz)
    ez_tot = float(np.dot(v_arr, ez_arr))
    ey_tot = float(np.dot(v_arr, ey_arr))
    ex_tot = float(np.dot(v_arr, ex_arr))

    # Electric acceleration:  a = (q_C * E_Vmm * 1e3) / m_kg [m/s²] * 1e-9 [mm/µs²]
    #                           = q_C * E_Vmm / m_kg * 1e-6
    scale = q_C / m_kg * 1e-6   # converts V/mm → mm/µs²

    ax = scale * ex_tot
    ay = scale * ey_tot - G_MM_US2    # gravity: −Y in GEM coords
    az = scale * ez_tot

    return np.array([vx, vy, vz, ax, ay, az])


# ─────────────────────────────────────────────────────────────────────────────
# Single-particle integrator
# ─────────────────────────────────────────────────────────────────────────────

def integrate_particle(ion_num, y0, t_start, sim_params, sched, phi_stack, grid):
    """Integrate one particle trajectory using Dormand-Prince RK45 + Langevin.

    Parameters
    ----------
    ion_num     : int  (1-based, for logging and output)
    y0          : ndarray [px, py, pz, vx, vy, vz]  GEM coords (mm, mm/µs)
    t_start     : float  (µs, starting time; typically 0)
    sim_params  : dict with keys:
                    q_C, m_kg, gamma_per_pa, P_baseline, kT_over_m,
                    drag_scale, langevin_on, v_stop, record_stride,
                    t_max_us, atol, rtol, dt_init, dt_min, dt_max,
                    gem_off (dict x,y,z), triggers (list of dicts),
                    trig_for_electrode (dict), omega_rf, omega_rf3,
                    V0_default, V0_3_default,
                    pressure_ramp (None or dict)
    sched       : dict from load_voltage_schedule()
    phi_stack   : ndarray, shape (N_ELEC, NZ, NY, NX)
    grid        : dict NX,NY,NZ,DX

    Returns
    -------
    rows : list of str  (CSV lines, without the header)
    info : dict with keys: steps_accepted, t_sim_us, reason
    """
    NX  = grid["NX"]
    NY  = grid["NY"]
    NZ  = grid["NZ"]
    DX  = grid["DX"]

    q_C             = sim_params["q_C"]
    m_kg            = sim_params["m_kg"]
    gamma_per_pa    = sim_params["gamma_per_pa"]
    P_baseline      = sim_params["P_baseline"]
    kT_over_m       = sim_params["kT_over_m"]
    drag_scale      = sim_params["drag_scale"]
    langevin_on     = sim_params["langevin_on"]
    v_stop          = sim_params["v_stop"]
    record_stride   = sim_params["record_stride"]
    t_max_us        = sim_params["t_max_us"]
    atol            = sim_params["atol"]
    rtol            = sim_params["rtol"]
    dt_min          = sim_params["dt_min"]
    dt_max          = sim_params["dt_max"]
    gem_off         = sim_params["gem_off"]
    triggers        = sim_params["triggers"]
    trig_for_elec   = sim_params["trig_for_electrode"]
    omega_rf        = sim_params["omega_rf"]
    omega_rf3       = sim_params["omega_rf3"]
    V0_default      = sim_params["V0_default"]
    V0_3_default    = sim_params["V0_3_default"]
    pressure_ramp   = sim_params["pressure_ramp"]

    # Per-particle trigger state (1-indexed)
    n_trigs = len(triggers)
    trig_fired     = [False] * (n_trigs + 1)   # index 1..n_trigs
    trig_fire_time = [0.0]   * (n_trigs + 1)

    def current_pressure(t):
        if pressure_ramp is None:
            return P_baseline
        pr_idx = pressure_ramp["trigger"]   # 1-based
        if pr_idx < 1 or pr_idx > n_trigs or not trig_fired[pr_idx]:
            return P_baseline
        t_since = t - trig_fire_time[pr_idx]
        if t_since <= 0:
            return P_baseline
        dur = pressure_ramp["duration_us"]
        if dur <= 0 or t_since >= dur:
            return pressure_ramp["P_final_pa"]
        return P_baseline + (t_since / dur) * (pressure_ramp["P_final_pa"] - P_baseline)

    def check_triggers(pos_gem, t):
        """Update trigger state. pos_gem = [px, py, pz] in GEM coords."""
        z_fusion = pos_gem[2] - gem_off["z"]
        for i, trig in enumerate(triggers, start=1):
            if not trig_fired[i] and z_fusion >= trig["z_mm"]:
                trig_fired[i]     = True
                trig_fire_time[i] = t
                print(f"  Trigger {i} fired: ion {ion_num} at "
                      f"Z={z_fusion:.2f} mm, t={t:.1f} µs")

    def is_outside_grid(pos_gem):
        """True if pos_gem is outside the usable grid (would extrapolate)."""
        px, py, pz = pos_gem
        if px < 0 or px > (NX - 1) * DX:
            return True
        if py < 0 or py > (NY - 1) * DX:
            return True
        if pz < 0 or pz > (NZ - 1) * DX:
            return True
        return False

    # ── RK45 Dormand-Prince constants ────────────────────────────────────────
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    # 5th-order weights (= k7 weights, FSAL)
    b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
    # Error weights (5th − 4th order)
    e1, e3, e4, e5, e6, e7 = 71/57600, -71/16695, 71/1920, -17253/339200, 22/525, -1/40

    rng = np.random.default_rng(seed=ion_num)

    y    = y0.copy()
    t    = float(t_start)
    dt   = sim_params.get("dt_init", 1.0)   # µs, initial step

    rows              = []
    steps_accepted    = 0
    steps_rejected    = 0
    consec_rejected   = 0    # consecutive rejection counter (infinite-loop guard)
    step_counter      = 0    # for record_stride modulo
    reason            = "max_time"
    t_wall_start      = time.perf_counter()
    t_wall_last_print = t_wall_start

    # Compute k1 for FSAL
    adj  = compute_voltages(t, sched, trig_fired, trig_fire_time,
                            omega_rf, omega_rf3, V0_default, V0_3_default,
                            trig_for_elec)
    k1   = _dynamics(t, y, adj, phi_stack, NX, NY, NZ, DX, q_C, m_kg)

    while t < t_max_us:
        # Clamp dt so we don't overshoot t_max
        dt = min(dt, t_max_us - t)
        if dt < dt_min:
            dt = dt_min

        # ── Dormand-Prince stages ───────────────────────────────────────────
        # k1 already computed (FSAL)
        y2   = y + dt * a21 * k1
        adj2 = compute_voltages(t + dt/5, sched, trig_fired, trig_fire_time,
                                omega_rf, omega_rf3, V0_default, V0_3_default,
                                trig_for_elec)
        k2   = _dynamics(t + dt/5, y2, adj2, phi_stack, NX, NY, NZ, DX, q_C, m_kg)

        y3   = y + dt * (a31*k1 + a32*k2)
        adj3 = compute_voltages(t + 3*dt/10, sched, trig_fired, trig_fire_time,
                                omega_rf, omega_rf3, V0_default, V0_3_default,
                                trig_for_elec)
        k3   = _dynamics(t + 3*dt/10, y3, adj3, phi_stack, NX, NY, NZ, DX, q_C, m_kg)

        y4   = y + dt * (a41*k1 + a42*k2 + a43*k3)
        adj4 = compute_voltages(t + 4*dt/5, sched, trig_fired, trig_fire_time,
                                omega_rf, omega_rf3, V0_default, V0_3_default,
                                trig_for_elec)
        k4   = _dynamics(t + 4*dt/5, y4, adj4, phi_stack, NX, NY, NZ, DX, q_C, m_kg)

        y5   = y + dt * (a51*k1 + a52*k2 + a53*k3 + a54*k4)
        adj5 = compute_voltages(t + 8*dt/9, sched, trig_fired, trig_fire_time,
                                omega_rf, omega_rf3, V0_default, V0_3_default,
                                trig_for_elec)
        k5   = _dynamics(t + 8*dt/9, y5, adj5, phi_stack, NX, NY, NZ, DX, q_C, m_kg)

        y6   = y + dt * (a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)
        adj6 = compute_voltages(t + dt, sched, trig_fired, trig_fire_time,
                                omega_rf, omega_rf3, V0_default, V0_3_default,
                                trig_for_elec)
        k6   = _dynamics(t + dt, y6, adj6, phi_stack, NX, NY, NZ, DX, q_C, m_kg)

        # 5th-order solution
        y_new = y + dt * (b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)

        # k7 for FSAL and error estimate
        adj7 = compute_voltages(t + dt, sched, trig_fired, trig_fire_time,
                                omega_rf, omega_rf3, V0_default, V0_3_default,
                                trig_for_elec)
        k7   = _dynamics(t + dt, y_new, adj7, phi_stack, NX, NY, NZ, DX, q_C, m_kg)

        # Error vector (5th − 4th order) = h * sum(e_i * k_i)
        err_vec = dt * (e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6 + e7*k7)

        # Error norm (mixed absolute/relative)
        sc      = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = float(np.max(np.abs(err_vec) / sc))

        if err_norm <= 1.0:
            # ── Accept step ─────────────────────────────────────────────────
            t   += dt
            y    = y_new
            k1   = k7   # FSAL
            steps_accepted += 1
            step_counter   += 1
            consec_rejected = 0

            # ── Progress report every 10 s of wall time ───────────────────────
            now = time.perf_counter()
            if now - t_wall_last_print >= 10.0:
                t_wall_last_print = now
                z_f = y[2] - gem_off["z"]
                print(f"  [ion {ion_num}] t={t:.0f}/{t_max_us:.0f} µs  "
                      f"Z={z_f:.1f} mm  dt={dt:.2g} µs  "
                      f"acc={steps_accepted}  rej={steps_rejected}",
                      flush=True)

            # ── Check triggers ───────────────────────────────────────────────
            check_triggers(y[:3], t)

            # ── Drag + Langevin split step ───────────────────────────────────
            P_now   = current_pressure(t)
            gamma   = drag_scale * gamma_per_pa * P_now   # µs⁻¹
            if gamma * dt > 1e-12:
                # Exact exponential drag on velocity
                decay = math.exp(-gamma * dt)
                y[3] *= decay
                y[4] *= decay
                y[5] *= decay
                # Langevin thermal kick
                if langevin_on:
                    sigma = math.sqrt(kT_over_m * (1.0 - math.exp(-2.0 * gamma * dt))) * 1e-3
                    # Convert kT/m from (m/s)² to (mm/µs)²: × 1e-6; σ in mm/µs
                    y[3] += sigma * rng.standard_normal()
                    y[4] += sigma * rng.standard_normal()
                    y[5] += sigma * rng.standard_normal()

            # ── Boundary check ───────────────────────────────────────────────
            if is_outside_grid(y[:3]):
                reason = "lost"
                # Record final position before breaking
                if record_stride > 0:
                    x_f = y[0] - gem_off["x"]
                    y_f = y[1] - gem_off["y"]
                    z_f = y[2] - gem_off["z"]
                    rows.append(f"{ion_num},{t:.4f},{x_f:.5f},{y_f:.5f},{z_f:.5f}")
                break

            # ── Speed-based termination ──────────────────────────────────────
            speed = math.sqrt(y[3]**2 + y[4]**2 + y[5]**2)
            if v_stop > 0 and speed < v_stop:
                reason = "trapped"
                if record_stride > 0:
                    x_f = y[0] - gem_off["x"]
                    y_f = y[1] - gem_off["y"]
                    z_f = y[2] - gem_off["z"]
                    rows.append(f"{ion_num},{t:.4f},{x_f:.5f},{y_f:.5f},{z_f:.5f}")
                break

            # ── Record trajectory ────────────────────────────────────────────
            if record_stride > 0 and step_counter % record_stride == 0:
                x_f = y[0] - gem_off["x"]
                y_f = y[1] - gem_off["y"]
                z_f = y[2] - gem_off["z"]
                rows.append(f"{ion_num},{t:.4f},{x_f:.5f},{y_f:.5f},{z_f:.5f}")

            # ── Adapt dt ────────────────────────────────────────────────────
            if err_norm > 1e-10:
                factor = 0.9 * err_norm ** (-0.2)
            else:
                factor = 5.0
            dt = min(dt * min(5.0, factor), dt_max)
            dt = max(dt, dt_min)

        else:
            # ── Reject step — shrink dt ──────────────────────────────────────
            factor = max(0.1, 0.9 * err_norm ** (-0.2))
            dt = max(dt * factor, dt_min)
            steps_rejected  += 1
            consec_rejected += 1
            if consec_rejected >= 200_000:
                z_f = y[2] - gem_off["z"]
                print(f"  [ion {ion_num}] WARNING: stuck at t={t:.2f} µs  "
                      f"Z={z_f:.1f} mm  err_norm={err_norm:.2e}  dt={dt:.4g} µs — "
                      f"advancing by dt_min and continuing",
                      flush=True)
                t  += dt_min
                y   = y + dt_min * k1   # Euler step to escape
                k1  = _dynamics(t, y, adj, phi_stack, NX, NY, NZ, DX, q_C, m_kg)
                consec_rejected = 0

    info = {
        "steps_accepted": steps_accepted,
        "t_sim_us":       t,
        "reason":         reason,
    }
    return rows, info


# ─────────────────────────────────────────────────────────────────────────────
# Worker (called in each subprocess)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args):
    """Entry point for multiprocessing.Pool workers.

    phi_stack and grid are taken from the module globals _worker_phi_stack /
    _worker_grid, which are set either by _shm_initializer (multiprocessing)
    or directly in main() (single-process mode).  This avoids pickling the
    large potential arrays into every task argument.
    """
    ion_num, y0, t_start, sim_params, sched = args
    phi_stack = _worker_phi_stack
    grid      = _worker_grid
    print(f"  [ion {ion_num}] starting at GEM "
          f"({y0[0]:.2f}, {y0[1]:.2f}, {y0[2]:.2f}) mm, "
          f"v=({y0[3]:.4g}, {y0[4]:.4g}, {y0[5]:.4g}) mm/µs",
          flush=True)
    t0 = time.perf_counter()
    rows, info = integrate_particle(ion_num, y0, t_start, sim_params, sched, phi_stack, grid)
    elapsed = time.perf_counter() - t0
    print(f"  [ion {ion_num}] done: {info['steps_accepted']} steps, "
          f"{info['t_sim_us']:.1f} µs simulated, "
          f"reason={info['reason']}, "
          f"wall={elapsed:.1f}s",
          flush=True)
    return ion_num, rows, info


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Pure-Python Paul trap particle integrator (replaces SIMION fly step).")
    ap.add_argument("--vol",     type=int, default=1,
                    help="Voltage schedule file number N → voltages_N.csv (default: 1)")
    ap.add_argument("--run",     type=int, default=1,
                    help="Run number N → trajectories_N.csv (default: 1)")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of parallel worker processes (default: CPU count)")
    args = ap.parse_args()

    n_workers = args.workers or os.cpu_count() or 1

    print("=" * 70)
    print(f"fly.py  —  vol={args.vol}  run={args.run}  workers={n_workers}")
    print("=" * 70)

    # ── Load configuration ───────────────────────────────────────────────────
    print("\nLoading trap_config.py …")
    cfg = load_config(BASE)

    T_gas   = cfg.get("temperature_k", 293)
    P_gas   = cfg.get("pressure_pa", 0.1)
    M_gas   = cfg.get("gas_molar_mass_amu", 28.0)
    r_p     = cfg.get("particle_radius_m", 83e-9)
    rho_p   = cfg.get("particle_density_kgm3", 2200)
    drag_scale   = cfg.get("drag_scale", 1.0)
    langevin_on  = cfg.get("langevin_noise", True)
    v_stop       = cfg.get("v_stop_mm_us", 1e-6)
    record_stride = cfg.get("record_stride", 20)
    gem_off_raw  = cfg.get("gem_offset", {"x": 25.0, "y": 8.0, "z": 132.0})
    triggers_raw = cfg.get("triggers", [])
    particles_cfg = cfg.get("particles", {})
    pressure_ramp_cfg = cfg.get("pressure_ramp", None)

    # Derived particle quantities
    m_kg   = (4.0 / 3.0) * math.pi * r_p**3 * rho_p
    q_e    = particles_cfg.get("charge", 100)
    q_C    = q_e * E_C

    # Epstein drag:  γ = (8π/3) r² P / (m c̄)  [s⁻¹]
    c_bar       = math.sqrt(8.0 * KB_J * T_gas / (math.pi * M_gas * AMU_KG))
    gamma_per_pa = (8.0 * math.pi / 3.0) * r_p**2 / (m_kg * c_bar) * 1e-6  # µs⁻¹ Pa⁻¹
    kT_over_m   = KB_J * T_gas / m_kg    # (m/s)²

    print(f"  T={T_gas} K,  P={P_gas} Pa,  M_gas={M_gas} amu")
    print(f"  r_p={r_p*1e9:.0f} nm,  rho={rho_p} kg/m³,  m={m_kg:.3e} kg")
    print(f"  q={q_e}e = {q_C:.3e} C")
    print(f"  gamma_per_pa={gamma_per_pa:.4e} µs⁻¹ Pa⁻¹,  drag_scale={drag_scale}")
    print(f"  gamma_baseline={gamma_per_pa*P_gas:.4e} µs⁻¹ (at {P_gas} Pa)")
    print(f"  v_rms_1D = {math.sqrt(kT_over_m)*1e-3:.3e} mm/µs")
    print(f"  Langevin: {langevin_on},  v_stop={v_stop} mm/µs")

    gem_off = {k: float(gem_off_raw[k]) for k in ("x", "y", "z")}
    print(f"  GEM offset: x={gem_off['x']}, y={gem_off['y']}, z={gem_off['z']} mm")

    # Triggers
    trig_for_electrode = {}
    for i, trig in enumerate(triggers_raw, start=1):
        for en in trig.get("electrodes", []):
            trig_for_electrode[en] = i
        print(f"  Trigger {i}: Z_Fusion >= {trig['z_mm']} mm → "
              f"electrodes {trig.get('electrodes', [])}")

    if pressure_ramp_cfg:
        print(f"  Pressure ramp: trigger {pressure_ramp_cfg['trigger']}, "
              f"{P_gas} → {pressure_ramp_cfg['P_final_pa']} Pa "
              f"over {pressure_ramp_cfg['duration_us']:.0f} µs")

    # ── Load voltage schedule ────────────────────────────────────────────────
    vol_path = os.path.join(BASE, f"voltages_{args.vol}.csv")
    print(f"\nLoading {vol_path} …")
    if not os.path.exists(vol_path):
        sys.exit(f"ERROR: voltage file not found: {vol_path}")
    sched = load_voltage_schedule(vol_path)

    f_RF_Hz  = sched.get("f_RF_Hz")  or 2000.0
    f_RF3_Hz = sched.get("f_RF3_Hz") or 2000.0
    omega_rf  = 2.0 * math.pi * f_RF_Hz  * 1e-6   # rad/µs
    omega_rf3 = 2.0 * math.pi * f_RF3_Hz * 1e-6

    # Determine simulation time from voltage schedule
    vt = sched["vt"]
    t_max_us = float(vt[-1]) if len(vt) else 2e5
    print(f"  Simulation time: 0 → {t_max_us:.0f} µs")

    # ── Load PA files ────────────────────────────────────────────────────────
    print("\nLoading PA files (unit-potential φ arrays) …")
    t_pa_start = time.perf_counter()
    phi_stack, grid = load_all_phi(BASE)
    t_pa_end = time.perf_counter()
    mem_mb = phi_stack.nbytes / 1e6
    print(f"  PA loading complete in {t_pa_end - t_pa_start:.1f} s  "
          f"({mem_mb:.0f} MB total φ memory, shape {phi_stack.shape})")
    print(f"  Grid: {grid['NX']}×{grid['NY']}×{grid['NZ']},  DX={grid['DX']} mm")

    # ── Particle starts ──────────────────────────────────────────────────────
    n_ions   = particles_cfg.get("n", 1)
    starts   = particles_cfg.get("starts", [])
    if not starts:
        sys.exit("ERROR: particles.starts is empty in trap_config.py")

    print(f"\nPreparing {n_ions} ions …")
    rng_main = np.random.default_rng(seed=42)

    ion_args = []
    for ion_idx in range(n_ions):
        ion_num = ion_idx + 1
        s = starts[(ion_idx) % len(starts)]
        sig = s.get("sigma_mm", {})
        sx = sig.get("x", 0.0) if isinstance(sig, dict) else 0.0
        sy = sig.get("y", 0.0) if isinstance(sig, dict) else 0.0
        sz = sig.get("z", 0.0) if isinstance(sig, dict) else 0.0

        # Start in GEM coordinates (add gem_offset)
        px0 = gem_off["x"] + float(s.get("x_mm", 0)) + sx * rng_main.standard_normal()
        py0 = gem_off["y"] + float(s.get("y_mm", 0)) + sy * rng_main.standard_normal()
        pz0 = gem_off["z"] + float(s.get("z_mm", 0)) + sz * rng_main.standard_normal()

        ke_ev = float(s.get("ke_ev", 0.0))
        if ke_ev == 0.0:
            vx0, vy0, vz0 = 0.0, 0.0, 0.0
        else:
            v_mag = math.sqrt(2.0 * ke_ev * E_C / m_kg) * 1e-3  # mm/µs
            el_r  = math.radians(float(s.get("el", 0.0)))
            az_r  = math.radians(float(s.get("az", 0.0)))
            vx0   = v_mag * math.cos(el_r) * math.sin(az_r)
            vy0   = v_mag * math.sin(el_r)
            vz0   = v_mag * math.cos(el_r) * math.cos(az_r)

        y0 = np.array([px0, py0, pz0, vx0, vy0, vz0])

        sim_params = dict(
            q_C=q_C,
            m_kg=m_kg,
            gamma_per_pa=gamma_per_pa,
            P_baseline=P_gas,
            kT_over_m=kT_over_m,
            drag_scale=drag_scale,
            langevin_on=langevin_on,
            v_stop=v_stop,
            record_stride=record_stride,
            t_max_us=t_max_us,
            atol=1e-3,       # mm (or mm/µs for velocity)
            rtol=1e-4,
            dt_init=1.0,     # µs
            dt_min=0.01,     # µs
            dt_max=25.0,     # µs
            gem_off=gem_off,
            triggers=triggers_raw,
            trig_for_electrode=trig_for_electrode,
            omega_rf=omega_rf,
            omega_rf3=omega_rf3,
            V0_default=100.0,
            V0_3_default=0.0,
            pressure_ramp=pressure_ramp_cfg,
        )

        ion_args.append((ion_num, y0, 0.0, sim_params, sched))

    # ── Copy phi_stack into a single shared memory block ─────────────────────
    # Workers attach via _shm_initializer; no per-task pickle of the arrays.
    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=phi_stack.nbytes)
    shm_buf = np.ndarray(phi_stack.shape, dtype=phi_stack.dtype, buffer=shm.buf)
    np.copyto(shm_buf, phi_stack)
    del phi_stack   # parent no longer needs its own copy

    # ── Run integrations ─────────────────────────────────────────────────────
    out_path = os.path.join(BASE, f"trajectories_{args.run}.csv")
    print(f"\nSimulating {n_ions} ions with {n_workers} worker(s) …")
    print(f"Output: {out_path}")
    t_sim_start = time.perf_counter()

    all_results = []
    try:
        if n_workers == 1:
            # Single-process: set globals directly (no spawn overhead)
            global _worker_phi_stack, _worker_grid
            _worker_phi_stack = shm_buf
            _worker_grid      = grid
            for arg in ion_args:
                result = _worker(arg)
                all_results.append(result)
        else:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=n_workers,
                          initializer=_shm_initializer,
                          initargs=(shm.name, shm_buf.shape, grid)) as pool:
                for result in pool.imap_unordered(_worker, ion_args):
                    all_results.append(result)
    finally:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass

    t_sim_end = time.perf_counter()

    # Sort results by ion number for deterministic output order
    all_results.sort(key=lambda r: r[0])

    # ── Write output CSV ─────────────────────────────────────────────────────
    with open(out_path, "w") as f:
        f.write("ion,time_us,x_mm,y_mm,z_mm\n")
        for ion_num, rows, info in all_results:
            for line in rows:
                f.write(line + "\n")

    total_rows = sum(len(r[1]) for r in all_results)
    print(f"\nWrote {total_rows} trajectory rows to {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Simulation summary ──────────────────────────────────────────────")
    for ion_num, rows, info in all_results:
        print(f"  ion {ion_num:3d}: {info['steps_accepted']:7d} steps accepted, "
              f"{info['t_sim_us']:.1f} µs simulated, "
              f"reason={info['reason']}")
    print(f"\nTotal wall time: {t_sim_end - t_sim_start:.1f} s")


if __name__ == "__main__":
    main()
