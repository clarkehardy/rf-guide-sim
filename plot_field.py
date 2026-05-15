"""
plot_field.py  –  Parse SIMION PA files and visualise the electric field
inside the Paul trap for any of the 10 electrodes.

Reads a paulTrap.paN file, computes E = -grad(V), and saves a two-panel PNG:
  Left  – X-Z cross-section at the trap axis showing potential + E vectors
  Right – Ez vs Z along the trap axis (RF screening visible)

All rods are along Z in the new geometry, so every electrode uses the X-Z slice.

PA binary format (56-byte header, confirmed for this project):
  [0:4]  int32 flags  [4:8] int32  [8:16] float64 scale_ref
  [16:20] int32 nx    [20:24] int32 ny    [24:28] int32 nz
  [28:32] int32       [32:40] float64 dx  [40:48] float64 dy  [48:56] float64 dz
  Data: nx*ny*nz float64, z outermost (slowest), x innermost (fastest).
  Other electrode surfaces: sign bit set.  This electrode: value ≈ 200000.

Usage:
    python plot_field.py                  # all DC electrodes (3, 4, 9, 10)
    python plot_field.py --elec 3         # load endcap U
    python plot_field.py --elec 5         # set-3 rod TL
    python plot_field.py --3d             # also open PyVista window
    python plot_field.py --screenshot s.png
"""

import argparse, os, struct, sys
import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Grid constants (must match pa_define in paulTrap.gem) ─────────────────────
# PLACEHOLDER: update NX, NY, NZ, DX, and GEM_OFF after the new pa_define is set.
NX, NY, NZ = 131, 91, 855   # grid points
DX         = 0.5             # mm per grid unit
HEADER     = 56              # bytes

# GEM index → Fusion world:  Fusion = i*DX + GEM_OFF
# PLACEHOLDER: must match the locate(...) block in paulTrap.gem.
GEM_OFF = np.array([-25.0, -8.0, -132.0])

# Trap axis (GEM indices) — shared by sets 1, 2, 3 (all rods parallel to Z).
# PLACEHOLDER: update after new Fusion geometry sets the (x, y) of the axis.
# The user adjusted the axis to coincide with the midpoint of the optical trap.
MAIN_IX = 50
MAIN_IY = 54

# Per-electrode metadata used for slice placement and STL overlay in 3-D view.
# z_gem is the Z (GEM, mm) at the geometric centre of the electrode body.
# PLACEHOLDER: replace every z_gem with the actual centroid Z from Fusion (in
# Fusion coords, add 132 mm if using the current default GEM_OFF).
_ELEC_INFO = {
    1:  dict(cross_plane='xz', z_gem=0.0, stl="rod_1_TL.stl",
             label="Sets 1+2 +RF (1)"),
    2:  dict(cross_plane='xz', z_gem=0.0, stl="rod_1_TR.stl",
             label="Sets 1+2 −RF (2)"),
    3:  dict(cross_plane='xz', z_gem=0.0, stl="endcap_load_U.stl",
             label="Load endcap U (3, V_endcap_load_U)"),
    4:  dict(cross_plane='xz', z_gem=0.0, stl="endcap_load_D.stl",
             label="Load endcap D (4, V_endcap_load_D)"),
    5:  dict(cross_plane='xz', z_gem=0.0, stl="rod_3_TL.stl",
             label="rod_3_TL (5, +RF3 + V_dc_3_TL)"),
    6:  dict(cross_plane='xz', z_gem=0.0, stl="rod_3_TR.stl",
             label="rod_3_TR (6, −RF3 + V_dc_3_TR)"),
    7:  dict(cross_plane='xz', z_gem=0.0, stl="rod_3_BL.stl",
             label="rod_3_BL (7, −RF3 + V_dc_3_BL)"),
    8:  dict(cross_plane='xz', z_gem=0.0, stl="rod_3_BR.stl",
             label="rod_3_BR (8, +RF3 + V_dc_3_BR)"),
    9:  dict(cross_plane='xz', z_gem=0.0, stl="endcap_optical_U.stl",
             label="Optical endcap U (9, V_endcap_optical_U)"),
    10: dict(cross_plane='xz', z_gem=0.0, stl="endcap_optical_D.stl",
             label="Optical endcap D (10, V_endcap_optical_D)"),
}


def gem_axes():
    x = np.arange(NX) * DX + GEM_OFF[0]
    y = np.arange(NY) * DX + GEM_OFF[1]
    z = np.arange(NZ) * DX + GEM_OFF[2]
    return x, y, z


# ── PA reader ─────────────────────────────────────────────────────────────────

def read_pa(path):
    fsize    = os.path.getsize(path)
    n_pts    = NX * NY * NZ
    expected = HEADER + n_pts * 8
    if fsize != expected:
        print(f"  WARNING: {os.path.basename(path)}: "
              f"size {fsize} != expected {expected}  ({NX}×{NY}×{NZ})")

    with open(path, "rb") as f:
        hdr = f.read(HEADER)
        raw = np.frombuffer(f.read(n_pts * 8), dtype="<f8").copy()

    nx_h = struct.unpack_from("<i", hdr, 16)[0]
    ny_h = struct.unpack_from("<i", hdr, 20)[0]
    nz_h = struct.unpack_from("<i", hdr, 24)[0]
    dx_h = struct.unpack_from("<d", hdr, 32)[0]
    print(f"  Header: dims {nx_h}×{ny_h}×{nz_h}  dx={dx_h:.3g} mm")

    elec_other = np.signbit(raw)
    max_val    = raw.max()
    elec_this  = raw > 0.5 * max_val
    V_norm     = np.clip(np.abs(raw) / max_val, 0.0, 1.0)

    V_norm     = V_norm.reshape(NZ, NY, NX)
    elec_other = elec_other.reshape(NZ, NY, NX)
    elec_this  = elec_this.reshape(NZ, NY, NX)

    print(f"  Other-electrode pts: {elec_other.sum()}  |  This-electrode pts: {elec_this.sum()}")
    return V_norm, elec_other, elec_this, dx_h if dx_h > 0 else DX


# ── Read pa file as volts (unit electrode = 1 V) ──────────────────────────────

def read_pa_volts(path):
    """Read a unit-potential PA and return V in volts.

    Conventions:
      - This-electrode boundary cells (stored as ~200000+N) → 1.0 V
      - Other-electrode boundary cells (sign-bit set)       → 0.0 V
      - Free-space cells                                    → stored/scale_ref V
    SIMION's scale_ref is read from the header (offset 8); typically 100000,
    i.e. a stored value of 100000 corresponds to 1 V at unit electrode potential.
    """
    n_pts = NX * NY * NZ
    with open(path, "rb") as f:
        hdr = f.read(HEADER)
        raw = np.frombuffer(f.read(n_pts * 8), dtype="<f8").copy()
    scale       = struct.unpack_from("<d", hdr, 8)[0]
    elec_other  = np.signbit(raw)
    # The electrode marker is ~2*scale + N (≈ 200001..200010 for scale=100000),
    # well above any free-cell stored value (which maxes at scale = 1 V).
    elec_this   = raw > 1.5 * scale
    V           = np.abs(raw) / scale
    V[elec_this]  = 1.0
    V[elec_other] = 0.0
    return V.reshape(NZ, NY, NX)


# ── Optical Paul trap centre (auto-detected from STL bboxes) ──────────────────

def _stl_bbox(name):
    """(lo, hi) bounding box for an STL, or None if absent."""
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        raw = f.read(n * 50)
    if n == 0 or len(raw) < n * 50:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n, 50)
    v   = np.frombuffer(arr[:, 12:48].tobytes(), dtype="<f4").reshape(-1, 3)
    return v.min(axis=0), v.max(axis=0)


def auto_optical_centre():
    """Return (x, y, z) of the optical Paul trap centre in Fusion world mm.

    Centre defined as:
      - (x, y): geometric centre of the four set-3 rod STL bboxes
      - z:       midpoint of the two optical endcap centroids
    Returns None if any required STL is missing.
    """
    rod_bbs = [_stl_bbox(f"rod_3_{s}.stl") for s in ("TL", "TR", "BL", "BR")]
    rod_bbs = [b for b in rod_bbs if b is not None]
    ec_U    = _stl_bbox("endcap_optical_U.stl")
    ec_D    = _stl_bbox("endcap_optical_D.stl")
    if not rod_bbs or ec_U is None or ec_D is None:
        return None
    lo = np.array([b[0] for b in rod_bbs])
    hi = np.array([b[1] for b in rod_bbs])
    cx = 0.5 * (lo[:, 0].min() + hi[:, 0].max())
    cy = 0.5 * (lo[:, 1].min() + hi[:, 1].max())
    cz = 0.25 * (ec_U[0][2] + ec_U[1][2] + ec_D[0][2] + ec_D[1][2])
    return float(cx), float(cy), float(cz)


# ── Voltage extraction from CSV ───────────────────────────────────────────────

def read_voltages_csv(csv_path, time_us):
    """Read voltages_N.csv and return ({electrode → V}, actual_time_us).

    RF amplitudes (V_RF, V_RF3) are treated as the instantaneous voltage on
    their respective electrodes (i.e. phase = 0 snapshot: cos(ωt+φ)=1).  Per-
    rod DC trims on set 3 are added to the appropriate RF rod.
    """
    rows   = []
    header = None
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if header is None:
                header = parts
                continue
            rows.append([float(x) for x in parts])
    if not rows:
        return {}, 0.0
    arr = np.array(rows)
    t_idx     = header.index("time_us")
    chosen_i  = int(np.argmin(np.abs(arr[:, t_idx] - time_us)))
    chosen    = arr[chosen_i]

    def get(name):
        return float(chosen[header.index(name)]) if name in header else 0.0

    V_RF, V_RF3 = get("V_RF"), get("V_RF3")
    voltages = {
        1:  +V_RF,
        2:  -V_RF,
        3:  get("V_endcap_load_U"),
        4:  get("V_endcap_load_D"),
        5:  +V_RF3 + get("V_dc_3_TL"),
        6:  -V_RF3 + get("V_dc_3_TR"),
        7:  -V_RF3 + get("V_dc_3_BL"),
        8:  +V_RF3 + get("V_dc_3_BR"),
        9:  get("V_endcap_optical_U"),
        10: get("V_endcap_optical_D"),
    }
    return voltages, float(chosen[t_idx])


# ── Total potential through a centre point ────────────────────────────────────

def plot_total(voltages, centre, span_mm, label):
    """Plot V_total(Δx,0,0), V_total(0,Δy,0), V_total(0,0,Δz) through `centre`.

    V_total = Σ_N V_N · pa_N_unit, where pa_N_unit is the SIMION unit-potential
    PA for electrode N (1 V on that electrode, 0 V on the others).

    Outputs `field_total_<label>.png` (or `field_total.png` if label is None).
    """
    x_f, y_f, z_f = gem_axes()
    cx, cy, cz    = centre

    ix0 = int(round((cx - GEM_OFF[0]) / DX))
    iy0 = int(round((cy - GEM_OFF[1]) / DX))
    iz0 = int(round((cz - GEM_OFF[2]) / DX))
    n   = int(np.ceil(span_mm / DX))
    ix_lo, ix_hi = max(0, ix0 - n), min(NX, ix0 + n + 1)
    iy_lo, iy_hi = max(0, iy0 - n), min(NY, iy0 + n + 1)
    iz_lo, iz_hi = max(0, iz0 - n), min(NZ, iz0 + n + 1)

    V_x = np.zeros(ix_hi - ix_lo, dtype=np.float64)
    V_y = np.zeros(iy_hi - iy_lo, dtype=np.float64)
    V_z = np.zeros(iz_hi - iz_lo, dtype=np.float64)

    n_loaded = 0
    for en in range(1, 11):
        v = voltages.get(en, 0.0)
        if abs(v) < 1e-12:
            continue
        pa_path = os.path.join(BASE, f"paulTrap.pa{en}")
        if not os.path.exists(pa_path):
            print(f"  [skip] paulTrap.pa{en} not found")
            continue
        print(f"  pa{en}: V = {v:+9.3f} V  — loading & accumulating")
        Vu = read_pa_volts(pa_path)
        V_x += v * Vu[iz0, iy0, ix_lo:ix_hi]
        V_y += v * Vu[iz0, iy_lo:iy_hi, ix0]
        V_z += v * Vu[iz_lo:iz_hi, iy0, ix0]
        del Vu
        n_loaded += 1
    print(f"  Accumulated contributions from {n_loaded} electrode(s)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, axis_pos, V_line, axis_name, other_a, other_b, color in [
        (axes[0], x_f[ix_lo:ix_hi] - cx, V_x, "x", ("y", cy), ("z", cz), "steelblue"),
        (axes[1], y_f[iy_lo:iy_hi] - cy, V_y, "y", ("x", cx), ("z", cz), "seagreen"),
        (axes[2], z_f[iz_lo:iz_hi] - cz, V_z, "z", ("x", cx), ("y", cy), "crimson"),
    ]:
        ax.plot(axis_pos, V_line, color=color, lw=1.8)
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.grid(True, alpha=0.18)
        ax.set_xlabel(rf"$\Delta {axis_name}$  (mm)")
        ax.set_title(f"V along {axis_name}  "
                     f"({other_a[0]}={other_a[1]:.3f}, {other_b[0]}={other_b[1]:.3f})")
    axes[0].set_ylabel("V  (volts)")

    nonzero = [f"e{n}={voltages.get(n,0):+.0f}V"
               for n in range(1, 11) if abs(voltages.get(n, 0)) > 1e-12]
    fig.suptitle(
        f"Total potential through optical Paul trap centre "
        f"({cx:.2f}, {cy:.2f}, {cz:.2f}) mm  —  span ±{span_mm:g} mm\n"
        f"{'  '.join(nonzero) if nonzero else 'all electrodes at 0 V'}",
        fontsize=10)
    fig.tight_layout()

    out_name = f"field_total_{label}.png" if label else "field_total.png"
    out      = os.path.join(BASE, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    return fig


# ── NaN fill ──────────────────────────────────────────────────────────────────

def _fill_nan(V):
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        return np.where(np.isnan(V), 0.0, V)
    filled = V.copy()
    for _ in range(60):
        nan_now = np.isnan(filled)
        if not nan_now.any():
            break
        vals = np.where(nan_now, 0.0, filled)
        w    = (~nan_now).astype(float)
        vf   = uniform_filter(vals, size=3)
        wf   = uniform_filter(w,    size=3)
        with np.errstate(invalid="ignore", divide="ignore"):
            est = np.where(wf > 1e-12, vf / wf, np.nan)
        filled = np.where(nan_now, est, filled)
    return np.where(np.isnan(filled), 0.0, filled)


# ── Electric field ─────────────────────────────────────────────────────────────

def efield(V, elec_other, elec_this, dx=DX):
    V_work = V.copy()
    V_work[elec_other] = 0.0
    V_work[elec_this]  = 1.0
    if np.isnan(V_work).any():
        V_work = _fill_nan(V_work)
    gz, gy, gx = np.gradient(V_work, dx, dx, dx)
    return -gx, -gy, -gz


# ── Shared colormap helper ────────────────────────────────────────────────────

def _colormap_panel(ax, H, A, V_plot, E1_sl, E2_sl, eo_sl, et_sl,
                    hlabel, alabel, title):
    """
    Render a 2-D potential map with E-field quivers onto ax.
    H, A  : 2-D coordinate arrays (nh, na) for horizontal and vertical axes.
    V_plot, E1_sl, E2_sl, eo_sl, et_sl : 2-D arrays (nh, na).
    E1_sl is the horizontal E component, E2_sl the vertical.
    """
    v_max = max(0.05, float(np.nanmax(V_plot))) if np.any(~np.isnan(V_plot)) else 1.0
    v_max = min(v_max, 1.0)
    im = ax.pcolormesh(H, A, V_plot, cmap="RdBu_r", vmin=0, vmax=v_max,
                       shading="nearest", rasterized=True)
    clabel = (f"Unit potential  (max = {v_max:.3f} × electrode V)"
              if v_max < 0.99 else "Unit potential  (electrode = 1 V)")
    plt.colorbar(im, ax=ax, label=clabel, shrink=0.9)

    ax.contourf(H, A, et_sl.astype(float), levels=[0.5, 1.5],
                colors="goldenrod", alpha=0.7, zorder=2)
    ax.contourf(H, A, eo_sl.astype(float), levels=[0.5, 1.5],
                colors="dimgrey",   alpha=0.55, zorder=2)

    nh, na = H.shape
    sh = max(1, nh // 25)
    sa = max(1, na // 8)
    mag = np.hypot(E1_sl[::sh, ::sa], E2_sl[::sh, ::sa])
    mag[mag < 1e-12] = 1e-12
    ax.quiver(H[::sh, ::sa], A[::sh, ::sa],
              E1_sl[::sh, ::sa] / mag, E2_sl[::sh, ::sa] / mag,
              mag, cmap="YlOrRd", scale=28, alpha=0.9, zorder=3,
              width=0.004, headwidth=3)

    ax.set_xlabel(hlabel)
    ax.set_ylabel(alabel)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18)


# ── X-Z panel (main trap) ─────────────────────────────────────────────────────

def _plot_xz(ax1, ax2, V, Ex, Ey, Ez, elec_other, elec_this,
             x_f, y_f, z_f, info, label):
    zc_gem = info.get("z_gem", 30.0)
    iz0 = max(0, int((zc_gem - 50) / DX))
    iz1 = min(NZ - 1, int((zc_gem + 50) / DX))
    z_sub = z_f[iz0:iz1 + 1]

    V_sl  = V         [iz0:iz1 + 1, MAIN_IY, :]
    Ez_sl = Ez        [iz0:iz1 + 1, MAIN_IY, :]
    Ex_sl = Ex        [iz0:iz1 + 1, MAIN_IY, :]
    eo_sl = elec_other[iz0:iz1 + 1, MAIN_IY, :]
    et_sl = elec_this [iz0:iz1 + 1, MAIN_IY, :]

    V_disp = V_sl.copy()
    V_disp[eo_sl] = 0.0
    V_disp[et_sl] = 1.0
    V_plot = np.where(eo_sl | et_sl, np.nan, V_disp)

    # pcolormesh(ZZ, XX, data): ZZ[iz,ix]=z, XX[iz,ix]=x → data[iz,ix]
    ZZ, XX = np.meshgrid(z_sub, x_f, indexing='ij')   # (nz_sub, NX)

    _colormap_panel(ax1, ZZ, XX, V_plot, Ez_sl, Ex_sl, eo_sl, et_sl,
                    "Z  (mm, Fusion world)", "X  (mm, Fusion world)",
                    f"X-Z cross-section  (Y = {y_f[MAIN_IY]:.2f} mm)\n{label}")

    # Right panel: Ez along Z at trap axis
    ax2.plot(z_f, Ez[:, MAIN_IY, MAIN_IX], color="steelblue", lw=1.8,
             label=r"$E_z$  (V/mm per unit V)")
    ax2b = ax2.twinx()
    ax2b.plot(z_f, V[:, MAIN_IY, MAIN_IX], color="crimson", lw=1.3,
              ls="--", alpha=0.75, label="V  (unit potential)")

    zc_f = zc_gem + GEM_OFF[2]
    ax2.axvline(zc_f, color="dimgrey", lw=1, ls=":", alpha=0.8,
                label=f"Electrode  z = {zc_f:.1f} mm")
    # PLACEHOLDER: update Z ranges to shade the new rod sections (sets 1, 2, 3 in Fusion world).
    # ax2.axvspan(z_set1_min, z_set1_max, alpha=0.07, color="navy",      label="Set 1 (loading)")
    # ax2.axvspan(z_set2_min, z_set2_max, alpha=0.07, color="purple",    label="Set 2 (RF guide)")
    # ax2.axvspan(z_set3_min, z_set3_max, alpha=0.07, color="darkgreen", label="Set 3 (optical PT)")

    ax2.set_xlabel("Z  (mm, Fusion world)")
    ax2.set_ylabel(r"$E_z$  (V/mm per unit V)", color="steelblue")
    ax2b.set_ylabel("Unit potential  V",         color="crimson")
    ax2.set_title("Axial field along trap centre\n(field screening by RF rods visible)")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2b.tick_params(axis="y", labelcolor="crimson")
    l1, n1 = ax2.get_legend_handles_labels()
    l2, n2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1 + l2, n1 + n2, fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.18)


# ── Y-Z panel (perp-trap) ─────────────────────────────────────────────────────

def _plot_yz(ax1, ax2, V, Ex, Ey, Ez, elec_other, elec_this,
             x_f, y_f, z_f, info, label):
    zc_gem = info.get("z_gem", 408.0)
    ix_sl  = min(NX - 1, max(0, round(info.get("x_gem", MAIN_IX * DX) / DX)))

    iz_c  = round(zc_gem / DX)
    half  = round(20.0 / DX)
    iz0   = max(0, iz_c - half)
    iz1   = min(NZ - 1, iz_c + half)
    z_sub = z_f[iz0:iz1 + 1]

    # YZ slice at ix_sl: V[iz, iy, ix] → arrays are (nz_sub, NY)
    V_sl  = V         [iz0:iz1 + 1, :, ix_sl]
    Ey_sl = Ey        [iz0:iz1 + 1, :, ix_sl]
    Ez_sl = Ez        [iz0:iz1 + 1, :, ix_sl]
    eo_sl = elec_other[iz0:iz1 + 1, :, ix_sl]
    et_sl = elec_this [iz0:iz1 + 1, :, ix_sl]

    V_disp = V_sl.copy()
    V_disp[eo_sl] = 0.0
    V_disp[et_sl] = 1.0
    V_plot = np.where(eo_sl | et_sl, np.nan, V_disp)

    # pcolormesh(ZZ, YY, data): ZZ[iz,iy]=z, YY[iz,iy]=y → data[iz,iy]
    ZZ, YY = np.meshgrid(z_sub, y_f, indexing='ij')   # (nz_sub, NY)

    _colormap_panel(ax1, ZZ, YY, V_plot, Ez_sl, Ey_sl, eo_sl, et_sl,
                    "Z  (mm, Fusion world)", "Y  (mm, Fusion world)",
                    f"Y-Z cross-section  (X = {x_f[ix_sl]:.1f} mm)\n{label}")

    # Right panel: Ex along X at perp-trap axis
    Ex_ax = Ex[PERP_IZ, PERP_IY, :]
    V_ax  = V [PERP_IZ, PERP_IY, :]

    ax2.plot(x_f, Ex_ax, color="steelblue", lw=1.8,
             label=r"$E_x$  (V/mm per unit V)")
    ax2b = ax2.twinx()
    ax2b.plot(x_f, V_ax, color="crimson", lw=1.3, ls="--", alpha=0.75,
              label="V  (unit potential)")

    # Shade the trap region (between lens holders: Fusion X ≈ −6 to +5 mm)
    ax2.axvspan(-6.1, 4.9, alpha=0.10, color="mediumpurple", label="Trap region")
    ax2.axvline(x_f[ix_sl], color="dimgrey", lw=1, ls=":", alpha=0.8,
                label=f"Slice X = {x_f[ix_sl]:.1f} mm")

    ax2.set_xlabel("X  (mm, Fusion world)")
    ax2.set_ylabel(r"$E_x$  (V/mm per unit V)", color="steelblue")
    ax2b.set_ylabel("Unit potential  V",         color="crimson")
    ax2.set_title(f"Axial field along perp-trap axis (X)\n"
                  f"(Y = {y_f[PERP_IY]:.2f} mm,  Z = {z_f[PERP_IZ]:.1f} mm)")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2b.tick_params(axis="y", labelcolor="crimson")
    l1, n1 = ax2.get_legend_handles_labels()
    l2, n2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1 + l2, n1 + n2, fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.18)


# ── 2-D figure ────────────────────────────────────────────────────────────────

def plot_2d(label, elec_num, V, Ex, Ey, Ez, elec_other, elec_this):
    x_f, y_f, z_f = gem_axes()
    info = _ELEC_INFO.get(elec_num, {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    if info.get("cross_plane") == "yz":
        _plot_yz(ax1, ax2, V, Ex, Ey, Ez, elec_other, elec_this,
                 x_f, y_f, z_f, info, label)
    else:
        _plot_xz(ax1, ax2, V, Ex, Ey, Ez, elec_other, elec_this,
                 x_f, y_f, z_f, info, label)

    fig.suptitle(label, fontweight="bold", fontsize=12)
    fig.tight_layout()
    out = os.path.join(BASE, f"field_{elec_num}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    return fig


# ── 3-D PyVista view ──────────────────────────────────────────────────────────

def plot_3d(label, elec_num, V, Ex, Ey, Ez, elec_other, elec_this, screenshot=None):
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed — skipping 3-D view.  pip install pyvista")
        return

    x_f, y_f, z_f = gem_axes()
    info   = _ELEC_INFO.get(elec_num, {})
    zc_gem = info.get("z_gem", 30.0)
    iz0 = max(0, int((zc_gem - 50) / DX))
    iz1 = min(NZ - 1, int((zc_gem + 50) / DX))
    nz_s = iz1 - iz0 + 1

    V_s  = V  [iz0:iz1 + 1, :, :]
    Ex_s = Ex [iz0:iz1 + 1, :, :]
    Ey_s = Ey [iz0:iz1 + 1, :, :]
    Ez_s = Ez [iz0:iz1 + 1, :, :]

    V_disp = V_s.copy()
    V_disp[elec_other[iz0:iz1 + 1, :, :]] = 0.0
    V_disp[elec_this [iz0:iz1 + 1, :, :]] = 1.0

    grid = pv.ImageData()
    grid.dimensions = (NX, NY, nz_s)
    grid.origin     = (x_f[0], y_f[0], z_f[iz0])
    grid.spacing    = (DX, DX, DX)
    grid.point_data["V"]   = V_disp.flatten(order="C")
    Emag = np.sqrt(Ex_s**2 + Ey_s**2 + Ez_s**2)
    grid.point_data["|E|"] = Emag.flatten(order="C")
    grid.point_data["E"]   = np.stack([Ex_s, Ey_s, Ez_s], axis=-1).reshape(-1, 3, order="C")

    pl = pv.Plotter(off_screen=(screenshot is not None),
                    title=f"Paul trap field — {label}")
    pl.set_background("white")

    for lev in [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]:
        iso = grid.contour([lev], scalars="V")
        if iso.n_points:
            pl.add_mesh(iso, opacity=0.20, scalars="V", cmap="RdBu_r",
                        clim=[0, 1], show_scalar_bar=False)

    thresh = grid.threshold(1e-6, scalars="|E|")
    if thresh.n_points > 400:
        ids = np.linspace(0, thresh.n_points - 1, 400, dtype=int)
        thresh = thresh.extract_points(ids, include_cells=False)
    if thresh.n_points:
        glyphs = thresh.glyph(orient="E", scale="|E|", factor=0.25, geom=pv.Arrow())
        pl.add_mesh(glyphs, color="darkorange", opacity=0.85)

    stl_name = info.get("stl")
    if stl_name:
        stl_path = os.path.join(BASE, stl_name)
        if os.path.exists(stl_path):
            pl.add_mesh(pv.read(stl_path), color="goldenrod",
                        opacity=0.70, smooth_shading=True)

    dummy = grid.contour([0.5], scalars="V")
    if dummy.n_points:
        pl.add_mesh(dummy, opacity=0, scalars="V", show_scalar_bar=True,
                    scalar_bar_args=dict(title="Unit potential V",
                                        title_font_size=13, label_font_size=11,
                                        n_labels=5, fmt="%.2f",
                                        width=0.5, height=0.06,
                                        position_x=0.25, position_y=0.02,
                                        color="black"))
    pl.add_axes(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")

    if screenshot:
        pl.show(screenshot=screenshot, auto_close=True)
        print(f"  Saved: {screenshot}")
    else:
        pl.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elec", type=int,
                    choices=list(range(1, 11)), default=None,
                    help="Electrode number 1–10 (default: all DC electrodes)")
    ap.add_argument("--3d",   dest="show3d", action="store_true",
                    help="Open interactive 3-D PyVista window")
    ap.add_argument("--screenshot", default=None,
                    help="Save 3-D view to PNG (implies --3d)")
    # --total mode: sum V_N · pa_N along x/y/z through a centre point.
    ap.add_argument("--total", action="store_true",
                    help="Plot the total potential V = Σ V_N · pa_N along the "
                         "three Cartesian axes through a centre point.  "
                         "Useful for comparing field shape with and without "
                         "the dielectric step.")
    ap.add_argument("--centre", "--center", dest="centre",
                    type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Fusion-world centre (mm) for --total.  Default: "
                         "the optical Paul trap centre, auto-detected from "
                         "rod_3_*.stl and endcap_optical_*.stl bboxes.")
    ap.add_argument("--span", type=float, default=10.0,
                    help="Half-width of each 1-D plot in --total mode "
                         "(mm, default 10).")
    ap.add_argument("--volts-file", default="voltages_1.csv",
                    help="Voltage schedule CSV to pull electrode voltages "
                         "from for --total (default: voltages_1.csv).")
    ap.add_argument("--time", type=float, default=0.0,
                    help="Time (µs) at which to sample voltages from the CSV "
                         "(default: 0).  RF amplitudes are taken as the "
                         "instantaneous voltage (phase = 0 snapshot).")
    ap.add_argument("--voltages", type=float, nargs=10, default=None,
                    metavar=("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10"),
                    help="Explicit voltages (V) for electrodes 1..10, "
                         "overriding --volts-file/--time.")
    ap.add_argument("--label", default=None,
                    help="Suffix for the output filename in --total mode "
                         "(field_total_<label>.png).  Useful for saving "
                         "before/after-dielectric pairs.")
    args   = ap.parse_args()
    do_3d  = args.show3d or bool(args.screenshot)

    # ── --total mode ─────────────────────────────────────────────────────────
    if args.total:
        # Centre
        if args.centre is not None:
            centre = tuple(args.centre)
        else:
            centre = auto_optical_centre()
            if centre is None:
                sys.exit("ERROR: cannot auto-detect optical Paul trap centre "
                         "(some rod_3_*.stl or endcap_optical_*.stl missing).  "
                         "Pass --centre X Y Z explicitly.")
            print(f"Optical Paul trap centre (auto): "
                  f"({centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f}) mm")
        # Voltages
        if args.voltages is not None:
            voltages = {n + 1: v for n, v in enumerate(args.voltages)}
            print("Voltages from --voltages CLI:")
        else:
            vpath = os.path.join(BASE, args.volts_file)
            if not os.path.exists(vpath):
                sys.exit(f"Voltage file not found: {vpath}")
            voltages, t_used = read_voltages_csv(vpath, args.time)
            print(f"Voltages from {args.volts_file} at t={t_used:.0f} µs "
                  f"(requested {args.time:.0f} µs):")
        for n in range(1, 11):
            print(f"  elec {n:2d}: {voltages.get(n, 0.0):+9.3f} V")
        plot_total(voltages, centre, args.span, args.label)
        plt.show()
        return

    # Default: all DC endcaps (the four bias electrodes — rest are RF-driven)
    elecs = [args.elec] if args.elec else [3, 4, 9, 10]

    for en in elecs:
        pa_path = os.path.join(BASE, f"paulTrap.pa{en}")
        if not os.path.exists(pa_path):
            print(f"[skip] {pa_path} not found")
            continue
        label = _ELEC_INFO.get(en, {}).get("label", f"Electrode {en}")
        print(f"\n── {label} ──")
        print(f"  Reading {pa_path} …")
        V, eo, et, dx = read_pa(pa_path)
        print("  Computing E = -grad(V) …")
        Ex, Ey, Ez = efield(V, eo, et, dx)
        plot_2d(label, en, V, Ex, Ey, Ez, eo, et)
        if do_3d:
            scr = None
            if args.screenshot:
                base_name, ext = os.path.splitext(args.screenshot)
                scr = f"{base_name}_{en}{ext}"
            plot_3d(label, en, V, Ex, Ey, Ez, eo, et, screenshot=scr)

    plt.show()


if __name__ == "__main__":
    main()
