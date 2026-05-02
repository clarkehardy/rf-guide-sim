"""
plot_field.py  –  Parse SIMION PA files and visualise the electric field
inside the Paul trap for any electrode.

Reads a paulTrap.paN file (where N is an electrode number), computes
E = -grad(V), and saves a two-panel PNG:
  Left  – X-Z cross-section at trap centre showing potential + E vectors
  Right – Axial E_z along the trap centre demonstrating field screening

PA binary format (56-byte header, confirmed for this project):
  [0:4]  int32 flags  [4:8] int32  [8:16] float64 scale_ref
  [16:20] int32 nx    [20:24] int32 ny    [24:28] int32 nz
  [28:32] int32       [32:40] float64 dx  [40:48] float64 dy  [48:56] float64 dz
  Data: nx*ny*nz float64, z outermost (slowest), x innermost (fastest).
  Other electrode surfaces: sign bit set (≈ −0.0).
  This electrode surface: value ≈ max (≈ 200000).
  Free space: 0 … max.  Normalise by raw.max() to get 0–1.

Usage:
    python plot_field.py                  # all DC electrodes (3, 6, 7, 8)
    python plot_field.py --elec 3         # left endcap
    python plot_field.py --elec 6         # ring_L  (in the gap — shows real field)
    python plot_field.py --elec 7         # ring_R
    python plot_field.py --elec 8         # right endcap
    python plot_field.py --3d             # also open PyVista window
    python plot_field.py --screenshot s.png
"""

import argparse, os, struct, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Grid constants (must match pa_define in paulTrap.gem) ─────────────────────
NX, NY, NZ = 29, 29, 765   # grid points  (14/0.5+1, 14/0.5+1, 382/0.5+1)
DX         = 0.5            # mm per grid unit
HEADER     = 56             # bytes

# GEM → Fusion world:  Fusion = GEM + OFFSET
# (from paulTrap.gem: GEM x = Fusion X + 7, GEM y = Fusion Y − 12.05, GEM z = Fusion Z + 131)
GEM_OFF = np.array([-7.0, 12.05, -131.0])

# Electrode GEM z-centres (mm) and STL filenames
_ELEC_INFO = {
    3: dict(z_gem=16.0,    stl="endcap_L.stl",  label="Left endcap  (electrode 3, V_endcap)"),
    6: dict(z_gem=198.4,   stl="ring_L.stl",    label="Ring L  (electrode 6, V_ring_L)"),
    7: dict(z_gem=241.4,   stl="ring_R.stl",    label="Ring R  (electrode 7, V_ring_R)"),
    8: dict(z_gem=49.76,   stl="endcap_R.stl",  label="Right endcap (electrode 8, V_endcap_R)"),
}
# Ring GEM z from animate.py: ring_L_z=67.4, ring_R_z=110.4 in Fusion  → GEM = Fusion + 131
# ring_L GEM z = 67.4+131 = 198.4;  ring_R GEM z = 110.4+131 = 241.4


def gem_axes():
    x = np.arange(NX) * DX + GEM_OFF[0]   # Fusion X values
    y = np.arange(NY) * DX + GEM_OFF[1]   # Fusion Y values
    z = np.arange(NZ) * DX + GEM_OFF[2]   # Fusion Z values
    return x, y, z


# ── PA reader ─────────────────────────────────────────────────────────────────

def read_pa(path):
    """
    Parse one SIMION fast-adjust PA binary file.

    Returns
    -------
    V_norm  : ndarray (NZ, NY, NX) float64 – unit potential 0..1,  V[iz, iy, ix]
    elec_other : bool ndarray – True where another electrode surface (0 V)
    elec_this  : bool ndarray – True where this electrode surface (1 V)
    dx_pa   : float – grid spacing from header (mm)
    """
    fsize = os.path.getsize(path)
    n_pts = NX * NY * NZ
    expected = HEADER + n_pts * 8
    if fsize != expected:
        print(f"  WARNING: {os.path.basename(path)}: "
              f"size {fsize} != expected {expected}  "
              f"(header={HEADER}, {NX}×{NY}×{NZ} points)")

    with open(path, "rb") as f:
        hdr = f.read(HEADER)
        raw = np.frombuffer(f.read(n_pts * 8), dtype="<f8").copy()

    nx_h  = struct.unpack_from("<i", hdr, 16)[0]
    ny_h  = struct.unpack_from("<i", hdr, 20)[0]
    nz_h  = struct.unpack_from("<i", hdr, 24)[0]
    dx_h  = struct.unpack_from("<d", hdr, 32)[0]
    print(f"  Header: dims {nx_h}×{ny_h}×{nz_h}  dx={dx_h:.3g} mm")

    elec_other = np.signbit(raw)                     # sign bit set → other electrode at 0 V
    max_val    = raw.max()
    elec_this  = raw > 0.5 * max_val                 # large positive → this electrode at 1 V
    V_norm     = np.clip(np.abs(raw) / max_val, 0.0, 1.0)

    # Reshape: z outermost, x innermost → V[iz, iy, ix]
    V_norm     = V_norm.reshape(NZ, NY, NX)
    elec_other = elec_other.reshape(NZ, NY, NX)
    elec_this  = elec_this.reshape(NZ, NY, NX)

    n_eo = elec_other.sum();  n_et = elec_this.sum()
    print(f"  Other-electrode pts: {n_eo}  |  This-electrode pts: {n_et}")
    return V_norm, elec_other, elec_this, dx_h if dx_h > 0 else DX


# ── NaN fill for gradient (iterative dilation) ────────────────────────────────

def _fill_nan(V):
    """Fill NaN by iterative box-filter dilation from valid neighbours."""
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
    """
    Compute E = -grad(V).

    Electrode interiors are set to their boundary value (0 or 1) before
    differentiation so the gradient inside conductors is near zero.
    Returns Ex, Ey, Ez each (NZ, NY, NX) in V/mm (unit-potential / mm).
    """
    V_work = V.copy()
    V_work[elec_other] = 0.0
    V_work[elec_this]  = 1.0

    any_nan = np.isnan(V_work)
    if any_nan.any():
        V_work = _fill_nan(V_work)

    # V[iz, iy, ix] → gradient returns (dV/dz, dV/dy, dV/dx)
    gz, gy, gx = np.gradient(V_work, dx, dx, dx)
    return -gx, -gy, -gz


# ── 2-D plots ─────────────────────────────────────────────────────────────────

def plot_2d(label, elec_num, V, Ex, Ey, Ez, elec_other, elec_this):
    """
    Two-panel figure:
      Left  – X-Z cross-section at Y = trap centre showing potential + E vectors.
      Right – Axial E_z decay along trap axis, illustrating field screening by rods.
    """
    x_f, y_f, z_f = gem_axes()
    iy_c = NY // 2                          # y = trap centre
    ix_c = NX // 2

    zc_gem = _ELEC_INFO.get(elec_num, {}).get("z_gem", 30.0)
    # Z window: ±50 mm around the electrode
    iz0 = max(0, int((zc_gem - 50) / DX))
    iz1 = min(NZ - 1, int((zc_gem + 50) / DX))
    z_sub = z_f[iz0:iz1 + 1]

    # V[iz, iy, ix] → XZ slice at iy_c:  shape (nz_sub, NX)
    V_sl  = V         [iz0:iz1 + 1, iy_c, :]
    Ez_sl = Ez        [iz0:iz1 + 1, iy_c, :]
    Ex_sl = Ex        [iz0:iz1 + 1, iy_c, :]
    eo_sl = elec_other[iz0:iz1 + 1, iy_c, :]
    et_sl = elec_this [iz0:iz1 + 1, iy_c, :]
    any_e = eo_sl | et_sl

    # Electrode potential for display: 0 for other, 1 for this
    V_disp = V_sl.copy()
    V_disp[eo_sl] = 0.0
    V_disp[et_sl] = 1.0

    XX, ZZ = np.meshgrid(x_f, z_sub)    # both (nz_sub, NX);  XX[iz,ix]=x, ZZ[iz,ix]=z

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # ── Left panel: X-Z potential map + E vectors ─────────────────────────────
    V_plot = np.where(any_e, np.nan, V_disp)
    # Auto-scale: cap at 1.0 but show full dynamic range if max < 0.1
    v_max_plot = max(0.05, float(np.nanmax(V_plot))) if np.any(~np.isnan(V_plot)) else 1.0
    v_max_plot = min(v_max_plot, 1.0)
    im = ax1.pcolormesh(ZZ, XX, V_plot,
                        cmap="RdBu_r", vmin=0, vmax=v_max_plot,
                        shading="nearest", rasterized=True)
    cbar_label = (f"Unit potential  (max = {v_max_plot:.3f} × electrode V)"
                  if v_max_plot < 0.99 else "Unit potential  (electrode = 1 V)")
    plt.colorbar(im, ax=ax1, label=cbar_label, shrink=0.9)

    # Electrode cross-hatching
    ax1.contourf(ZZ, XX, et_sl.astype(float), levels=[0.5, 1.5],
                 colors="goldenrod", alpha=0.7, zorder=2)
    ax1.contourf(ZZ, XX, eo_sl.astype(float), levels=[0.5, 1.5],
                 colors="dimgrey",  alpha=0.55, zorder=2)

    # Normalised E-field arrows (colour = magnitude)
    # slices: axis-0 = iz, axis-1 = ix in the (nz_sub, NX) arrays
    sz = max(1, (iz1 - iz0) // 25)
    sx = max(1, NX // 8)
    mag = np.hypot(Ex_sl[::sz, ::sx], Ez_sl[::sz, ::sx])
    mag[mag < 1e-12] = 1e-12
    ax1.quiver(ZZ[::sz, ::sx], XX[::sz, ::sx],
               Ez_sl[::sz, ::sx] / mag,
               Ex_sl[::sz, ::sx] / mag,
               mag, cmap="YlOrRd", scale=28, alpha=0.9, zorder=3,
               width=0.004, headwidth=3)

    ax1.set_xlabel("Z  (mm, Fusion world)")
    ax1.set_ylabel("X  (mm, Fusion world)")
    ax1.set_title(f"X-Z cross-section  (Y = trap centre)\n{label}")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.18)

    # ── Right panel: axial E_z decay ──────────────────────────────────────────
    # V[iz, iy, ix] → vary iz along full Z axis at trap centre (ix_c, iy_c)
    Ez_ax = Ez[:, iy_c, ix_c]
    V_ax  = V [:, iy_c, ix_c]

    ax2.plot(z_f, Ez_ax, color="steelblue", lw=1.8, label=r"$E_z$  (V/mm per unit V)")
    ax2b = ax2.twinx()
    ax2b.plot(z_f, V_ax, color="crimson", lw=1.3, ls="--", alpha=0.75,
              label="V  (unit potential)")

    zc_f = zc_gem + GEM_OFF[2]
    ax2.axvline(zc_f, color="dimgrey", lw=1, ls=":", alpha=0.8,
                label=f"Electrode  z = {zc_f:.1f} mm")

    # Shade the rod sections (left rods: Fusion z = -131 → +75.3 mm
    #                          right rods: Fusion z = +102.4 → +251 mm)
    ax2.axvspan(-131.0, 75.3,  alpha=0.07, color="navy",   label="Left rod section")
    ax2.axvspan(102.4,  251.0, alpha=0.07, color="purple", label="Right rod section")

    ax2.set_xlabel("Z  (mm, Fusion world)")
    ax2.set_ylabel(r"$E_z$  (V/mm per unit V)",  color="steelblue")
    ax2b.set_ylabel("Unit potential  V",          color="crimson")
    ax2.set_title("Axial field along trap centre\n(field screening by RF rods visible)")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2b.tick_params(axis="y", labelcolor="crimson")
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.18)

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
    zc_gem = _ELEC_INFO.get(elec_num, {}).get("z_gem", 30.0)
    iz0 = max(0, int((zc_gem - 50) / DX))
    iz1 = min(NZ - 1, int((zc_gem + 50) / DX))
    nz_s = iz1 - iz0 + 1

    # V[iz, iy, ix] → z-window sub-arrays have shape (nz_s, NY, NX)
    V_s  = V  [iz0:iz1 + 1, :, :]
    Ex_s = Ex [iz0:iz1 + 1, :, :]
    Ey_s = Ey [iz0:iz1 + 1, :, :]
    Ez_s = Ez [iz0:iz1 + 1, :, :]
    et_s = elec_this[iz0:iz1 + 1, :, :]

    # Set electrode surface to boundary value before display
    V_disp = V_s.copy()
    V_disp[elec_other[iz0:iz1 + 1, :, :]] = 0.0
    V_disp[et_s] = 1.0

    # PyVista ImageData needs x-fastest ordering.
    # V_disp shape is (nz_s, NY, NX): C-order flatten → ix fastest → x fastest ✓
    grid = pv.ImageData()
    grid.dimensions = (NX, NY, nz_s)
    grid.origin     = (x_f[0], y_f[0], z_f[iz0])
    grid.spacing    = (DX, DX, DX)
    grid.point_data["V"]  = V_disp.flatten(order="C")

    Emag = np.sqrt(Ex_s**2 + Ey_s**2 + Ez_s**2)
    grid.point_data["|E|"] = Emag.flatten(order="C")
    Evec = np.stack([Ex_s, Ey_s, Ez_s], axis=-1)   # shape (nz_s, NY, NX, 3)
    grid.point_data["E"]   = Evec.reshape(-1, 3, order="C")

    pl = pv.Plotter(off_screen=(screenshot is not None),
                    title=f"Paul trap field — {label}")
    pl.set_background("white")

    # Potential isosurfaces (semi-transparent, coloured by V)
    for lev in [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]:
        iso = grid.contour([lev], scalars="V")
        if iso.n_points:
            pl.add_mesh(iso, opacity=0.20, scalars="V", cmap="RdBu_r",
                        clim=[0, 1], show_scalar_bar=False)

    # E-field glyph arrows (subsample to avoid clutter)
    thresh = grid.threshold(1e-6, scalars="|E|")
    if thresh.n_points > 400:
        # Sample uniformly using cell centres subsample
        ids = np.linspace(0, thresh.n_points - 1, 400, dtype=int)
        thresh = thresh.extract_points(ids, include_cells=False)
    if thresh.n_points:
        glyphs = thresh.glyph(orient="E", scale="|E|", factor=0.25,
                              geom=pv.Arrow())
        pl.add_mesh(glyphs, color="darkorange", opacity=0.85)

    # Electrode STL for context
    stl_name = _ELEC_INFO.get(elec_num, {}).get("stl")
    if stl_name:
        stl_path = os.path.join(BASE, stl_name)
        if os.path.exists(stl_path):
            pl.add_mesh(pv.read(stl_path), color="goldenrod",
                        opacity=0.70, smooth_shading=True)

    # Add a dummy mesh just for the scalar bar
    dummy = grid.contour([0.5], scalars="V")
    if dummy.n_points:
        pl.add_mesh(dummy, opacity=0, scalars="V",
                    show_scalar_bar=True,
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
    ap.add_argument("--elec",  type=int, choices=[3, 6, 7, 8], default=None,
                    help="Which electrode (default: all DC electrodes 3,6,7,8)")
    ap.add_argument("--3d",    dest="show3d", action="store_true",
                    help="Open interactive 3-D PyVista window")
    ap.add_argument("--screenshot", default=None,
                    help="Save 3-D view to PNG (implies --3d)")
    args = ap.parse_args()
    do_3d = args.show3d or bool(args.screenshot)

    elecs = [args.elec] if args.elec else [3, 6, 7, 8]

    for en in elecs:
        pa_path = os.path.join(BASE, f"paulTrap.pa{en}")
        if not os.path.exists(pa_path):
            print(f"[skip] {pa_path} not found")
            continue
        info  = _ELEC_INFO.get(en, {})
        label = info.get("label", f"Electrode {en}")
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
