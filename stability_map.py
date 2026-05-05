"""
stability_map.py  –  Interactive lens-holder bias explorer for the perpendicular trap.

Loads SIMION unit-potential PA files for electrodes 11 (trapping lens holder) and
12 (collection lens holder), superimposes them at adjustable voltages, and shows
the combined potential field around the trap.  Drag the sliders to find a combination
that creates a confining well along the trap axis (X direction).

Panels
------
  Top-left  : X–Z cross-section (heatmap) at perp-trap axis Y
  Top-right : X–Y cross-section (heatmap) at perp-trap axis Z
  Bottom    : 1-D potential + E_x along the trap axis X, with stability annotation

Usage
-----
    python stability_map.py                       # interactive (default V11=50 V12=80)
    python stability_map.py --v11 50 --v12 80     # set initial biases (V)
    python stability_map.py --3d                  # also open PyVista 3-D window
    python stability_map.py --3d --screenshot out.png
"""

import argparse, os, struct, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Grid constants (must match pa_define in paulTrap.gem) ─────────────────────
NX, NY, NZ = 131, 91, 855
DX         = 0.5
HEADER     = 56
GEM_OFF    = np.array([-25.0, -8.0, -132.0])   # Fusion = i*DX + GEM_OFF

# Perp-trap axis (GEM grid indices; see plot_field.py)
PERP_IY = 55    # → Fusion Y = 19.50 mm
PERP_IZ = 816   # → Fusion Z = 276.00 mm

# Lens-holder Fusion X positions (from _ELEC_INFO in plot_field.py)
TRAP_LENS_X =  4.8   # electrode 11  (x_gem = 29.8)
COLL_LENS_X = -6.1   # electrode 12  (x_gem = 18.9)

# Region of interest around the perp-trap (Fusion world, mm)
ROI_X = (-22.0,  22.0)
ROI_Y = ( 10.0,  30.0)
ROI_Z = (258.0, 297.0)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _axes():
    x = np.arange(NX) * DX + GEM_OFF[0]
    y = np.arange(NY) * DX + GEM_OFF[1]
    z = np.arange(NZ) * DX + GEM_OFF[2]
    return x, y, z


def _fi(fusion_val, off):
    """Fusion-world coordinate → nearest grid index."""
    return int(round((fusion_val - off) / DX))


# ── PA reader (adapted from plot_field.py) ────────────────────────────────────

def read_pa(path):
    n_pts    = NX * NY * NZ
    expected = HEADER + n_pts * 8
    sz = os.path.getsize(path)
    if sz != expected:
        print(f"  WARNING: {os.path.basename(path)}: size {sz} ≠ expected {expected}")

    with open(path, "rb") as f:
        hdr = f.read(HEADER)
        raw = np.frombuffer(f.read(n_pts * 8), dtype="<f8").copy()

    nx_h = struct.unpack_from("<i", hdr, 16)[0]
    ny_h = struct.unpack_from("<i", hdr, 20)[0]
    nz_h = struct.unpack_from("<i", hdr, 24)[0]
    dx_h = struct.unpack_from("<d", hdr, 32)[0]
    print(f"  Header: {nx_h}×{ny_h}×{nz_h}  dx={dx_h:.3g} mm")

    elec_other = np.signbit(raw)
    max_val    = raw.max()
    elec_this  = raw > 0.5 * max_val
    V_norm     = np.clip(np.abs(raw) / max_val, 0.0, 1.0)

    V_norm     = V_norm    .reshape(NZ, NY, NX)
    elec_other = elec_other.reshape(NZ, NY, NX)
    elec_this  = elec_this .reshape(NZ, NY, NX)
    return V_norm, elec_other, elec_this


# ── Load both PA files and crop to ROI ────────────────────────────────────────

def load_data():
    x_f, y_f, z_f = _axes()

    ix0 = max(0,      _fi(ROI_X[0], GEM_OFF[0]))
    ix1 = min(NX - 1, _fi(ROI_X[1], GEM_OFF[0]))
    iy0 = max(0,      _fi(ROI_Y[0], GEM_OFF[1]))
    iy1 = min(NY - 1, _fi(ROI_Y[1], GEM_OFF[1]))
    iz0 = max(0,      _fi(ROI_Z[0], GEM_OFF[2]))
    iz1 = min(NZ - 1, _fi(ROI_Z[1], GEM_OFF[2]))

    xs = x_f[ix0:ix1 + 1]
    ys = y_f[iy0:iy1 + 1]
    zs = z_f[iz0:iz1 + 1]

    sub_iy = PERP_IY - iy0
    sub_iz = PERP_IZ - iz0

    if not (0 <= sub_iy < len(ys)):
        sys.exit(f"ERROR: PERP_IY={PERP_IY} falls outside ROI_Y — adjust ROI_Y.")
    if not (0 <= sub_iz < len(zs)):
        sys.exit(f"ERROR: PERP_IZ={PERP_IZ} falls outside ROI_Z — adjust ROI_Z.")

    Vs, masks = {}, {}
    for en in (11, 12):
        pa = os.path.join(BASE, f"paulTrap.pa{en}")
        if not os.path.exists(pa):
            sys.exit(f"ERROR: {pa} not found — run SIMION Refine first.")
        print(f"\nReading paulTrap.pa{en} …")
        V_full, eo_full, et_full = read_pa(pa)
        sl = np.s_[iz0:iz1 + 1, iy0:iy1 + 1, ix0:ix1 + 1]
        Vs[en]    = V_full[sl]
        masks[en] = eo_full[sl] | et_full[sl]

    elec_mask = masks[11] | masks[12]
    return xs, ys, zs, sub_iy, sub_iz, Vs[11], Vs[12], elec_mask


# ── Stability analysis ────────────────────────────────────────────────────────

def _stability(xs, V_ax):
    """
    Return (stable, x_min, v_min, trap_depth) for the on-axis 1-D potential.
    'stable' means a local minimum exists strictly between the two lens holders.
    trap_depth = V_barrier - V_min  (barrier = lower of the two lens-holder values)
    """
    # narrow window between the two lens holders
    x_lo = min(COLL_LENS_X, TRAP_LENS_X)
    x_hi = max(COLL_LENS_X, TRAP_LENS_X)
    inner = (xs > x_lo + 0.5) & (xs < x_hi - 0.5)

    if not inner.any() or np.all(np.isnan(V_ax[inner])):
        return False, float("nan"), float("nan"), 0.0

    v_inner = V_ax[inner]
    v_inner_finite = np.where(np.isnan(v_inner), np.inf, v_inner)
    mi_inner = np.argmin(v_inner_finite)
    x_min = xs[inner][mi_inner]
    v_min = v_inner[mi_inner]

    if np.isnan(v_min):
        return False, float("nan"), float("nan"), 0.0

    # potential at the lens-holder positions (nearest non-NaN point)
    def _v_at(x_target):
        i = np.argmin(np.abs(xs - x_target))
        window = V_ax[max(0, i - 3):min(len(xs), i + 4)]
        vals = window[~np.isnan(window)]
        return float(vals.mean()) if len(vals) else np.nan

    v_trap = _v_at(TRAP_LENS_X)
    v_coll = _v_at(COLL_LENS_X)

    if np.isnan(v_trap) or np.isnan(v_coll):
        return False, x_min, v_min, 0.0

    v_barrier = min(v_trap, v_coll)
    trap_depth = v_barrier - v_min

    # a true minimum requires V to drop from both sides; also check it's a minimum
    # by verifying trap_depth > 0
    stable = (trap_depth > 0)
    return stable, x_min, v_min, trap_depth


# ── Interactive matplotlib figure ─────────────────────────────────────────────

def show_interactive(xs, ys, zs, sub_iy, sub_iz, V11, V12, elec_mask,
                     v11_init, v12_init):

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(
        3, 2,
        height_ratios=[1.8, 1.8, 0.45],
        hspace=0.50, wspace=0.35,
    )
    ax_xz  = fig.add_subplot(gs[0, 0])
    ax_xy  = fig.add_subplot(gs[0, 1])
    ax_1d  = fig.add_subplot(gs[1, :])
    ax_s11 = fig.add_subplot(gs[2, 0])
    ax_s12 = fig.add_subplot(gs[2, 1])

    slider11 = Slider(ax_s11, 'V_trap_lens (11)  V',
                      -300.0, 600.0, valinit=v11_init, valstep=1.0, color='goldenrod')
    slider12 = Slider(ax_s12, 'V_coll_lens (12)  V',
                      -300.0, 600.0, valinit=v12_init, valstep=1.0, color='orchid')

    # Pre-build mesh artists for in-place update (avoids re-adding colorbars)
    XX_xz, ZZ_xz = np.meshgrid(xs, zs)          # (nz_roi, nx_roi)
    XX_xy, YY_xy = np.meshgrid(xs, ys)          # (ny_roi, nx_roi)

    mesh_xz = ax_xz.pcolormesh(XX_xz, ZZ_xz,
                                np.zeros((len(zs), len(xs))),
                                cmap='RdBu_r', shading='nearest')
    cb_xz = fig.colorbar(mesh_xz, ax=ax_xz, label='V  (V)', shrink=0.85, pad=0.02)

    ax_xz.axvline(TRAP_LENS_X, color='gold',   lw=1.3, ls='--', label='Trap lens (11)')
    ax_xz.axvline(COLL_LENS_X, color='orchid', lw=1.3, ls='--', label='Coll lens (12)')
    ax_xz.axhline(zs[sub_iz],  color='white',  lw=0.8, ls=':',  alpha=0.6)
    ax_xz.set_xlabel('X  (mm)')
    ax_xz.set_ylabel('Z  (mm)')
    ax_xz.set_title(f'X–Z plane  (Y = {ys[sub_iy]:.2f} mm)', fontsize=10)
    ax_xz.legend(fontsize=8, loc='upper right', framealpha=0.7)

    mesh_xy = ax_xy.pcolormesh(XX_xy, YY_xy,
                                np.zeros((len(ys), len(xs))),
                                cmap='RdBu_r', shading='nearest')
    cb_xy = fig.colorbar(mesh_xy, ax=ax_xy, label='V  (V)', shrink=0.85, pad=0.02)

    ax_xy.axvline(TRAP_LENS_X, color='gold',   lw=1.3, ls='--')
    ax_xy.axvline(COLL_LENS_X, color='orchid', lw=1.3, ls='--')
    ax_xy.axhline(ys[sub_iy],  color='white',  lw=0.8, ls=':',  alpha=0.6)
    ax_xy.set_xlabel('X  (mm)')
    ax_xy.set_ylabel('Y  (mm)')
    ax_xy.set_title(f'X–Y plane  (Z = {zs[sub_iz]:.1f} mm)', fontsize=10)

    ax2_1d = ax_1d.twinx()

    line_V,  = ax_1d.plot(xs, np.zeros_like(xs),
                          color='steelblue', lw=2.0, label='V  (V)')
    line_Ex, = ax2_1d.plot(xs, np.zeros_like(xs),
                           color='crimson', lw=1.4, ls='--', alpha=0.8,
                           label=r'$E_x$  (V/mm)')

    vline_min = ax_1d.axvline(np.nan, color='limegreen', lw=1.8, ls='-.', alpha=0.9,
                               zorder=5, label='Potential min')
    ax_1d.axvline(TRAP_LENS_X, color='gold',   lw=1.0, ls='--', alpha=0.7)
    ax_1d.axvline(COLL_LENS_X, color='orchid', lw=1.0, ls='--', alpha=0.7)
    ax_1d.axhline(0, color='gray', lw=0.6, alpha=0.4)
    ax2_1d.axhline(0, color='crimson', lw=0.5, alpha=0.3)

    ax_1d.set_xlabel('X  (mm, trap axis)')
    ax_1d.set_ylabel('V  (V)', color='steelblue')
    ax2_1d.set_ylabel(r'$E_x$  (V/mm)', color='crimson')
    ax_1d.tick_params(axis='y', labelcolor='steelblue')
    ax2_1d.tick_params(axis='y', labelcolor='crimson')
    ax_1d.grid(True, alpha=0.18)

    lines  = [line_V, line_Ex, vline_min]
    labels = [l.get_label() for l in lines]
    ax_1d.legend(lines, labels, fontsize=8, loc='upper left', framealpha=0.8)

    stab_text = ax_1d.text(
        0.99, 0.97, '', transform=ax_1d.transAxes,
        va='top', ha='right', fontsize=8.5, family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.88))

    def _redraw(v11, v12):
        Vc      = V11 * v11 + V12 * v12
        Vc_disp = np.where(elec_mask, np.nan, Vc)

        # colour limits: symmetric around 0
        vabs = float(np.nanmax(np.abs(Vc_disp))) or 1.0

        # X-Z heatmap
        xz_data = Vc_disp[:, sub_iy, :]            # (nz_roi, nx_roi)
        mesh_xz.set_array(xz_data.ravel())
        mesh_xz.set_clim(-vabs, vabs)

        # X-Y heatmap
        xy_data = Vc_disp[sub_iz, :, :]            # (ny_roi, nx_roi)
        mesh_xy.set_array(xy_data.ravel())
        mesh_xy.set_clim(-vabs, vabs)

        # 1-D axial profile
        V_ax  = Vc_disp[sub_iz, sub_iy, :]
        V_num = np.where(np.isnan(V_ax), np.nan, V_ax)
        Ex_ax = -np.gradient(np.where(np.isnan(V_num), 0.0, V_num), DX)
        Ex_ax = np.where(np.isnan(V_num), np.nan, Ex_ax)

        line_V.set_ydata(V_num)
        line_Ex.set_ydata(Ex_ax)
        ax_1d.relim();  ax_1d.autoscale_view()
        ax2_1d.relim(); ax2_1d.autoscale_view()

        # Stability annotation
        stable, x_min, v_min, depth = _stability(xs, V_num)
        if not np.isnan(x_min):
            vline_min.set_xdata([x_min, x_min])
        else:
            vline_min.set_xdata([np.nan, np.nan])

        if stable:
            stab_text.set_text(
                f"STABLE  ✓\n"
                f"Min at X = {x_min:.1f} mm,  V = {v_min:.1f} V\n"
                f"Trap depth = {depth:.1f} V")
            stab_text.get_bbox_patch().set_facecolor('#d4f5d4')
        elif not np.isnan(x_min):
            stab_text.set_text(
                f"UNSTABLE\n"
                f"Deepest inner point: X = {x_min:.1f} mm,  V = {v_min:.1f} V\n"
                f"No barrier (depth = {depth:.1f} V)")
            stab_text.get_bbox_patch().set_facecolor('#ffd4d4')
        else:
            stab_text.set_text("No free-space data in trap region")
            stab_text.get_bbox_patch().set_facecolor('lightyellow')

        ax_1d.set_title(
            f'Axial potential  (Y = {ys[sub_iy]:.1f} mm,  Z = {zs[sub_iz]:.1f} mm) '
            f'|  V₁₁ = {v11:.0f} V,  V₁₂ = {v12:.0f} V',
            fontsize=9)

        fig.canvas.draw_idle()

    slider11.on_changed(lambda _: _redraw(slider11.val, slider12.val))
    slider12.on_changed(lambda _: _redraw(slider11.val, slider12.val))

    _redraw(v11_init, v12_init)
    fig.suptitle(
        'Perpendicular trap – lens holder stability explorer\n'
        'Gold dashes = trap lens (11),  Orchid dashes = coll lens (12)',
        fontsize=11, fontweight='bold')
    plt.show()


# ── PyVista 3-D view ──────────────────────────────────────────────────────────

def show_3d(xs, ys, zs, V11, V12, elec_mask, v11, v12, screenshot=None):
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed — skipping 3-D view.  pip install pyvista")
        return

    Vc      = V11 * v11 + V12 * v12
    Vc_disp = np.where(elec_mask, np.nan, Vc)
    Vc_work = np.where(np.isnan(Vc_disp), 0.0, Vc_disp)

    gz, gy, gx = np.gradient(Vc_work, DX, DX, DX)
    Ex = np.where(elec_mask, 0.0, -gx)
    Ey = np.where(elec_mask, 0.0, -gy)
    Ez = np.where(elec_mask, 0.0, -gz)
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    nz_r, ny_r, nx_r = Vc_work.shape

    # ImageData: dimensions=(nx,ny,nz), data in C order (z outermost → x fastest) ✓
    grid = pv.ImageData()
    grid.dimensions = (nx_r, ny_r, nz_r)
    grid.origin     = (xs[0], ys[0], zs[0])
    grid.spacing    = (DX, DX, DX)
    grid.point_data["V"]   = Vc_work.flatten(order="C")
    grid.point_data["|E|"] = Emag.flatten(order="C")
    grid.point_data["E"]   = np.stack([Ex, Ey, Ez], axis=-1).reshape(-1, 3, order="C")

    pl = pv.Plotter(
        off_screen=(screenshot is not None),
        title=f"Perp-trap lens holders  V₁₁={v11:.0f} V  V₁₂={v12:.0f} V")
    pl.set_background("white")

    # Isosurfaces of V_total
    v_range = float(np.nanmax(np.abs(Vc_disp))) or 1.0
    levels  = np.linspace(0.05 * v_range, 0.95 * v_range, 9).tolist()
    iso = grid.contour(levels, scalars="V")
    if iso.n_points:
        pl.add_mesh(iso, scalars="V", cmap="RdBu_r",
                    clim=[-v_range, v_range], opacity=0.18,
                    smooth_shading=True,
                    scalar_bar_args=dict(
                        title=f"V (V)",
                        title_font_size=12, label_font_size=10,
                        n_labels=5, fmt="%.0f",
                        width=0.45, height=0.06,
                        position_x=0.27, position_y=0.02,
                        color="black"))

    # E-field arrows (sparse sample)
    e_thresh = grid.threshold(0.01 * Emag.max() if Emag.max() > 0 else 1e-9, scalars="|E|")
    if e_thresh.n_points > 500:
        ids = np.linspace(0, e_thresh.n_points - 1, 500, dtype=int)
        e_thresh = e_thresh.extract_points(ids, include_cells=False)
    if e_thresh.n_points:
        scale = 0.35 / max(float(Emag.max()), 1e-9)
        glyphs = e_thresh.glyph(orient="E", scale="|E|", factor=scale, geom=pv.Arrow())
        pl.add_mesh(glyphs, color="darkorange", opacity=0.8)

    # STL meshes
    stl_list = [
        ("trapping_lens_holder.stl",   "gold",      0.70),
        ("collection_lens_holder.stl", "orchid",    0.70),
        ("trap_rod_TL.stl",            "lightblue", 0.30),
        ("trap_rod_BR.stl",            "lightblue", 0.30),
        ("trap_rod_TR.stl",            "lightblue", 0.30),
        ("trap_rod_BL.stl",            "lightblue", 0.30),
        ("trapping_lens.stl",          "lightcyan", 0.25),
        ("collection_lens.stl",        "lightcyan", 0.25),
    ]
    for fname, color, opacity in stl_list:
        fpath = os.path.join(BASE, fname)
        if os.path.exists(fpath):
            pl.add_mesh(pv.read(fpath), color=color, opacity=opacity,
                        smooth_shading=True)

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
    ap.add_argument("--v11", type=float, default=50.0,
                    help="Initial trapping lens holder bias (V, electrode 11)")
    ap.add_argument("--v12", type=float, default=80.0,
                    help="Initial collection lens holder bias (V, electrode 12)")
    ap.add_argument("--3d",  dest="show3d", action="store_true",
                    help="Open PyVista 3-D window at the specified biases")
    ap.add_argument("--screenshot", default=None,
                    help="Save 3-D screenshot to PNG (implies --3d)")
    args  = ap.parse_args()
    do_3d = args.show3d or bool(args.screenshot)

    print("Loading lens PA files …")
    xs, ys, zs, sub_iy, sub_iz, V11, V12, elec_mask = load_data()
    print(f"\nROI  X ∈ [{xs[0]:.1f}, {xs[-1]:.1f}]  "
          f"Y ∈ [{ys[0]:.1f}, {ys[-1]:.1f}]  "
          f"Z ∈ [{zs[0]:.1f}, {zs[-1]:.1f}]  mm")
    print(f"Trap axis: Y = {ys[sub_iy]:.2f} mm,  Z = {zs[sub_iz]:.2f} mm")
    print(f"Lens holders: X = {TRAP_LENS_X:.1f} mm (trap, 11),  "
          f"X = {COLL_LENS_X:.1f} mm (coll, 12)")

    if do_3d:
        show_3d(xs, ys, zs, V11, V12, elec_mask,
                args.v11, args.v12, screenshot=args.screenshot)

    show_interactive(xs, ys, zs, sub_iy, sub_iz, V11, V12, elec_mask,
                     args.v11, args.v12)


if __name__ == "__main__":
    main()
