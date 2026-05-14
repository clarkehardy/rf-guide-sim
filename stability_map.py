"""
stability_map.py  –  Interactive bias explorer.

NOTE (post-redesign): this script was written for the OLD perpendicular Paul
trap geometry (axis along X, with lens-holder DC electrodes 11, 12).  In the
new parallel-axis geometry the lens holders are dielectric (no longer biased),
and axial confinement comes from real endcaps (electrodes 9, 10).  The
electrode references below have been updated to load the new PAs, but the
1-D axial analysis (which sweeps along X) and the visualisation panes need
to be rewritten to sweep along Z and to reflect the new physics.  Use this
script as a starting point; it will produce nonsensical X-axis profiles
until the analysis layer is adapted.

Loads SIMION unit-potential PA files for:
  5, 6, 7, 8 – set-3 (optical Paul trap) rod pairs with per-rod DC trim
  9          – endcap_optical_U
  10         – endcap_optical_D

Combined potential (proposed for the rewrite):
  V_total = V_pa5*v_dcTL + V_pa6*v_dcTR + V_pa7*v_dcBL + V_pa8*v_dcBR
          + V_pa9*v_ecU  + V_pa10*v_ecD

Usage (legacy CLI, slider semantics still match OLD electrodes 11/12/9-10):
    python stability_map.py
    python stability_map.py --3d
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

# PLACEHOLDER: axis indices/coords for the new optical Paul trap region.
# Update these to match the new geometry once Fusion coords are known.
PERP_IY = 55    # legacy: → Fusion Y = 19.50 mm
PERP_IZ = 816   # legacy: → Fusion Z = 276.00 mm

# PLACEHOLDER: positions of any axial features of interest (e.g. lens centres,
# trap focus).  The old script used these as reference lines on the X profile;
# the rewrite should use Z positions for endcaps and lens focii instead.
TRAP_LENS_X =  4.8
COLL_LENS_X = -6.1

# PLACEHOLDER: region of interest around the optical Paul trap (Fusion world, mm)
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


# ── Load PA files and crop to ROI ─────────────────────────────────────────────

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
    sl = np.s_[iz0:iz1 + 1, iy0:iy1 + 1, ix0:ix1 + 1]

    # Optical endcaps (required for axial confinement).  These play the role
    # the lens holders did in the old geometry.
    for en in (9, 10):
        pa = os.path.join(BASE, f"paulTrap.pa{en}")
        if not os.path.exists(pa):
            sys.exit(f"ERROR: {pa} not found — run SIMION Refine first.")
        print(f"\nReading paulTrap.pa{en} …")
        V_full, eo_full, et_full = read_pa(pa)
        Vs[en]    = V_full[sl]
        masks[en] = eo_full[sl] | et_full[sl]

    elec_mask = masks[9] | masks[10]

    # Set-3 rods (optional; gracefully absent before first Refine).
    # Old script had 2 RF electrodes; we now have 4 independent ones.
    # Map: old en=9 → new 5 (rod_3_TL),  old en=10 → new 8 (rod_3_BR);
    # both are on the +RF phase, so a common-mode DC slider still makes sense
    # for these two as a starting point.  Extend to all four when ready.
    for new_en in (5, 8):
        pa = os.path.join(BASE, f"paulTrap.pa{new_en}")
        if os.path.exists(pa):
            print(f"\nReading paulTrap.pa{new_en} …")
            V_full, eo_full, et_full = read_pa(pa)
            Vs[new_en] = V_full[sl]
        else:
            print(f"\n  [skip] paulTrap.pa{new_en} not found — rod contribution will be zero")
            Vs[new_en] = None

    has_rods = (Vs[5] is not None and Vs[8] is not None)
    # Return signature preserved (V9_rod, V10_rod, V11_endcap, V12_endcap):
    # legacy callers still pass them as "rods" and "lens holders" but the
    # contents are now set-3 rods and optical endcaps respectively.
    return xs, ys, zs, sub_iy, sub_iz, Vs[5], Vs[8], Vs[9], Vs[10], elec_mask, has_rods


# ── Stability analysis ────────────────────────────────────────────────────────

def _stability(xs, V_ax):
    """
    Return (stable, x_min, v_min, trap_depth).
    stable = local minimum exists strictly between the two lens holders.
    trap_depth = lower barrier height − V_min.
    """
    x_lo = min(COLL_LENS_X, TRAP_LENS_X)
    x_hi = max(COLL_LENS_X, TRAP_LENS_X)
    inner = (xs > x_lo + 0.5) & (xs < x_hi - 0.5)

    if not inner.any() or np.all(np.isnan(V_ax[inner])):
        return False, float("nan"), float("nan"), 0.0

    v_inner = V_ax[inner]
    v_inf   = np.where(np.isnan(v_inner), np.inf, v_inner)
    mi      = np.argmin(v_inf)
    x_min   = xs[inner][mi]
    v_min   = v_inner[mi]

    if np.isnan(v_min):
        return False, float("nan"), float("nan"), 0.0

    def _v_at(x_target):
        i = np.argmin(np.abs(xs - x_target))
        win = V_ax[max(0, i - 3):min(len(xs), i + 4)]
        vals = win[~np.isnan(win)]
        return float(vals.mean()) if len(vals) else np.nan

    v_trap    = _v_at(TRAP_LENS_X)
    v_coll    = _v_at(COLL_LENS_X)
    if np.isnan(v_trap) or np.isnan(v_coll):
        return False, x_min, v_min, 0.0

    trap_depth = min(v_trap, v_coll) - v_min
    return (trap_depth > 0), x_min, v_min, trap_depth


# ── Interactive matplotlib figure ─────────────────────────────────────────────

def show_interactive(xs, ys, zs, sub_iy, sub_iz,
                     V9, V10, V11, V12, elec_mask, has_rods,
                     v11_init, v12_init, dc2_init):

    fig = plt.figure(figsize=(15, 9))
    # 6 virtual columns → sliders in pairs of 2
    gs = gridspec.GridSpec(
        3, 6,
        height_ratios=[1.8, 1.9, 0.45],
        hspace=0.50, wspace=0.38,
    )
    ax_xz  = fig.add_subplot(gs[0, :3])
    ax_xy  = fig.add_subplot(gs[0, 3:])
    ax_1d  = fig.add_subplot(gs[1, :])
    ax_s11 = fig.add_subplot(gs[2, :2])
    ax_s12 = fig.add_subplot(gs[2, 2:4])
    ax_sdc = fig.add_subplot(gs[2, 4:])

    slider11 = Slider(ax_s11, 'V_trap_lens (11)  V',
                      -300.0, 600.0, valinit=v11_init, valstep=1.0, color='goldenrod')
    slider12 = Slider(ax_s12, 'V_coll_lens (12)  V',
                      -300.0, 600.0, valinit=v12_init, valstep=1.0, color='orchid')
    rod_label = 'V_DC2 (rods 9+10)  V' if has_rods else 'V_DC2  [rods not loaded]'
    slider_dc = Slider(ax_sdc, rod_label,
                       0.0, 300.0, valinit=dc2_init, valstep=1.0,
                       color='cornflowerblue' if has_rods else 'lightgray')
    if not has_rods:
        slider_dc.set_active(False)

    XX_xz, ZZ_xz = np.meshgrid(xs, zs)
    XX_xy, YY_xy = np.meshgrid(xs, ys)

    mesh_xz = ax_xz.pcolormesh(XX_xz, ZZ_xz,
                                np.zeros((len(zs), len(xs))),
                                cmap='RdBu_r', shading='nearest')
    fig.colorbar(mesh_xz, ax=ax_xz, label='V  (V)', shrink=0.85, pad=0.02)
    ax_xz.axvline(TRAP_LENS_X, color='gold',   lw=1.3, ls='--', label='Trap lens (11)')
    ax_xz.axvline(COLL_LENS_X, color='orchid', lw=1.3, ls='--', label='Coll lens (12)')
    ax_xz.axhline(zs[sub_iz],  color='white',  lw=0.8, ls=':',  alpha=0.6)
    ax_xz.set_xlabel('X  (mm)');  ax_xz.set_ylabel('Z  (mm)')
    ax_xz.set_title(f'X–Z plane  (Y = {ys[sub_iy]:.2f} mm)', fontsize=10)
    ax_xz.legend(fontsize=8, loc='upper right', framealpha=0.7)

    mesh_xy = ax_xy.pcolormesh(XX_xy, YY_xy,
                                np.zeros((len(ys), len(xs))),
                                cmap='RdBu_r', shading='nearest')
    fig.colorbar(mesh_xy, ax=ax_xy, label='V  (V)', shrink=0.85, pad=0.02)
    ax_xy.axvline(TRAP_LENS_X, color='gold',   lw=1.3, ls='--')
    ax_xy.axvline(COLL_LENS_X, color='orchid', lw=1.3, ls='--')
    ax_xy.axhline(ys[sub_iy],  color='white',  lw=0.8, ls=':',  alpha=0.6)
    ax_xy.set_xlabel('X  (mm)');  ax_xy.set_ylabel('Y  (mm)')
    ax_xy.set_title(f'X–Y plane  (Z = {zs[sub_iz]:.1f} mm)', fontsize=10)

    ax2_1d = ax_1d.twinx()
    line_V,      = ax_1d.plot(xs, np.zeros_like(xs),
                              color='steelblue', lw=2.2, label='V total  (V)')
    line_V_lens, = ax_1d.plot(xs, np.zeros_like(xs),
                              color='steelblue', lw=1.0, ls=':', alpha=0.55,
                              label='V lens only (dc2=0)')
    line_Ex,     = ax2_1d.plot(xs, np.zeros_like(xs),
                               color='crimson', lw=1.4, ls='--', alpha=0.8,
                               label=r'$E_x$ total  (V/mm)')
    vline_min = ax_1d.axvline(np.nan, color='limegreen', lw=1.8, ls='-.', alpha=0.9,
                               zorder=5, label='Potential min')
    ax_1d.axvline(TRAP_LENS_X, color='gold',   lw=1.0, ls='--', alpha=0.7)
    ax_1d.axvline(COLL_LENS_X, color='orchid', lw=1.0, ls='--', alpha=0.7)
    ax_1d.axhline(0, color='gray',   lw=0.6, alpha=0.4)
    ax2_1d.axhline(0, color='crimson', lw=0.5, alpha=0.3)

    ax_1d.set_xlabel('X  (mm, trap axis)')
    ax_1d.set_ylabel('V  (V)', color='steelblue')
    ax2_1d.set_ylabel(r'$E_x$  (V/mm)', color='crimson')
    ax_1d.tick_params(axis='y', labelcolor='steelblue')
    ax2_1d.tick_params(axis='y', labelcolor='crimson')
    ax_1d.grid(True, alpha=0.18)
    lines  = [line_V, line_V_lens, line_Ex, vline_min]
    ax_1d.legend(lines, [l.get_label() for l in lines],
                 fontsize=8, loc='upper left', framealpha=0.8)

    stab_text = ax_1d.text(
        0.99, 0.97, '', transform=ax_1d.transAxes,
        va='top', ha='right', fontsize=8.5, family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.88))

    def _combined(v11, v12, dc2):
        Vc = V11 * v11 + V12 * v12
        if has_rods and dc2 != 0.0:
            Vc = Vc + V9 * dc2 + V10 * dc2
        return np.where(elec_mask, np.nan, Vc)

    def _redraw(v11, v12, dc2):
        Vc_total = _combined(v11, v12, dc2)
        Vc_lens  = np.where(elec_mask, np.nan, V11 * v11 + V12 * v12)

        vabs = float(np.nanmax(np.abs(Vc_total))) or 1.0

        mesh_xz.set_array(Vc_total[:, sub_iy, :].ravel())
        mesh_xz.set_clim(-vabs, vabs)

        mesh_xy.set_array(Vc_total[sub_iz, :, :].ravel())
        mesh_xy.set_clim(-vabs, vabs)

        V_ax       = Vc_total[sub_iz, sub_iy, :]
        V_ax_lens  = Vc_lens [sub_iz, sub_iy, :]
        Ex_ax = -np.gradient(np.where(np.isnan(V_ax), 0.0, V_ax), DX)
        Ex_ax = np.where(np.isnan(V_ax), np.nan, Ex_ax)

        line_V.set_ydata(V_ax)
        line_V_lens.set_ydata(V_ax_lens)
        line_Ex.set_ydata(Ex_ax)
        ax_1d.relim();  ax_1d.autoscale_view()
        ax2_1d.relim(); ax2_1d.autoscale_view()

        stable, x_min, v_min, depth = _stability(xs, V_ax)
        vline_min.set_xdata([x_min, x_min] if not np.isnan(x_min)
                            else [np.nan, np.nan])

        rod_note = f"  (rods: {dc2:.0f} V)" if has_rods else "  (rods not loaded)"
        if stable:
            stab_text.set_text(
                f"STABLE  ✓{rod_note}\n"
                f"Min at X = {x_min:.1f} mm,  V = {v_min:.1f} V\n"
                f"Trap depth = {depth:.1f} V")
            stab_text.get_bbox_patch().set_facecolor('#d4f5d4')
        elif not np.isnan(x_min):
            stab_text.set_text(
                f"UNSTABLE{rod_note}\n"
                f"No barrier between lens holders\n"
                f"(depth = {depth:.1f} V at X = {x_min:.1f} mm)")
            stab_text.get_bbox_patch().set_facecolor('#ffd4d4')
        else:
            stab_text.set_text("No free-space data in trap region")
            stab_text.get_bbox_patch().set_facecolor('lightyellow')

        ax_1d.set_title(
            f'Axial potential  (Y = {ys[sub_iy]:.1f} mm,  Z = {zs[sub_iz]:.1f} mm)'
            f' |  V₁₁={v11:.0f} V  V₁₂={v12:.0f} V  V_DC2={dc2:.0f} V',
            fontsize=9)
        fig.canvas.draw_idle()

    def _update(_):
        _redraw(slider11.val, slider12.val, slider_dc.val)

    slider11.on_changed(_update)
    slider12.on_changed(_update)
    slider_dc.on_changed(_update)

    _redraw(v11_init, v12_init, dc2_init)
    fig.suptitle(
        'Perpendicular trap – axial stability explorer\n'
        'Dotted blue = lens holders only (dc2=0) vs. solid = total with rod DC bias',
        fontsize=11, fontweight='bold')
    plt.show()


# ── PyVista 3-D view ──────────────────────────────────────────────────────────

def show_3d(xs, ys, zs, V9, V10, V11, V12, elec_mask, has_rods,
            v11, v12, dc2, screenshot=None):
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed — skipping 3-D view.  pip install pyvista")
        return

    Vc = V11 * v11 + V12 * v12
    if has_rods and dc2 != 0.0:
        Vc = Vc + V9 * dc2 + V10 * dc2
    Vc_disp = np.where(elec_mask, np.nan, Vc)
    Vc_work = np.where(np.isnan(Vc_disp), 0.0, Vc_disp)

    gz, gy, gx = np.gradient(Vc_work, DX, DX, DX)
    Ex = np.where(elec_mask, 0.0, -gx)
    Ey = np.where(elec_mask, 0.0, -gy)
    Ez = np.where(elec_mask, 0.0, -gz)
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    nz_r, ny_r, nx_r = Vc_work.shape
    grid = pv.ImageData()
    grid.dimensions = (nx_r, ny_r, nz_r)
    grid.origin     = (xs[0], ys[0], zs[0])
    grid.spacing    = (DX, DX, DX)
    grid.point_data["V"]   = Vc_work.flatten(order="C")
    grid.point_data["|E|"] = Emag.flatten(order="C")
    grid.point_data["E"]   = np.stack([Ex, Ey, Ez], axis=-1).reshape(-1, 3, order="C")

    title = (f"Perp-trap  V₁₁={v11:.0f} V  V₁₂={v12:.0f} V  V_DC2={dc2:.0f} V"
             if has_rods else f"Perp-trap  V₁₁={v11:.0f} V  V₁₂={v12:.0f} V")
    pl = pv.Plotter(off_screen=(screenshot is not None), title=title)
    pl.set_background("white")

    v_range = float(np.nanmax(np.abs(Vc_disp))) or 1.0
    iso = grid.contour(np.linspace(0.05 * v_range, 0.95 * v_range, 9).tolist(),
                       scalars="V")
    if iso.n_points:
        pl.add_mesh(iso, scalars="V", cmap="RdBu_r", clim=[-v_range, v_range],
                    opacity=0.18, smooth_shading=True,
                    scalar_bar_args=dict(title="V (V)", title_font_size=12,
                                        label_font_size=10, n_labels=5, fmt="%.0f",
                                        width=0.45, height=0.06,
                                        position_x=0.27, position_y=0.02,
                                        color="black"))

    e_thresh = grid.threshold(0.01 * Emag.max() if Emag.max() > 0 else 1e-9,
                              scalars="|E|")
    if e_thresh.n_points > 500:
        ids = np.linspace(0, e_thresh.n_points - 1, 500, dtype=int)
        e_thresh = e_thresh.extract_points(ids, include_cells=False)
    if e_thresh.n_points:
        glyphs = e_thresh.glyph(orient="E", scale="|E|",
                                 factor=0.35 / max(float(Emag.max()), 1e-9),
                                 geom=pv.Arrow())
        pl.add_mesh(glyphs, color="darkorange", opacity=0.8)

    for fname, color, opacity in [
        ("endcap_optical_U.stl", "gold",      0.70),
        ("endcap_optical_D.stl", "orchid",    0.70),
        ("rod_3_TL.stl",         "lightblue", 0.30),
        ("rod_3_TR.stl",         "lightblue", 0.30),
        ("rod_3_BL.stl",         "lightblue", 0.30),
        ("rod_3_BR.stl",         "lightblue", 0.30),
        ("trapping_lens.stl",    "lightcyan", 0.25),
        ("collection_lens.stl",  "lightcyan", 0.25),
        ("lens_holder.stl",      "wheat",     0.25),
    ]:
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
    ap.add_argument("--v11",  type=float, default=50.0,
                    help="Initial trapping lens holder bias (V, electrode 11)")
    ap.add_argument("--v12",  type=float, default=80.0,
                    help="Initial collection lens holder bias (V, electrode 12)")
    ap.add_argument("--dc2",  type=float, default=70.0,
                    help="Initial rod common-mode DC bias (V, electrodes 9+10; "
                         "matches V_DC2 in generate_voltages.py default)")
    ap.add_argument("--3d",   dest="show3d", action="store_true",
                    help="Open PyVista 3-D window at the specified biases")
    ap.add_argument("--screenshot", default=None,
                    help="Save 3-D screenshot to PNG (implies --3d)")
    args  = ap.parse_args()
    do_3d = args.show3d or bool(args.screenshot)

    print("Loading PA files …")
    xs, ys, zs, sub_iy, sub_iz, V9, V10, V11, V12, elec_mask, has_rods = load_data()

    if not has_rods:
        print("\nWARNING: PA9/PA10 not found — rod contribution omitted.\n"
              "         The stability assessment will match what SIMION computes\n"
              "         only when V_DC2 = 0 (rods grounded).")

    print(f"\nROI  X ∈ [{xs[0]:.1f}, {xs[-1]:.1f}]  "
          f"Y ∈ [{ys[0]:.1f}, {ys[-1]:.1f}]  "
          f"Z ∈ [{zs[0]:.1f}, {zs[-1]:.1f}]  mm")
    print(f"Trap axis: Y = {ys[sub_iy]:.2f} mm,  Z = {zs[sub_iz]:.2f} mm")
    print(f"Lens holders: X = {TRAP_LENS_X:.1f} mm (trap, 11),  "
          f"X = {COLL_LENS_X:.1f} mm (coll, 12)")
    if has_rods:
        print(f"Rod DC bias V_DC2 = {args.dc2:.0f} V  (matches simulation default)")

    if do_3d:
        show_3d(xs, ys, zs, V9, V10, V11, V12, elec_mask, has_rods,
                args.v11, args.v12, args.dc2, screenshot=args.screenshot)

    show_interactive(xs, ys, zs, sub_iy, sub_iz,
                     V9, V10, V11, V12, elec_mask, has_rods,
                     args.v11, args.v12, args.dc2)


if __name__ == "__main__":
    main()
