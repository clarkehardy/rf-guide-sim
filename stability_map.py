"""
stability_map.py  –  Interactive potential explorer for the optical Paul trap.

Loads SIMION unit-potential PA files for the optical-trap region:
  5 – rod_3_TL  (+RF phase / DC trim)
  6 – rod_3_TR  (−RF phase / DC trim)
  7 – rod_3_BL  (−RF phase / DC trim)
  8 – rod_3_BR  (+RF phase / DC trim)
  9 – endcap_optical_U  (axial, +z side, lower Z)
  10 – endcap_optical_D  (axial, −z side, higher Z)

Combined effective potential:
  V_eff = V_pseudo(V_RF3) + v_rod*(V5+V6+V7+V8) + v_ecU*V9 + v_ecD*V10

Pseudopotential: Φ_pseudo = (q·V_RF3²)/(4m·ω²) · |∇Φ_rf_unit|²
  where Φ_rf_unit = V5 – V6 – V7 + V8  (unit RF quadrupole pattern).
  Particle parameters are set by the PARTICLE_* constants below.

Usage:
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

# ── Grid constants (must match pa_define / locate in paulTrap.gem) ────────────
NX, NY, NZ = 131, 91, 855
DX         = 0.5
HEADER     = 56
GEM_OFF    = np.array([-25.0, -8.0, -132.0])   # Fusion = i*DX + GEM_OFF

# ── Particle parameters (match trap_config.lua / generate_voltages.py) ────────
PARTICLE_RADIUS_M    = 83e-9    # m  (166 nm diameter silica sphere)
PARTICLE_DENSITY     = 2200.0   # kg/m³  (fused silica)
PARTICLE_CHARGE      = 100      # elementary charges
RF3_FREQ_HZ          = 2e3      # Hz  (set-3 RF carrier; match f_RF3 in generate_voltages.py)

# ── Optical trap geometry (Fusion world coords, mm) ───────────────────────────
# Derived from STL bounding boxes; update if the geometry changes.
TRAP_CENTRE_X = 0.0     # mm  (trap axis sits on X = 0)
TRAP_CENTRE_Y = 19.05   # mm  (mid-point between inner faces of top/bottom set-3 rods)
ENDCAP_U_Z    = 268.06  # mm  (centre of endcap_optical_U, lower Z)
ENDCAP_D_Z    = 283.94  # mm  (centre of endcap_optical_D, higher Z)

# ── Region of interest (Fusion world, mm) ─────────────────────────────────────
ROI_X = (-15.0,  15.0)
ROI_Y = (  8.0,  30.0)
ROI_Z = (255.0, 295.0)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _axes():
    x = np.arange(NX) * DX + GEM_OFF[0]
    y = np.arange(NY) * DX + GEM_OFF[1]
    z = np.arange(NZ) * DX + GEM_OFF[2]
    return x, y, z


def _fi(fusion_val, off):
    return int(round((fusion_val - off) / DX))


# ── PA reader ─────────────────────────────────────────────────────────────────

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

    sub_iy = _fi(TRAP_CENTRE_Y, GEM_OFF[1]) - iy0
    sub_ix = _fi(TRAP_CENTRE_X, GEM_OFF[0]) - ix0

    if not (0 <= sub_iy < len(ys)):
        sys.exit(f"ERROR: TRAP_CENTRE_Y={TRAP_CENTRE_Y} outside ROI_Y — adjust ROI_Y.")
    if not (0 <= sub_ix < len(xs)):
        sys.exit(f"ERROR: TRAP_CENTRE_X={TRAP_CENTRE_X} outside ROI_X — adjust ROI_X.")

    sl = np.s_[iz0:iz1 + 1, iy0:iy1 + 1, ix0:ix1 + 1]

    Vs        = {}
    elec_mask = np.zeros((len(zs), len(ys), len(xs)), dtype=bool)

    # Optical endcaps — required.
    for en in (9, 10):
        pa = os.path.join(BASE, f"paulTrap.pa{en}")
        if not os.path.exists(pa):
            sys.exit(f"ERROR: {pa} not found — run SIMION Refine first.")
        print(f"\nReading paulTrap.pa{en} …")
        V_full, eo_full, et_full = read_pa(pa)
        Vs[en]     = V_full[sl]
        elec_mask |= eo_full[sl] | et_full[sl]

    # Set-3 rods — optional.
    for en in (5, 6, 7, 8):
        pa = os.path.join(BASE, f"paulTrap.pa{en}")
        if os.path.exists(pa):
            print(f"\nReading paulTrap.pa{en} …")
            V_full, eo_full, et_full = read_pa(pa)
            Vs[en]     = V_full[sl]
            elec_mask |= eo_full[sl] | et_full[sl]
        else:
            print(f"\n  [skip] paulTrap.pa{en} not found — rod contribution will be zero")
            Vs[en] = None

    has_rods = all(Vs.get(en) is not None for en in (5, 6, 7, 8))
    return xs, ys, zs, sub_iy, sub_ix, Vs, elec_mask, has_rods


# ── Pseudopotential ───────────────────────────────────────────────────────────

def _pseudopotential(Vs, V_RF3):
    """Return pseudopotential array [V] or None if any rod PA is missing."""
    if any(Vs.get(en) is None for en in (5, 6, 7, 8)):
        return None
    # RF unit pattern: TL(5) and BR(8) at +phase, TR(6) and BL(7) at −phase.
    Phi_rf  = Vs[5] - Vs[6] - Vs[7] + Vs[8]
    gz = np.gradient(Phi_rf, DX, axis=0)
    gy = np.gradient(Phi_rf, DX, axis=1)
    gx = np.gradient(Phi_rf, DX, axis=2)
    grad_sq = gx**2 + gy**2 + gz**2          # (V/mm)²
    e_charge = 1.602176634e-19                # C
    q        = PARTICLE_CHARGE * e_charge
    m        = (4.0 / 3.0) * np.pi * PARTICLE_RADIUS_M**3 * PARTICLE_DENSITY
    omega    = 2.0 * np.pi * RF3_FREQ_HZ
    # ×1e6 converts (V/mm)² → (V/m)² for SI consistency; result in V.
    scale = q * 1e6 / (4.0 * m * omega**2)
    return scale * V_RF3**2 * grad_sq


# ── Combined effective potential ──────────────────────────────────────────────

def _combined(Vs, elec_mask, v_ecU, v_ecD, v_rod, V_RF3):
    Vc = Vs[9] * v_ecU + Vs[10] * v_ecD
    for en in (5, 6, 7, 8):
        if Vs.get(en) is not None:
            Vc = Vc + Vs[en] * v_rod
    if V_RF3 > 0:
        pseudo = _pseudopotential(Vs, V_RF3)
        if pseudo is not None:
            Vc = Vc + pseudo
    return np.where(elec_mask, np.nan, Vc)


# ── Stability analysis ────────────────────────────────────────────────────────

def _stability(zs, V_ax):
    """Check for a potential minimum along Z between the two endcap centres."""
    z_lo = min(ENDCAP_U_Z, ENDCAP_D_Z)
    z_hi = max(ENDCAP_U_Z, ENDCAP_D_Z)
    inner = (zs > z_lo + 0.5) & (zs < z_hi - 0.5)

    if not inner.any() or np.all(np.isnan(V_ax[inner])):
        return False, float("nan"), float("nan"), 0.0

    v_inner = V_ax[inner]
    v_inf   = np.where(np.isnan(v_inner), np.inf, v_inner)
    mi      = np.argmin(v_inf)
    z_min   = zs[inner][mi]
    v_min   = v_inner[mi]

    if np.isnan(v_min):
        return False, float("nan"), float("nan"), 0.0

    def _v_at(z_target):
        i   = np.argmin(np.abs(zs - z_target))
        win = V_ax[max(0, i - 3):min(len(zs), i + 4)]
        vals = win[~np.isnan(win)]
        return float(vals.mean()) if len(vals) else np.nan

    v_at_U = _v_at(ENDCAP_U_Z)
    v_at_D = _v_at(ENDCAP_D_Z)
    if np.isnan(v_at_U) or np.isnan(v_at_D):
        return False, z_min, v_min, 0.0

    trap_depth = min(v_at_U, v_at_D) - v_min
    return (trap_depth > 0), z_min, v_min, trap_depth


# ── Interactive matplotlib figure ─────────────────────────────────────────────

def show_interactive(xs, ys, zs, sub_iy, sub_ix, Vs, elec_mask, has_rods,
                     v_ecU_init, v_ecD_init, v_rod_init, V_RF3_init):

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(
        3, 4,
        height_ratios=[1.8, 1.9, 0.45],
        hspace=0.50, wspace=0.38,
    )
    ax_zy  = fig.add_subplot(gs[0, :2])   # Z–Y side view (X fixed)
    ax_zx  = fig.add_subplot(gs[0, 2:])   # Z–X top view (Y fixed)
    ax_1d  = fig.add_subplot(gs[1, :])    # 1-D Z profile
    ax_s9  = fig.add_subplot(gs[2, 0])
    ax_s10 = fig.add_subplot(gs[2, 1])
    ax_sdc = fig.add_subplot(gs[2, 2])
    ax_srf = fig.add_subplot(gs[2, 3])

    slider_ecU = Slider(ax_s9,  'V_endcap_U (9)  V',
                        -300.0, 600.0, valinit=v_ecU_init, valstep=1.0,
                        color='teal')
    slider_ecD = Slider(ax_s10, 'V_endcap_D (10) V',
                        -300.0, 600.0, valinit=v_ecD_init, valstep=1.0,
                        color='seagreen')
    rod_label = 'V_rod_DC (5–8)  V' if has_rods else 'V_rod_DC  [not loaded]'
    slider_dc  = Slider(ax_sdc, rod_label,
                        -300.0, 300.0, valinit=v_rod_init, valstep=1.0,
                        color='cornflowerblue' if has_rods else 'lightgray')
    rf_label = 'V_RF3 amplitude  V' if has_rods else 'V_RF3  [rods not loaded]'
    slider_rf  = Slider(ax_srf, rf_label,
                        0.0, 600.0, valinit=V_RF3_init, valstep=5.0,
                        color='darkorange' if has_rods else 'lightgray')
    if not has_rods:
        slider_dc.set_active(False)
        slider_rf.set_active(False)

    # Z–Y plane at X = TRAP_CENTRE_X: horizontal = Z, vertical = Y.
    # meshgrid → shape (len(ys), len(zs)); data slice .T → same shape.
    ZZ_zy, YY_zy = np.meshgrid(zs, ys)
    mesh_zy = ax_zy.pcolormesh(ZZ_zy, YY_zy,
                                np.zeros((len(ys), len(zs))),
                                cmap='RdBu_r', shading='nearest')
    fig.colorbar(mesh_zy, ax=ax_zy, label='V  (V)', shrink=0.85, pad=0.02)
    ax_zy.axvline(ENDCAP_U_Z, color='teal',     lw=1.3, ls='--',
                  label=f'Endcap U (9)  Z={ENDCAP_U_Z:.1f} mm')
    ax_zy.axvline(ENDCAP_D_Z, color='seagreen', lw=1.3, ls='--',
                  label=f'Endcap D (10) Z={ENDCAP_D_Z:.1f} mm')
    ax_zy.axhline(ys[sub_iy], color='white',    lw=0.8, ls=':', alpha=0.6)
    ax_zy.set_xlabel('Z  (mm)');  ax_zy.set_ylabel('Y  (mm)')
    ax_zy.set_title(f'Z–Y plane  (X = {xs[sub_ix]:.2f} mm)', fontsize=10)
    ax_zy.legend(fontsize=8, loc='upper right', framealpha=0.7)

    # Z–X plane at Y = TRAP_CENTRE_Y: horizontal = Z, vertical = X.
    ZZ_zx, XX_zx = np.meshgrid(zs, xs)
    mesh_zx = ax_zx.pcolormesh(ZZ_zx, XX_zx,
                                np.zeros((len(xs), len(zs))),
                                cmap='RdBu_r', shading='nearest')
    fig.colorbar(mesh_zx, ax=ax_zx, label='V  (V)', shrink=0.85, pad=0.02)
    ax_zx.axvline(ENDCAP_U_Z, color='teal',     lw=1.3, ls='--')
    ax_zx.axvline(ENDCAP_D_Z, color='seagreen', lw=1.3, ls='--')
    ax_zx.axhline(xs[sub_ix], color='white',    lw=0.8, ls=':', alpha=0.6)
    ax_zx.set_xlabel('Z  (mm)');  ax_zx.set_ylabel('X  (mm)')
    ax_zx.set_title(f'Z–X plane  (Y = {ys[sub_iy]:.1f} mm)', fontsize=10)

    ax2_1d = ax_1d.twinx()
    line_V,    = ax_1d.plot(zs, np.zeros_like(zs),
                            color='steelblue', lw=2.2, label='V eff  (V)')
    line_V_ec, = ax_1d.plot(zs, np.zeros_like(zs),
                            color='steelblue', lw=1.0, ls=':', alpha=0.55,
                            label='V endcaps only  (V_rod=0, V_RF3=0)')
    line_Ez,   = ax2_1d.plot(zs, np.zeros_like(zs),
                             color='crimson', lw=1.4, ls='--', alpha=0.8,
                             label=r'$E_z$ total  (V/mm)')
    vline_min = ax_1d.axvline(np.nan, color='limegreen', lw=1.8, ls='-.', alpha=0.9,
                               zorder=5, label='Potential min')
    ax_1d.axvline(ENDCAP_U_Z, color='teal',     lw=1.0, ls='--', alpha=0.7)
    ax_1d.axvline(ENDCAP_D_Z, color='seagreen', lw=1.0, ls='--', alpha=0.7)
    ax_1d.axhline(0, color='gray',    lw=0.6, alpha=0.4)
    ax2_1d.axhline(0, color='crimson', lw=0.5, alpha=0.3)

    ax_1d.set_xlabel('Z  (mm, trap axis)')
    ax_1d.set_ylabel('V  (V)', color='steelblue')
    ax2_1d.set_ylabel(r'$E_z$  (V/mm)', color='crimson')
    ax_1d.tick_params(axis='y', labelcolor='steelblue')
    ax2_1d.tick_params(axis='y', labelcolor='crimson')
    ax_1d.grid(True, alpha=0.18)
    lines = [line_V, line_V_ec, line_Ez, vline_min]
    ax_1d.legend(lines, [l.get_label() for l in lines],
                 fontsize=8, loc='upper left', framealpha=0.8)

    stab_text = ax_1d.text(
        0.99, 0.97, '', transform=ax_1d.transAxes,
        va='top', ha='right', fontsize=8.5, family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.88))

    def _redraw(v_ecU, v_ecD, v_rod, V_RF3):
        Vc_total   = _combined(Vs, elec_mask, v_ecU, v_ecD, v_rod, V_RF3)
        Vc_ec_only = _combined(Vs, elec_mask, v_ecU, v_ecD, 0.0,   0.0)

        vabs = float(np.nanmax(np.abs(Vc_total))) or 1.0

        # Z–Y heatmap: data[:, :, sub_ix] is (len(zs), len(ys)); .T → (len(ys), len(zs)).
        mesh_zy.set_array(Vc_total[:, :, sub_ix].T.ravel())
        mesh_zy.set_clim(-vabs, vabs)

        # Z–X heatmap: data[:, sub_iy, :] is (len(zs), len(xs)); .T → (len(xs), len(zs)).
        mesh_zx.set_array(Vc_total[:, sub_iy, :].T.ravel())
        mesh_zx.set_clim(-vabs, vabs)

        # 1-D Z profile at (TRAP_CENTRE_X, TRAP_CENTRE_Y).
        V_ax    = Vc_total[:, sub_iy, sub_ix]
        V_ax_ec = Vc_ec_only[:, sub_iy, sub_ix]
        Ez_ax   = -np.gradient(np.where(np.isnan(V_ax), 0.0, V_ax), DX)
        Ez_ax   = np.where(np.isnan(V_ax), np.nan, Ez_ax)

        line_V.set_ydata(V_ax)
        line_V_ec.set_ydata(V_ax_ec)
        line_Ez.set_ydata(Ez_ax)
        ax_1d.relim();  ax_1d.autoscale_view()
        ax2_1d.relim(); ax2_1d.autoscale_view()

        stable, z_min, v_min, depth = _stability(zs, V_ax)
        vline_min.set_xdata([z_min, z_min] if not np.isnan(z_min)
                            else [np.nan, np.nan])

        rod_note = f"  V_rod={v_rod:.0f} V" if has_rods else "  (rods not loaded)"
        rf_note  = f",  V_RF3={V_RF3:.0f} V" if has_rods else ""
        if stable:
            stab_text.set_text(
                f"STABLE  ✓{rod_note}{rf_note}\n"
                f"Min at Z = {z_min:.1f} mm,  V = {v_min:.2f} V\n"
                f"Trap depth = {depth:.2f} V")
            stab_text.get_bbox_patch().set_facecolor('#d4f5d4')
        elif not np.isnan(z_min):
            stab_text.set_text(
                f"UNSTABLE{rod_note}{rf_note}\n"
                f"No barrier between endcaps\n"
                f"(depth = {depth:.2f} V at Z = {z_min:.1f} mm)")
            stab_text.get_bbox_patch().set_facecolor('#ffd4d4')
        else:
            stab_text.set_text("No free-space data in trap region")
            stab_text.get_bbox_patch().set_facecolor('lightyellow')

        ax_1d.set_title(
            f'Axial potential  (X = {xs[sub_ix]:.1f} mm,  Y = {ys[sub_iy]:.1f} mm)'
            f' |  V_ecU={v_ecU:.0f} V  V_ecD={v_ecD:.0f} V'
            f'  V_rod={v_rod:.0f} V  V_RF3={V_RF3:.0f} V',
            fontsize=9)
        fig.canvas.draw_idle()

    def _update(_):
        _redraw(slider_ecU.val, slider_ecD.val, slider_dc.val, slider_rf.val)

    slider_ecU.on_changed(_update)
    slider_ecD.on_changed(_update)
    slider_dc.on_changed(_update)
    slider_rf.on_changed(_update)

    _redraw(v_ecU_init, v_ecD_init, v_rod_init, V_RF3_init)
    fig.suptitle(
        'Optical Paul trap – axial stability explorer\n'
        'Dotted blue = endcaps only; solid = total (DC + pseudopotential)',
        fontsize=11, fontweight='bold')
    plt.show()


# ── PyVista 3-D view ──────────────────────────────────────────────────────────

def show_3d(xs, ys, zs, Vs, elec_mask, has_rods,
            v_ecU, v_ecD, v_rod, V_RF3, screenshot=None):
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed — skipping 3-D view.  pip install pyvista")
        return

    Vc_disp = _combined(Vs, elec_mask, v_ecU, v_ecD, v_rod, V_RF3)
    Vc_work = np.where(np.isnan(Vc_disp), 0.0, Vc_disp)

    gz, gy, gx = np.gradient(Vc_work, DX, DX, DX)
    Ex   = np.where(elec_mask, 0.0, -gx)
    Ey   = np.where(elec_mask, 0.0, -gy)
    Ez   = np.where(elec_mask, 0.0, -gz)
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    nz_r, ny_r, nx_r = Vc_work.shape
    grid = pv.ImageData()
    grid.dimensions = (nx_r, ny_r, nz_r)
    grid.origin     = (xs[0], ys[0], zs[0])
    grid.spacing    = (DX, DX, DX)
    grid.point_data["V"]   = Vc_work.flatten(order="C")
    grid.point_data["|E|"] = Emag.flatten(order="C")
    grid.point_data["E"]   = np.stack([Ex, Ey, Ez], axis=-1).reshape(-1, 3, order="C")

    title = (f"Optical trap  V_ecU={v_ecU:.0f} V  V_ecD={v_ecD:.0f} V"
             f"  V_rod={v_rod:.0f} V  V_RF3={V_RF3:.0f} V")
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
    ap.add_argument("--v-ecu",  type=float, default=20.0,
                    help="Initial endcap_optical_U voltage (V, electrode 9)")
    ap.add_argument("--v-ecd",  type=float, default=20.0,
                    help="Initial endcap_optical_D voltage (V, electrode 10)")
    ap.add_argument("--v-rod",  type=float, default=0.0,
                    help="Initial common-mode DC on set-3 rods (V, electrodes 5–8)")
    ap.add_argument("--v-rf3",  type=float, default=300.0,
                    help="Initial RF amplitude for pseudopotential (V)")
    ap.add_argument("--3d",     dest="show3d", action="store_true",
                    help="Open PyVista 3-D window at the specified biases")
    ap.add_argument("--screenshot", default=None,
                    help="Save 3-D screenshot to PNG (implies --3d)")
    args  = ap.parse_args()
    do_3d = args.show3d or bool(args.screenshot)

    print("Loading PA files …")
    xs, ys, zs, sub_iy, sub_ix, Vs, elec_mask, has_rods = load_data()

    if not has_rods:
        print("\nWARNING: one or more rod PA files (pa5–pa8) not found.\n"
              "         Rod DC and pseudopotential contributions will be omitted.")

    print(f"\nROI  X ∈ [{xs[0]:.1f}, {xs[-1]:.1f}]  "
          f"Y ∈ [{ys[0]:.1f}, {ys[-1]:.1f}]  "
          f"Z ∈ [{zs[0]:.1f}, {zs[-1]:.1f}]  mm")
    print(f"Trap-axis slice: X = {xs[sub_ix]:.2f} mm,  Y = {ys[sub_iy]:.2f} mm")
    print(f"Endcaps:  Z = {ENDCAP_U_Z:.2f} mm (U, elec 9),  "
          f"Z = {ENDCAP_D_Z:.2f} mm (D, elec 10)")

    if do_3d:
        show_3d(xs, ys, zs, Vs, elec_mask, has_rods,
                args.v_ecu, args.v_ecd, args.v_rod, args.v_rf3,
                screenshot=args.screenshot)

    show_interactive(xs, ys, zs, sub_iy, sub_ix, Vs, elec_mask, has_rods,
                     args.v_ecu, args.v_ecd, args.v_rod, args.v_rf3)


if __name__ == "__main__":
    main()
