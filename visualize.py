"""
visualize.py
Load paulTrap STL geometry and SIMION trajectories, render with PyVista.

Trajectories are colored by time-of-flight so you can read pulse timing
directly off the plot: the colorbar shows µs, and you can see exactly when
a particle enters/leaves each region.

Run from the project directory:
    python visualize.py
    python visualize.py --traj trajectories_2.csv
    python visualize.py --screenshot out.png
    python visualize.py --cmap viridis         # alternative colormap
"""

import argparse
import os
import struct
import sys
import numpy as np

try:
    import pyvista as pv
except ImportError:
    sys.exit("PyVista not found.  Install with:  pip install pyvista")

BASE = os.path.dirname(os.path.abspath(__file__))


# ── STL helpers (used for camera-path auto-detection) ─────────────────────────

def _stl_bbox(path):
    """Min/max corners of a binary STL, or None if absent/empty."""
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


def _bbox_union(*boxes):
    bs = [b for b in boxes if b is not None]
    if not bs:
        return None
    lo = np.minimum.reduce([b[0] for b in bs])
    hi = np.maximum.reduce([b[1] for b in bs])
    return lo, hi


def compute_camera_targets():
    """Find the loading-trap centre, optical-trap centre, and overall extent
    of the apparatus from STL bounding boxes.  Returns (load_centre,
    opt_centre, all_bbox) or None if the required STLs are missing."""
    rods_1 = _bbox_union(*[_stl_bbox(os.path.join(BASE, f"rod_1_{s}.stl"))
                           for s in ("TL", "TR", "BL", "BR")])
    rods_3 = _bbox_union(*[_stl_bbox(os.path.join(BASE, f"rod_3_{s}.stl"))
                           for s in ("TL", "TR", "BL", "BR")])
    if rods_1 is None or rods_3 is None:
        return None
    load_centre = tuple(0.5 * (rods_1[0] + rods_1[1]))
    opt_centre  = tuple(0.5 * (rods_3[0] + rods_3[1]))
    all_bb = _bbox_union(rods_1, rods_3,
        _bbox_union(*[_stl_bbox(os.path.join(BASE, f"rod_2_{s}.stl"))
                      for s in ("TL", "TR", "BL", "BR")]),
        _bbox_union(*[_stl_bbox(os.path.join(BASE, f"endcap_{r}.stl"))
                      for r in ("load_U", "load_D", "optical_U", "optical_D")]),
    )
    return load_centre, opt_centre, all_bb

# ── Electrode geometry ────────────────────────────────────────────────────────
# (filename, RGB color, opacity, label)
# +RF = warm red, −RF = steel blue, endcaps = teal/seagreen, dielectrics = pale cyan
ELECTRODES = [
    # Set 1 (loading Paul trap)
    ("rod_1_TL.stl",         (0.85, 0.20, 0.15), 0.40, "Sets 1+2 (+RF)"),
    ("rod_1_BR.stl",         (0.85, 0.20, 0.15), 0.40, None),
    ("rod_1_TR.stl",         (0.20, 0.45, 0.80), 0.40, "Sets 1+2 (−RF)"),
    ("rod_1_BL.stl",         (0.20, 0.45, 0.80), 0.40, None),
    # Set 2 (RF guide, after gate valve)
    ("rod_2_TL.stl",         (0.85, 0.20, 0.15), 0.40, None),
    ("rod_2_BR.stl",         (0.85, 0.20, 0.15), 0.40, None),
    ("rod_2_TR.stl",         (0.20, 0.45, 0.80), 0.40, None),
    ("rod_2_BL.stl",         (0.20, 0.45, 0.80), 0.40, None),
    # Set 3 (optical Paul trap, all 4 independent)
    ("rod_3_TL.stl",         (1.00, 0.45, 0.20), 0.40, "Set 3 TL"),
    ("rod_3_TR.stl",         (1.00, 0.65, 0.20), 0.40, "Set 3 TR"),
    ("rod_3_BL.stl",         (0.35, 0.55, 0.85), 0.40, "Set 3 BL"),
    ("rod_3_BR.stl",         (0.55, 0.65, 0.85), 0.40, "Set 3 BR"),
    # Endcaps (4)
    ("endcap_load_U.stl",    (0.15, 0.70, 0.55), 0.55, "Load endcap U (3)"),
    ("endcap_load_D.stl",    (0.15, 0.70, 0.55), 0.55, "Load endcap D (4)"),
    ("endcap_optical_U.stl", (0.10, 0.55, 0.30), 0.55, "Optical endcap U (9)"),
    ("endcap_optical_D.stl", (0.10, 0.55, 0.30), 0.55, "Optical endcap D (10)"),
    # Dielectric volumes
    ("trapping_lens.stl",    (0.50, 0.90, 0.95), 0.30, "Trap lens (dielectric)"),
    ("collection_lens.stl",  (0.50, 0.90, 0.95), 0.30, "Coll lens (dielectric)"),
    ("lens_holder.stl",      (0.85, 0.80, 0.65), 0.30, "Lens holder (dielectric)"),
]


def load_electrodes(plotter):
    for fname, rgb, opacity, label in ELECTRODES:
        path = os.path.join(BASE, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not found")
            continue
        plotter.add_mesh(
            pv.read(path),
            color=rgb,
            opacity=opacity,
            smooth_shading=True,
            label=label,
        )


# ── Trajectory loading ────────────────────────────────────────────────────────

def load_trajectories(path):
    """Return dict {ion_id: {"pts": Nx3, "times": N}} in Fusion world mm / µs."""
    if not os.path.exists(path):
        print(f"  [skip] trajectory file not found: {path}")
        return {}
    ions = {}
    with open(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            ion_id = int(parts[0])
            t = float(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            entry = ions.setdefault(ion_id, {"pts": [], "times": []})
            entry["pts"].append((x, y, z))
            entry["times"].append(t)
    return {k: {"pts": np.array(v["pts"]), "times": np.array(v["times"])}
            for k, v in ions.items()}


def add_trajectories(plotter, ions, cmap="plasma"):
    if not ions:
        print("  No trajectory data.")
        return

    # Global time range for a shared colorbar across all ions
    t_min = min(d["times"][0]  for d in ions.values())
    t_max = max(d["times"][-1] for d in ions.values())
    clim  = [t_min, t_max]

    scalar_bar_added = False
    for ion_id, data in sorted(ions.items()):
        pts   = data["pts"]
        times = data["times"]
        if len(pts) < 2:
            continue

        line = pv.lines_from_points(pts)
        line["Time (µs)"] = times

        plotter.add_mesh(
            line,
            scalars="Time (µs)",
            cmap=cmap,
            clim=clim,
            line_width=2.0,
            show_scalar_bar=not scalar_bar_added,
            scalar_bar_args=dict(
                title="Time (µs)",
                title_font_size=14,
                label_font_size=12,
                width=0.55,
                height=0.06,
                position_x=0.22,
                position_y=0.02,
                n_labels=5,
                fmt="%.0f",
                color="black",
            ),
        )
        scalar_bar_added = True

    total_pts = sum(len(d["pts"]) for d in ions.values())
    print(f"  {len(ions)} ion(s), {total_pts} points, "
          f"t = {t_min:.0f} – {t_max:.0f} µs")


# ── Camera flythrough ────────────────────────────────────────────────────────

def _smoothstep(s):
    s = max(0.0, min(1.0, float(s)))
    return s * s * (3.0 - 2.0 * s)


def render_flythrough(plotter, output_path, fps=30, duration=10.0,
                     orbit_radius=50.0, orbit_height=100.0,
                     hold_frac=0.10, fly_frac=0.40):
    """Save a cinematic camera flythrough:
       Phase 1 (hold_frac):       wide hold at the loading-trap end
       Phase 2 (fly_frac):        fly along the RF guide while zooming in
       Phase 3 (1 - hold - fly):  one full orbit around the optical trap
                                   from above

    All Fusion-world coordinates are auto-detected from the STL bounding
    boxes of the rod and endcap meshes.

    Output format: .mp4 / .mov / .avi → ffmpeg writer; .gif → imageio writer.
    The plotter must have been constructed with off_screen=True.
    """
    targets = compute_camera_targets()
    if targets is None:
        sys.exit("ERROR: cannot find rod_1_*.stl or rod_3_*.stl to compute "
                 "the camera path.  Pass --no-animation or add the STLs.")
    load_centre, opt_centre, _ = targets
    x_axis = 0.5 * (load_centre[0] + opt_centre[0])
    y_axis = 0.5 * (load_centre[1] + opt_centre[1])
    z_load = load_centre[2]
    z_opt  = opt_centre[2]
    length = z_opt - z_load

    print(f"Camera targets:")
    print(f"  loading centre:  ({load_centre[0]:.2f}, {load_centre[1]:.2f}, {load_centre[2]:.2f})")
    print(f"  optical centre:  ({opt_centre[0]:.2f}, {opt_centre[1]:.2f}, {opt_centre[2]:.2f})")
    print(f"  guide length:    {length:.1f} mm")

    # Phase 1: head-on wide view from behind the loading trap, slightly elevated.
    # Looking down the trap axis toward the optical end.
    P1_pos   = np.array([x_axis,
                         y_axis + 70.0,
                         z_load  - 220.0])
    P1_focal = np.array([x_axis,
                         y_axis,
                         z_load + 0.5 * length])

    # Phase 3 start: positioned above the optical trap at θ = 0, ready to orbit.
    P3_pos_start = np.array([x_axis + orbit_radius,
                             y_axis + orbit_height,
                             z_opt])
    P3_focal     = np.array([x_axis, y_axis, z_opt])

    view_up = [0.0, 1.0, 0.0]

    # Output writer.  Higher movie quality (default 5) is barely-acceptable at
    # HD; 7 gives clean trajectory lines without ballooning the file size.
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".mp4", ".mov", ".avi"):
        plotter.open_movie(output_path, framerate=fps, quality=7)
    elif ext == ".gif":
        plotter.open_gif(output_path, fps=fps)
    else:
        sys.exit(f"ERROR: unknown animation extension {ext!r}. Use .mp4 or .gif.")

    n_frames = max(2, int(round(fps * duration)))
    print(f"Rendering {n_frames} frames at {fps} fps ({duration:.1f} s) "
          f"→ {os.path.basename(output_path)}")

    for frame in range(n_frames):
        t = frame / (n_frames - 1)         # 0 → 1 over the whole movie

        if t < hold_frac:
            pos, focal = P1_pos, P1_focal
        elif t < hold_frac + fly_frac:
            # Phase 2: smooth interpolation P1 → P3-start
            s  = (t - hold_frac) / fly_frac
            ss = _smoothstep(s)
            pos   = P1_pos   * (1.0 - ss) + P3_pos_start * ss
            focal = P1_focal * (1.0 - ss) + P3_focal     * ss
        else:
            # Phase 3: one revolution about the optical trap, viewed from above
            s     = (t - hold_frac - fly_frac) / (1.0 - hold_frac - fly_frac)
            theta = 2.0 * np.pi * s
            pos = np.array([x_axis + orbit_radius * np.cos(theta),
                            y_axis + orbit_height,
                            z_opt  + orbit_radius * np.sin(theta)])
            focal = P3_focal

        plotter.camera_position = [list(pos), list(focal), view_up]
        plotter.write_frame()

        if (frame + 1) % max(1, fps) == 0:
            print(f"  frame {frame + 1}/{n_frames}")

    plotter.close()
    print(f"Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=os.path.join(BASE, "trajectories_1.csv"),
                    help="Path to trajectories CSV")
    ap.add_argument("--screenshot", default=None,
                    help="Save still PNG instead of opening interactive window")
    ap.add_argument("--animation", default=None,
                    help="Save a cinematic camera flythrough to FILE.  Use "
                         "a .mp4 (needs imageio-ffmpeg) or .gif extension.  "
                         "Hold at loading end → fly along guide zooming in "
                         "→ orbit optical trap from above.")
    ap.add_argument("--duration", type=float, default=10.0,
                    help="Total animation duration in seconds (default: 10)")
    ap.add_argument("--fps", type=int, default=30,
                    help="Frames per second for the animation (default: 30)")
    ap.add_argument("--orbit-radius", type=float, default=50.0,
                    help="Orbit radius around the optical trap in mm (default: 50)")
    ap.add_argument("--orbit-height", type=float, default=100.0,
                    help="Orbit height above the trap axis in mm (default: 100)")
    ap.add_argument("--resolution", type=int, nargs=2, default=None,
                    metavar=("WIDTH", "HEIGHT"),
                    help="Render resolution in pixels.  Default: 1920 1080 "
                         "for --animation, otherwise PyVista's default "
                         "(1024×768).  Use e.g. 2560 1440 for QHD or "
                         "3840 2160 for 4K.")
    ap.add_argument("--cmap", default="plasma",
                    help="Matplotlib colormap name for time coloring (default: plasma)")
    args = ap.parse_args()

    off_screen = bool(args.screenshot) or bool(args.animation)
    plotter = pv.Plotter(off_screen=off_screen, title="Paul Trap — RF Guide")
    plotter.set_background("white")

    # Window / render resolution.  Explicit --resolution wins; otherwise the
    # animation mode defaults to HD and other modes keep PyVista's default.
    if args.resolution is not None:
        plotter.window_size = list(args.resolution)
    elif args.animation:
        plotter.window_size = [1920, 1080]

    print("Loading electrodes…")
    load_electrodes(plotter)

    print("Loading trajectories…")
    ions = load_trajectories(args.traj)
    add_trajectories(plotter, ions, cmap=args.cmap)

    plotter.add_axes(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    # Legend would obscure the orbit view, so skip it in animation mode.
    if not args.animation and any(label for *_, label in ELECTRODES):
        plotter.add_legend(bcolor="white", border=True, size=(0.16, 0.18))

    plotter.camera_position = "iso"

    if args.animation:
        render_flythrough(plotter, args.animation,
                          fps=args.fps, duration=args.duration,
                          orbit_radius=args.orbit_radius,
                          orbit_height=args.orbit_height)
    elif args.screenshot:
        plotter.show(screenshot=args.screenshot, auto_close=True)
        print(f"Saved: {args.screenshot}")
    else:
        plotter.show()


if __name__ == "__main__":
    main()
