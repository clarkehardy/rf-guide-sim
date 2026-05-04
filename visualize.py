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
import sys
import numpy as np

try:
    import pyvista as pv
except ImportError:
    sys.exit("PyVista not found.  Install with:  pip install pyvista")

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Electrode geometry ────────────────────────────────────────────────────────
# (filename, RGB color, opacity, label)
# +RF = warm red, -RF = steel blue, end cap = teal, rings = amber
ELECTRODES = [
    ("rod_P1_L1.stl",  (0.85, 0.20, 0.15), 0.40, "Pair 1 (+RF)"),
    ("rod_P1_L2.stl",  (0.85, 0.20, 0.15), 0.40, None),
    ("rod_P2_L1.stl",  (0.20, 0.45, 0.80), 0.40, "Pair 2 (−RF)"),
    ("rod_P2_L2.stl",  (0.20, 0.45, 0.80), 0.40, None),
    ("endcap_L.stl",   (0.15, 0.70, 0.55), 0.55, "End cap L"),
    ("endcap_R.stl",   (0.15, 0.70, 0.55), 0.55, "End cap R"),
    ("rod_P1_R1.stl",  (0.85, 0.20, 0.15), 0.40, None),
    ("rod_P1_R2.stl",  (0.85, 0.20, 0.15), 0.40, None),
    ("rod_P2_R1.stl",  (0.20, 0.45, 0.80), 0.40, None),
    ("rod_P2_R2.stl",  (0.20, 0.45, 0.80), 0.40, None),
    ("ring_L.stl",     (0.90, 0.65, 0.10), 0.60, "Ring L"),
    ("ring_R.stl",     (0.90, 0.65, 0.10), 0.60, "Ring R"),
    # Perpendicular Paul trap (axis along X)
    ("trap_rod_TL.stl",            (0.85, 0.20, 0.15), 0.40, "Perp +RF2"),
    ("trap_rod_BR.stl",            (0.85, 0.20, 0.15), 0.40, None),
    ("trap_rod_TR.stl",            (0.20, 0.45, 0.80), 0.40, "Perp −RF2"),
    ("trap_rod_BL.stl",            (0.20, 0.45, 0.80), 0.40, None),
    ("trapping_lens_holder.stl",   (0.60, 0.20, 0.70), 0.55, "Trap lens holder"),
    ("collection_lens_holder.stl", (0.60, 0.20, 0.70), 0.55, "Coll lens holder"),
    ("trapping_lens.stl",          (0.50, 0.90, 0.95), 0.30, "Trap lens (glass)"),
    ("collection_lens.stl",        (0.50, 0.90, 0.95), 0.30, None),
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=os.path.join(BASE, "trajectories_1.csv"),
                    help="Path to trajectories CSV")
    ap.add_argument("--screenshot", default=None,
                    help="Save PNG instead of opening interactive window")
    ap.add_argument("--cmap", default="plasma",
                    help="Matplotlib colormap name for time coloring (default: plasma)")
    args = ap.parse_args()

    off_screen = args.screenshot is not None
    plotter = pv.Plotter(off_screen=off_screen, title="Paul Trap — RF Guide")
    plotter.set_background("white")

    print("Loading electrodes…")
    load_electrodes(plotter)

    print("Loading trajectories…")
    ions = load_trajectories(args.traj)
    add_trajectories(plotter, ions, cmap=args.cmap)

    plotter.add_axes(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    if any(label for *_, label in ELECTRODES):
        plotter.add_legend(bcolor="white", border=True, size=(0.16, 0.18))

    plotter.camera_position = "iso"

    if off_screen:
        plotter.show(screenshot=args.screenshot, auto_close=True)
        print(f"Saved: {args.screenshot}")
    else:
        plotter.show()


if __name__ == "__main__":
    main()
