"""
animate.py  –  Two-panel animated view of Paul trap simulation results.

Top panel   : side view (Z = trap axis horizontal, Y = height).
              Each ion's trail grows with time; a dot marks the leading edge.
Bottom panel: DC electrode voltages vs time from the voltage schedule file,
              with a vertical cursor tracking the current animation frame.
              Omitted if no voltage file is found.

Usage
-----
    python animate.py
    python animate.py --traj trajectories_2.csv --volt voltages_2.csv
    python animate.py --speed 500     # 500 µs of sim time per wall-second
    python animate.py --fps 25
    python animate.py --save out.mp4  # requires ffmpeg  (pip install imageio[ffmpeg])
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Geometry (Fusion world coordinates, mm) ───────────────────────────────────
# Approximate bounding boxes from the sanity-check STL analysis.
# Update these if your CAD dimensions differ.
GEO = dict(
    rod_z_left   = (-116.6,  75.3),   # Z extent of left rod section
    rod_z_right  = ( 102.4, 236.0),   # Z extent of right rod section
    rod_y_top    = ( 19.66,  22.66),  # Y band of top rods (center ± 1.5 mm)
    rod_y_bot    = ( 15.47,  18.47),  # Y band of bottom rods
    gap_z        = (  75.3, 102.4),   # gate-valve gap
    endcap_z     =  -115.0,           # left end cap Z
    endcap_R_z   =  -81.2,            # right end cap Z (update to match GEM seed)
    ring_L_z     =    67.4,           # ring_L Z (from GEM seed point)
    ring_R_z     =   110.4,           # ring_R Z (from GEM seed point)
    perp_trap_z  = ( 264.0, 289.0),   # approximate Z span of perp-trap rods
    view_z       = (-131.0, 300.0),   # Z axis limits (extended to include perp-trap)
    view_y       = (  12.0,  27.5),   # Y axis limits (extended for perp-trap top rods)
)

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_trajectories(path):
    if not os.path.exists(path):
        print(f"[skip] trajectory file not found: {path}")
        return {}
    ions = {}
    with open(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            ion_id = int(parts[0])
            entry  = ions.setdefault(ion_id, {"t": [], "y": [], "z": []})
            entry["t"].append(float(parts[1]))
            entry["y"].append(float(parts[3]))   # Fusion Y
            entry["z"].append(float(parts[4]))   # Fusion Z
    return {k: {kk: np.array(vv) for kk, vv in v.items()}
            for k, v in ions.items()}


def load_voltages(path):
    if not os.path.exists(path):
        return None
    known = {"time_us", "V_endcap", "V_endcap_R", "V_ring_L", "V_ring_R",
             "V_RF", "V_RF2", "V_trap_lens", "V_coll_lens"}
    cols = {k: [] for k in known}
    headers = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if headers is None:
                try:
                    float(parts[0])   # data row — no header present
                except ValueError:
                    headers = parts   # this is the column-name header
                    continue
            if headers is None:
                continue
            for col, val in zip(headers, parts):
                if col in cols:
                    try:
                        cols[col].append(float(val))
                    except ValueError:
                        pass
    return {k: np.array(v) for k, v in cols.items()}


# ── Geometry drawing ──────────────────────────────────────────────────────────

def draw_geometry(ax):
    g = GEO
    rod_kw = dict(facecolor=(0.6, 0.6, 0.6), edgecolor="none", alpha=0.30, zorder=1)

    for z0, z1 in [g["rod_z_left"], g["rod_z_right"]]:
        for y0, y1 in [g["rod_y_top"], g["rod_y_bot"]]:
            ax.add_patch(mpatches.Rectangle(
                (z0, y0), z1 - z0, y1 - y0, **rod_kw))

    # Gap shading
    gz0, gz1 = g["gap_z"]
    ax.axvspan(gz0, gz1, color="lightyellow", alpha=0.7, zorder=0, label="Gap")

    # End cap and rings as vertical lines
    ax.axvline(g["endcap_z"],   color="teal",      lw=1.5, ls="--", alpha=0.75, label="End cap L")
    ax.axvline(g["endcap_R_z"], color="teal",      lw=1.5, ls="-.", alpha=0.75, label="End cap R")
    ax.axvline(g["ring_L_z"],   color="goldenrod", lw=1.5, ls=":",  alpha=0.85, label="Ring L")
    ax.axvline(g["ring_R_z"],   color="goldenrod", lw=1.5, ls=(0,(3,1,1,1)), alpha=0.85, label="Ring R")

    # Perp-trap region
    pz0, pz1 = g["perp_trap_z"]
    ax.axvspan(pz0, pz1, color="lavender", alpha=0.55, zorder=0, label="Perp trap")

    ax.set_xlim(g["view_z"])
    ax.set_ylim(g["view_y"])
    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Side view  (Z = trap axis,  Y = height)")
    ax.legend(loc="lower left", fontsize=7, ncol=4, framealpha=0.8)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj",  default=os.path.join(BASE, "trajectories_1.csv"))
    ap.add_argument("--volt",  default=os.path.join(BASE, "voltages_1.csv"))
    ap.add_argument("--fps",   type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=None,
                    help="µs of sim time per wall-second (default: 20 s total)")
    ap.add_argument("--save",  default=None,
                    help="Output file, e.g. animation.mp4 (requires ffmpeg)")
    args = ap.parse_args()

    ions  = load_trajectories(args.traj)
    volts = load_voltages(args.volt)

    if not ions:
        sys.exit("No trajectory data found — run a SIMION simulation first.")

    all_t  = np.concatenate([d["t"] for d in ions.values()])
    t_min, t_max = all_t.min(), all_t.max()
    duration = t_max - t_min

    speed    = args.speed if args.speed else duration / 20.0
    n_frames = max(2, int(args.fps * duration / speed))
    times    = np.linspace(t_min, t_max, n_frames)

    has_volt = volts is not None
    has_rf   = has_volt and len(volts.get("V_RF",  [])) > 0
    has_rf2  = has_volt and len(volts.get("V_RF2", [])) > 0
    if not has_volt:
        print("  No voltage file found — bottom panel omitted.")

    # ── Layout ────────────────────────────────────────────────────────────────
    if has_volt:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(13, 7),
            gridspec_kw={"height_ratios": [2, 1]})
        fig.subplots_adjust(hspace=0.38)
    else:
        fig, ax_top = plt.subplots(1, 1, figsize=(13, 5))
        ax_bot = None

    # ── Top panel ─────────────────────────────────────────────────────────────
    draw_geometry(ax_top)

    n_ions = len(ions)
    if n_ions <= 10:
        cmap = plt.cm.tab10
    else:
        cmap = plt.cm.viridis
    ion_ids = sorted(ions.keys())
    colors  = {iid: cmap(i / max(1, n_ions - 1)) for i, iid in enumerate(ion_ids)}

    trails = {}
    dots   = {}
    for iid in ion_ids:
        c = colors[iid]
        trail, = ax_top.plot([], [], lw=1.3, color=c, alpha=0.85, zorder=3)
        dot,   = ax_top.plot([], [], "o", ms=5, color=c, zorder=4)
        trails[iid] = trail
        dots[iid]   = dot

    time_label = ax_top.text(
        0.995, 0.97, "", transform=ax_top.transAxes,
        va="top", ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75))

    # Precompute the first time each ion crosses Z = 200 mm (Fusion world).
    # crossing_t[iid] = time of first crossing, or inf if never reached.
    GAP_THRESHOLD_Z = 200.0
    ax_top.axvline(GAP_THRESHOLD_Z, color="gray", lw=0.8, ls="--", alpha=0.5)

    crossing_t = {}
    for iid, data in ions.items():
        crossed = np.where(data["z"] >= GAP_THRESHOLD_Z)[0]
        crossing_t[iid] = data["t"][crossed[0]] if len(crossed) else np.inf

    n_total = len(ions)
    gap_label = ax_top.text(
        0.005, 0.97, "", transform=ax_top.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75))

    # ── Voltage panel (DC electrodes + RF amplitude) ──────────────────────────
    vcursor = None
    if has_volt:
        vt = volts["time_us"]
        volt_style = {
            "V_endcap":    ("teal",       "-",                "End cap L (3)  [DC]"),
            "V_endcap_R":  ("teal",       "--",               "End cap R (8)  [DC]"),
            "V_ring_L":    ("goldenrod",  (0,(5,2)),          "Ring L (6)  [DC]"),
            "V_ring_R":    ("goldenrod",  (0,(5,2,1,2)),      "Ring R (7)  [DC]"),
            "V_trap_lens": ("purple",     (0,(1,1)),          "Trap lens (11) [DC]"),
            "V_coll_lens": ("orchid",     (0,(1,1,3,1)),      "Coll lens (12) [DC]"),
        }
        for key, (color, ls, label) in volt_style.items():
            if key in volts and len(volts[key]):
                ax_bot.step(vt, volts[key], where="post",
                            color=color, ls=ls, lw=1.5, label=label)
        if has_rf:
            ax_bot.step(vt, volts["V_RF"], where="post",
                        color="crimson", lw=1.5, ls=(0, (3, 1, 1, 1)),
                        label="RF amplitude V₀")
        if has_rf2:
            ax_bot.step(vt, volts["V_RF2"], where="post",
                        color="darkorange", lw=1.5, ls=(0, (3, 1, 1, 1)),
                        label="RF2 amplitude V₀")

        vcursor = ax_bot.axvline(t_min, color="red", lw=1.0, ls="--", alpha=0.8, zorder=5)
        ax_bot.set_xlim(t_min, t_max)
        ax_bot.set_xlabel("Time (µs)")
        ax_bot.set_ylabel("Voltage (V)")
        ax_bot.set_title("Electrode voltages")
        ax_bot.legend(loc="upper right", fontsize=8)
        ax_bot.grid(True, alpha=0.25)

    # ── Animation ─────────────────────────────────────────────────────────────
    def update(frame_idx):
        t = times[frame_idx]
        time_label.set_text(f"t = {t:,.0f} µs")

        for iid, data in ions.items():
            mask = data["t"] <= t
            zz = data["z"][mask]
            yy = data["y"][mask]
            trails[iid].set_data(zz, yy)
            if mask.any():
                dots[iid].set_data([zz[-1]], [yy[-1]])
            else:
                dots[iid].set_data([], [])

        n_crossed = sum(1 for ct in crossing_t.values() if ct <= t)
        gap_label.set_text(f"Fraction crossed:  {n_crossed}/{n_total}")

        if vcursor is not None:
            vcursor.set_xdata([t, t])

    interval_ms = 1000.0 / args.fps
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval_ms, repeat=True)

    if args.save:
        print(f"Saving {n_frames} frames to {args.save} …")
        ani.save(args.save, fps=args.fps,
                 writer=animation.FFMpegWriter(fps=args.fps, bitrate=2000))
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
