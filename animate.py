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
import re
import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Geometry derived from STL files ───────────────────────────────────────────
# All Z spans, Y bands, endcap centres, and view limits are computed at startup
# from the bounding boxes of the STL files in this directory.  Missing files
# are silently skipped — their corresponding GEO entries become None and the
# affected geometry element is just not drawn.

def _read_stl_bbox(path):
    """Bounding box of a binary STL.  Returns ((xmin,xmax),(ymin,ymax),(zmin,zmax))
    or None if the file is absent."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        raw = f.read(n * 50)
    if n == 0 or len(raw) < n * 50:
        return None
    # Each triangle: 12 bytes normal + 36 bytes (3 vertices × 3 float32) + 2 bytes attr
    arr   = np.frombuffer(raw, dtype=np.uint8).reshape(n, 50)
    verts = np.frombuffer(arr[:, 12:48].tobytes(), dtype="<f4").reshape(-1, 3)
    return ((float(verts[:, 0].min()), float(verts[:, 0].max())),
            (float(verts[:, 1].min()), float(verts[:, 1].max())),
            (float(verts[:, 2].min()), float(verts[:, 2].max())))


def _bbox_union(*boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return ((min(b[0][0] for b in boxes), max(b[0][1] for b in boxes)),
            (min(b[1][0] for b in boxes), max(b[1][1] for b in boxes)),
            (min(b[2][0] for b in boxes), max(b[2][1] for b in boxes)))


def compute_geo(base=BASE):
    bb   = lambda name: _read_stl_bbox(os.path.join(base, name))
    cz   = lambda b: None if b is None else 0.5 * (b[2][0] + b[2][1])
    zsp  = lambda b: None if b is None else b[2]
    ysp  = lambda b: None if b is None else b[1]

    set1 = _bbox_union(*[bb(f"rod_1_{s}.stl") for s in ("TL", "TR", "BL", "BR")])
    set2 = _bbox_union(*[bb(f"rod_2_{s}.stl") for s in ("TL", "TR", "BL", "BR")])
    set3 = _bbox_union(*[bb(f"rod_3_{s}.stl") for s in ("TL", "TR", "BL", "BR")])

    rod12_top = _bbox_union(*[bb(f"rod_{i}_{s}.stl") for i in (1, 2) for s in ("TL", "TR")])
    rod12_bot = _bbox_union(*[bb(f"rod_{i}_{s}.stl") for i in (1, 2) for s in ("BL", "BR")])
    rod3_top  = _bbox_union(*[bb(f"rod_3_{s}.stl") for s in ("TL", "TR")])
    rod3_bot  = _bbox_union(*[bb(f"rod_3_{s}.stl") for s in ("BL", "BR")])

    ec_loadU = bb("endcap_load_U.stl")
    ec_loadD = bb("endcap_load_D.stl")
    ec_optU  = bb("endcap_optical_U.stl")
    ec_optD  = bb("endcap_optical_D.stl")

    # Gate-valve gap: the empty Z interval between set 1 and set 2 (whichever
    # ordering is correct in the new geometry).
    gap_z = None
    if set1 and set2:
        z1_lo, z1_hi = set1[2]
        z2_lo, z2_hi = set2[2]
        if z1_hi < z2_lo:
            gap_z = (z1_hi, z2_lo)
        elif z2_hi < z1_lo:
            gap_z = (z2_hi, z1_lo)

    # View bounds: union of every loaded body, with small padding.
    all_bb = _bbox_union(set1, set2, set3, ec_loadU, ec_loadD, ec_optU, ec_optD)
    view_z = view_y = None
    if all_bb:
        view_z = (all_bb[2][0] - 10.0, all_bb[2][1] + 10.0)
        view_y = (all_bb[1][0] -  2.0, all_bb[1][1] +  2.0)

    return dict(
        rod_z_set1         = zsp(set1),
        rod_z_set2         = zsp(set2),
        rod_z_set3         = zsp(set3),
        rod_y_top          = ysp(rod12_top),
        rod_y_bot          = ysp(rod12_bot),
        rod_y_top_3        = ysp(rod3_top),
        rod_y_bot_3        = ysp(rod3_bot),
        gap_z              = gap_z,
        endcap_load_U_z    = cz(ec_loadU),
        endcap_load_D_z    = cz(ec_loadD),
        endcap_optical_U_z = cz(ec_optU),
        endcap_optical_D_z = cz(ec_optD),
        view_z             = view_z,
        view_y             = view_y,
    )


GEO = compute_geo()

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
    known = {"time_us", "V_RF", "V_RF3",
             "V_endcap_load_U", "V_endcap_load_D",
             "V_dc_3_TL", "V_dc_3_TR", "V_dc_3_BL", "V_dc_3_BR",
             "V_endcap_optical_U", "V_endcap_optical_D"}
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


def load_n_particles(config_path):
    """Read particles.n from trap_config.lua; returns None if not found."""
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        text = f.read()
    text = re.sub(r'--[^\n]*', '', text)
    m = re.search(r'\bparticles\b.*?\bn\s*=\s*(\d+)', text, re.DOTALL)
    return int(m.group(1)) if m else None


def load_triggers(config_path):
    """Parse active trigger entries from trap_config.lua."""
    if not os.path.exists(config_path):
        return []
    with open(config_path) as f:
        text = f.read()
    text = re.sub(r'--[^\n]*', '', text)   # strip Lua line comments
    triggers = []
    for m in re.finditer(
        r'\{\s*z_mm\s*=\s*([\d.e+\-]+)\s*,\s*electrodes\s*=\s*\{([^}]*)\}',
        text
    ):
        triggers.append({
            'z_mm':       float(m.group(1)),
            'electrodes': [int(x) for x in re.findall(r'\d+', m.group(2))],
        })
    return triggers


def compute_fire_times(ions, triggers):
    """For each trigger, return the TOF when the first ion crosses z_mm."""
    times = []
    for trig in triggers:
        t_fire = np.inf
        for data in ions.values():
            idx = np.where(data['z'] >= trig['z_mm'])[0]
            if len(idx):
                t_fire = min(t_fire, data['t'][idx[0]])
        times.append(t_fire if np.isfinite(t_fire) else None)
    return times


# ── Geometry drawing ──────────────────────────────────────────────────────────

def draw_geometry(ax, triggers=(), trig_colors=()):
    g = GEO
    rod_kw = dict(facecolor=(0.6, 0.6, 0.6), edgecolor="none", alpha=0.30, zorder=1)

    # Sets 1 + 2 share the same (narrower) rod spacing.
    for zspan, ysp in [
        (g["rod_z_set1"], g["rod_y_top"]),
        (g["rod_z_set1"], g["rod_y_bot"]),
        (g["rod_z_set2"], g["rod_y_top"]),
        (g["rod_z_set2"], g["rod_y_bot"]),
    ]:
        if zspan is None or ysp is None:
            continue
        z0, z1 = zspan
        y0, y1 = ysp
        ax.add_patch(mpatches.Rectangle((z0, y0), z1 - z0, y1 - y0, **rod_kw))

    # Set 3 (optical Paul trap) uses wider rod spacing.
    if g["rod_z_set3"] is not None:
        z0, z1 = g["rod_z_set3"]
        for ysp in (g["rod_y_top_3"], g["rod_y_bot_3"]):
            if ysp is None:
                continue
            y0, y1 = ysp
            ax.add_patch(mpatches.Rectangle(
                (z0, y0), z1 - z0, y1 - y0,
                facecolor=(0.55, 0.45, 0.7), edgecolor="none", alpha=0.30, zorder=1))

    # Gate-valve gap.
    if g["gap_z"] is not None:
        gz0, gz1 = g["gap_z"]
        ax.axvspan(gz0, gz1, color="lightyellow", alpha=0.7, zorder=0, label="Gate valve gap")

    # Endcaps as vertical lines.
    endcap_lines = [
        ("endcap_load_U_z",    "teal",     "--", "Load endcap U (3)"),
        ("endcap_load_D_z",    "teal",     "-.", "Load endcap D (4)"),
        ("endcap_optical_U_z", "seagreen", "--", "Optical endcap U (9)"),
        ("endcap_optical_D_z", "seagreen", "-.", "Optical endcap D (10)"),
    ]
    for key, color, ls, label in endcap_lines:
        z = g[key]
        if z is None:
            continue
        ax.axvline(z, color=color, lw=1.5, ls=ls, alpha=0.8, label=label)

    for i, trig in enumerate(triggers):
        c = trig_colors[i] if i < len(trig_colors) else "darkorchid"
        elec_str = ", ".join(str(e) for e in trig["electrodes"])
        ax.axvline(trig["z_mm"], color=c, lw=1.5, ls=(0, (4, 2)), alpha=0.85,
                   label=f"Trigger {i+1}: Z={trig['z_mm']:.0f} mm → {{{elec_str}}}")

    if g["view_z"] is not None:
        ax.set_xlim(g["view_z"])
    if g["view_y"] is not None:
        ax.set_ylim(g["view_y"])
    ax.set_xlabel("Z (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Side view  (Z = trap axis,  Y = height)")
    ncol = 4 + len(triggers)
    ax.legend(loc="lower left", fontsize=7, ncol=ncol, framealpha=0.8)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj",   default=os.path.join(BASE, "trajectories_1.csv"))
    ap.add_argument("--volt",   default=os.path.join(BASE, "voltages_1.csv"))
    ap.add_argument("--config", default=os.path.join(BASE, "trap_config.lua"),
                    help="Path to trap_config.lua (for trigger overlays)")
    ap.add_argument("--fps",   type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=None,
                    help="µs of sim time per wall-second (default: 20 s total)")
    ap.add_argument("--save",  default=None,
                    help="Output file, e.g. animation.mp4 (requires ffmpeg)")
    args = ap.parse_args()

    ions  = load_trajectories(args.traj)
    volts = load_voltages(args.volt)

    # Keep only the first n ion IDs (by sorted order) to exclude workbench
    # placeholder ions that are terminated on the first timestep.
    n_cfg = load_n_particles(args.config)
    if n_cfg is not None and len(ions) > n_cfg:
        keep = sorted(ions.keys())[:n_cfg]
        print(f"  Showing {n_cfg} of {len(ions)} ions (filtered by particles.n in config)")
        ions = {k: ions[k] for k in keep}

    _TRIG_PALETTE = ["darkorchid", "tomato", "mediumseagreen", "saddlebrown"]
    triggers   = load_triggers(args.config)
    trig_colors = [_TRIG_PALETTE[i % len(_TRIG_PALETTE)] for i in range(len(triggers))]
    fire_times  = compute_fire_times(ions, triggers) if ions else []
    if triggers and not ions:
        fire_times = [None] * len(triggers)

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
    has_rf3  = has_volt and len(volts.get("V_RF3", [])) > 0
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
    draw_geometry(ax_top, triggers=triggers, trig_colors=trig_colors)

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
            "V_endcap_load_U":    ("teal",       "-",            "Load endcap U (3)  [DC]"),
            "V_endcap_load_D":    ("teal",       "--",           "Load endcap D (4)  [DC]"),
            "V_dc_3_TL":          ("steelblue",  (0,(5,2)),      "Rod 3 TL DC (5)"),
            "V_dc_3_TR":          ("steelblue",  (0,(5,2,1,2)),  "Rod 3 TR DC (6)"),
            "V_dc_3_BL":          ("navy",       (0,(5,2)),      "Rod 3 BL DC (7)"),
            "V_dc_3_BR":          ("navy",       (0,(5,2,1,2)),  "Rod 3 BR DC (8)"),
            "V_endcap_optical_U": ("seagreen",   "-",            "Optical endcap U (9)  [DC]"),
            "V_endcap_optical_D": ("seagreen",   "--",           "Optical endcap D (10) [DC]"),
        }
        for key, (color, ls, label) in volt_style.items():
            if key in volts and len(volts[key]):
                ax_bot.step(vt, volts[key], where="post",
                            color=color, ls=ls, lw=1.5, label=label)
        if has_rf:
            ax_bot.step(vt, volts["V_RF"], where="post",
                        color="crimson", lw=1.5, ls=(0, (3, 1, 1, 1)),
                        label="Sets 1+2 RF amplitude V₀")
        if has_rf3:
            ax_bot.step(vt, volts["V_RF3"], where="post",
                        color="darkorange", lw=1.5, ls=(0, (3, 1, 1, 1)),
                        label="Set 3 RF amplitude V₀")

        for i, (trig, t_fire) in enumerate(zip(triggers, fire_times)):
            if t_fire is not None:
                c = trig_colors[i]
                elec_str = ", ".join(str(e) for e in trig["electrodes"])
                ax_bot.axvline(t_fire, color=c, lw=1.5, ls=(0, (4, 2)), alpha=0.85,
                               label=f"Trigger {i+1} fires  (t={t_fire:.0f} µs, elec {{{elec_str}}})")

        vcursor = ax_bot.axvline(t_min, color="red", lw=1.0, ls="--", alpha=0.8, zorder=5)
        ax_bot.set_xlim(t_min, t_max)
        ax_bot.set_xlabel("Time (µs)")
        ax_bot.set_ylabel("Voltage (V)")
        ax_bot.set_title("Electrode voltages")
        ax_bot.legend(loc="upper right", fontsize=8, ncol=2)
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
                 writer='ffmpeg', bitrate=2000)
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
