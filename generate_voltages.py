"""
generate_voltages.py  –  Define voltage schedules for the Paul trap simulation.

Edit the SCHEDULE block below, then run:
    python generate_voltages.py

Output: voltages_<OUT_NUMBER>.csv  (loaded by SIMION at run start).
Set voltage_file_number in SIMION to match OUT_NUMBER.

Format: one row per time point; voltages are linearly interpolated between rows.
For a sharp step, add two rows at the same time (or 0.1 µs apart).

Electrode channels (10 SIMION electrodes total):

    Sets 1 + 2 (loading Paul trap + RF guide rods, wired in parallel):
        V_RF            –  electrodes 1 (+RF on TL/BR) and 2 (-RF on TR/BL)
        f_RF            –  carrier frequency, written as metadata comment

    Load Paul trap endcaps:
        V_endcap_load_U –  electrode 3 (+z endcap)
        V_endcap_load_D –  electrode 4 (-z endcap)

    Set 3 (optical Paul trap, 4 fully-independent SIMION electrodes):
        V_RF3           –  shared amplitude: applied as +V_RF3·cos(ω₃t) on
                            rod_3_TL & rod_3_BR; -V_RF3·cos(ω₃t) on rod_3_TR & rod_3_BL
        f_RF3           –  carrier frequency, written as metadata comment
        V_dc_3_TL       –  electrode 5 DC trim (rod_3_TL)
        V_dc_3_TR       –  electrode 6 DC trim (rod_3_TR)
        V_dc_3_BL       –  electrode 7 DC trim (rod_3_BL)
        V_dc_3_BR       –  electrode 8 DC trim (rod_3_BR)

    Optical Paul trap endcaps (gated by trigger; see trap_config.lua):
        V_endcap_optical_U –  electrode 9  (+z endcap)
        V_endcap_optical_D –  electrode 10 (-z endcap)
"""

import argparse
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description="Generate Paul trap voltage schedule CSV.")
ap.add_argument("--out", type=int, default=1, metavar="N",
                help="Output file number: writes voltages_N.csv (default: 1)")
ap.add_argument("--no-preview", action="store_true",
                help="Skip the matplotlib preview plot")
_args = ap.parse_args()
OUT_NUMBER = _args.out

skip_loading = False

# ── RF parameters ─────────────────────────────────────────────────────────────
f_RF  = 2e3          # Hz — carrier frequency, sets 1 + 2 (loading + RF guide)

# PLACEHOLDER: Set f_RF3 to the correct RF frequency for the optical Paul trap (set 3).
# Stability parameter q ≈ 4 Q V_rf / (m ω² r_0²) — choose with the known r_0 and a
# target q < 0.908.
f_RF3 = 2e3            # Hz — PLACEHOLDER

# ── Time base ─────────────────────────────────────────────────────────────────
max_time = 2e5
times    = np.linspace(0, max_time, 1000)

# ── Sets 1 + 2 RF amplitude envelope ──────────────────────────────────────────
# V_RF sets the zero-to-peak amplitude of the loading + RF guide quadrupole.
# SIMION interpolates linearly between rows.  Example adiabatic ramp-down:
#   V_RF = 100 * np.ones_like(times)
#   ramp_mask = times > 1e5
#   V_RF[ramp_mask] = 100 * np.exp(-(times[ramp_mask] - 1e5) / 3e4)
V_RF = 10 * np.ones_like(times)

# ── Load Paul trap endcaps ────────────────────────────────────────────────────
V_endcap_load_U = 10 * np.ones_like(times)
V_endcap_load_D = -10 * np.ones_like(times)

if skip_loading:
    V_endcap_load_D = np.zeros_like(times)
    V_endcap_load_D = 10 * np.ones_like(times)

# ── Set 3 (optical Paul trap) ─────────────────────────────────────────────────
# PLACEHOLDER: set V_RF3 to the trapping amplitude for the wider Paul trap.
V_RF3 = 300 * np.ones_like(times)

# Per-rod DC trims (positive shifts equilibrium away from that rod).
# Apply equal-and-opposite trims on opposite-side rods to shift equilibrium
# laterally; common-mode trims raise the mean rod potential.
V_dc_3_TL = np.zeros_like(times)
V_dc_3_TR = np.zeros_like(times)
V_dc_3_BL = np.zeros_like(times)
V_dc_3_BR = np.zeros_like(times)

# ── Optical Paul trap endcaps (gated by trigger in trap_config.lua) ───────────
# Fallback: used only if the post-trigger schedule below is absent from the CSV.
V_endcap_optical_U = np.zeros_like(times)
V_endcap_optical_D = np.zeros_like(times)

# ── Post-trigger schedule (electrodes 9 and 10) ───────────────────────────────
# Independent, finer time axis.  Time is measured from trigger-fire, not
# absolute simulation time.  paulTrap.lua prefers these columns over the
# V_endcap_optical_U/D fallback columns above.
max_time_trig = 1000000          # µs — total duration of the post-trigger schedule
dt_trig       = 10           # µs — time step (finer than the main schedule)
times_trig    = np.arange(0, max_time_trig + dt_trig / 2, dt_trig)

# PLACEHOLDER: define pulse shape here.
# Column names must be V_e{N}_trig where N is the SIMION electrode number.
# Add or remove entries to match the electrodes listed in trap_config.lua triggers.
V_e9_trig  = 20 * np.ones_like(times_trig)   # endcap_optical_U
V_e10_trig = 20 * np.ones_like(times_trig)   # endcap_optical_D
V_e4_trig  = V_endcap_load_D[-1] * np.ones_like(times_trig)
V_e4_trig[times_trig > 2e2] = 0

# Per-rod DC trims after trigger 2 fires (time since trigger, same axis as above).
# Before trigger 2, each rod uses the main-schedule V_dc_3_XX value.
V_e5_trig = np.zeros_like(times_trig)   # rod_3_TL DC trim
V_e6_trig = np.zeros_like(times_trig)   # rod_3_TR DC trim
V_e7_trig = np.zeros_like(times_trig)   # rod_3_BL DC trim
V_e8_trig = np.zeros_like(times_trig)   # rod_3_BR DC trim

V_e5_trig[times_trig > 1e5] = -(times_trig[times_trig > 1e5] - 1e5) * 15.0 / 1e5
V_e6_trig[times_trig > 1e5] = (times_trig[times_trig > 1e5] - 1e5) * 15.0 / 1e5
V_e7_trig[times_trig > 1e5] = -(times_trig[times_trig > 1e5] - 1e5) * 15.0 / 1e5
V_e8_trig[times_trig > 1e5] = (times_trig[times_trig > 1e5] - 1e5) * 15.0 / 1e5

V_e5_trig[times_trig > 2e5] = 15.0
V_e6_trig[times_trig > 2e5] = -15.0
V_e7_trig[times_trig > 2e5] = -15.0
V_e8_trig[times_trig > 2e5] = 15.0

# ── Assemble schedule ─────────────────────────────────────────────────────────
COLUMNS = [
    ("time_us",            times),
    ("V_RF",               V_RF),
    ("V_RF3",              V_RF3),
    ("V_endcap_load_U",    V_endcap_load_U),
    ("V_endcap_load_D",    V_endcap_load_D),
    ("V_dc_3_TL",          V_dc_3_TL),
    ("V_dc_3_TR",          V_dc_3_TR),
    ("V_dc_3_BL",          V_dc_3_BL),
    ("V_dc_3_BR",          V_dc_3_BR),
    ("V_endcap_optical_U", V_endcap_optical_U),
    ("V_endcap_optical_D", V_endcap_optical_D),
]
TRIG_COLUMNS = [
    ("time_trig_us", times_trig),
    ("V_e4_trig",  V_e4_trig),    # endcap_load_D — triggered
    ("V_e5_trig",  V_e5_trig),    # rod_3_TL DC trim — triggered
    ("V_e6_trig",  V_e6_trig),    # rod_3_TR DC trim — triggered
    ("V_e7_trig",  V_e7_trig),    # rod_3_BL DC trim — triggered
    ("V_e8_trig",  V_e8_trig),    # rod_3_BR DC trim — triggered
    ("V_e9_trig",  V_e9_trig),    # endcap_optical_U — triggered
    ("V_e10_trig", V_e10_trig),   # endcap_optical_D — triggered
    # To trigger electrode 3 (endcap_load_U): add ("V_e3_trig", V_e3_trig) here
    # and define V_e3_trig = ... above, and add electrode 3 to a trigger in trap_config.lua.
]

# ── Write CSV ─────────────────────────────────────────────────────────────────
out_path  = os.path.join(BASE, f"voltages_{OUT_NUMBER}.csv")
header    = ",".join(name for name, _ in COLUMNS + TRIG_COLUMNS)
data      = np.vstack([arr for _, arr in COLUMNS]).T
trig_data = np.vstack([arr for _, arr in TRIG_COLUMNS]).T
n_main    = len(COLUMNS)
n_trig    = len(TRIG_COLUMNS)

with open(out_path, "w") as f:
    f.write(f"# f_RF_Hz={f_RF}\n")
    f.write(f"# f_RF3_Hz={f_RF3}\n")
    f.write(header + "\n")
    # Main schedule rows: n_trig trailing empty fields
    for row in data:
        fields = [f"{row[0]:.2f}"] + [f"{v:.6f}" for v in row[1:]] + [""] * n_trig
        f.write(",".join(fields) + "\n")
    # Post-trigger rows: n_main leading empty fields
    for row in trig_data:
        fields = [""] * n_main + [f"{row[0]:.4f}"] + [f"{v:.6f}" for v in row[1:]]
        f.write(",".join(fields) + "\n")

print(f"Written {len(data)} main rows + {len(trig_data)} trigger rows → {out_path}")

# ── Preview plot ──────────────────────────────────────────────────────────────
try:
    if _args.no_preview:
        raise ImportError
    import matplotlib.pyplot as plt

    t = times

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
    ax2.sharex(ax1)   # main schedule panels share x; trigger panel is independent

    ax1.step(t, V_RF,            where="post", color="crimson",   lw=1.5,
             label=f"V_RF (sets 1+2,  f={f_RF:.0f} Hz)")
    ax1.step(t, V_endcap_load_U, where="post", color="teal",      lw=1.5,
             label="V_endcap_load_U (3)")
    ax1.step(t, V_endcap_load_D, where="post", color="teal",      lw=1.5, ls="--",
             label="V_endcap_load_D (4)")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(f"Loading region (sets 1+2) — voltages_{OUT_NUMBER}.csv")
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    ax2.step(t, V_RF3,     where="post", color="darkorange", lw=1.5,
             label=f"V_RF3 (set 3,  f={f_RF3:.0f} Hz)")
    ax2.step(t, V_dc_3_TL, where="post", color="steelblue",  lw=1.2, label="V_dc_3_TL (5)")
    ax2.step(t, V_dc_3_TR, where="post", color="steelblue",  lw=1.2, ls="--", label="V_dc_3_TR (6)")
    ax2.step(t, V_dc_3_BL, where="post", color="navy",       lw=1.2, label="V_dc_3_BL (7)")
    ax2.step(t, V_dc_3_BR, where="post", color="navy",       lw=1.2, ls="--", label="V_dc_3_BR (8)")
    ax2.set_xlabel("Simulation time (µs)")
    ax2.set_ylabel("Voltage (V)")
    ax2.set_title("Optical Paul trap (set 3 rods)")
    ax2.legend(fontsize=8, ncol=4)
    ax2.grid(True, alpha=0.3)
    ax3.step(times_trig, V_e4_trig,  where="post", color="seagreen",  lw=1.5, ls="--",
             label="V_e4_trig  (endcap_load_D)")
    ax3.step(times_trig, V_e5_trig,  where="post", color="steelblue", lw=1.2,
             label="V_e5_trig  (rod_3_TL DC)")
    ax3.step(times_trig, V_e6_trig,  where="post", color="steelblue", lw=1.2, ls="--",
             label="V_e6_trig  (rod_3_TR DC)")
    ax3.step(times_trig, V_e7_trig,  where="post", color="navy",      lw=1.2,
             label="V_e7_trig  (rod_3_BL DC)")
    ax3.step(times_trig, V_e8_trig,  where="post", color="navy",      lw=1.2, ls="--",
             label="V_e8_trig  (rod_3_BR DC)")
    ax3.step(times_trig, V_e9_trig,  where="post", color="seagreen",  lw=1.5,
             label="V_e9_trig  (endcap_optical_U)")
    ax3.step(times_trig, V_e10_trig, where="post", color="seagreen",  lw=1.5, ls=":",
             label="V_e10_trig (endcap_optical_D)")
    ax3.set_xlabel("Time since trigger (µs)")
    ax3.set_ylabel("Voltage (V)")
    ax3.set_title(f"Post-trigger schedule — electrodes 4–10  ({dt_trig} µs steps, {max_time_trig} µs total)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
except ImportError:
    pass
