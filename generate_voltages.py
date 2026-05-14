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

# ── RF parameters ─────────────────────────────────────────────────────────────
f_RF  = 5.5e3          # Hz — carrier frequency, sets 1 + 2 (loading + RF guide)

# PLACEHOLDER: Set f_RF3 to the correct RF frequency for the optical Paul trap (set 3).
# Stability parameter q ≈ 4 Q V_rf / (m ω² r_0²) — choose with the known r_0 and a
# target q < 0.908.
f_RF3 = 1e6            # Hz — PLACEHOLDER

# ── Time base ─────────────────────────────────────────────────────────────────
max_time = 2e5
times    = np.linspace(0, max_time, 1000)

# ── Sets 1 + 2 RF amplitude envelope ──────────────────────────────────────────
# V_RF sets the zero-to-peak amplitude of the loading + RF guide quadrupole.
# SIMION interpolates linearly between rows.  Example adiabatic ramp-down:
#   V_RF = 100 * np.ones_like(times)
#   ramp_mask = times > 1e5
#   V_RF[ramp_mask] = 100 * np.exp(-(times[ramp_mask] - 1e5) / 3e4)
V_RF = 100 * np.ones_like(times)

# ── Load Paul trap endcaps ────────────────────────────────────────────────────
V_endcap_load_U = np.zeros_like(times)
V_endcap_load_D = np.zeros_like(times)

# ── Set 3 (optical Paul trap) ─────────────────────────────────────────────────
# PLACEHOLDER: set V_RF3 to the trapping amplitude for the wider Paul trap.
V_RF3 = 0 * np.ones_like(times)

# Per-rod DC trims (positive shifts equilibrium away from that rod).
# Apply equal-and-opposite trims on opposite-side rods to shift equilibrium
# laterally; common-mode trims raise the mean rod potential.
V_dc_3_TL = np.zeros_like(times)
V_dc_3_TR = np.zeros_like(times)
V_dc_3_BL = np.zeros_like(times)
V_dc_3_BR = np.zeros_like(times)

# ── Optical Paul trap endcaps (gated by trigger in trap_config.lua) ───────────
# These follow the schedule from t = 0 of the trigger firing, not absolute t.
V_endcap_optical_U = np.zeros_like(times)
V_endcap_optical_D = np.zeros_like(times)

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

# ── Write CSV ─────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE, f"voltages_{OUT_NUMBER}.csv")
header   = ",".join(name for name, _ in COLUMNS)
data     = np.vstack([arr for _, arr in COLUMNS]).T

with open(out_path, "w") as f:
    f.write(f"# f_RF_Hz={f_RF}\n")
    f.write(f"# f_RF3_Hz={f_RF3}\n")
    f.write(header + "\n")
    for row in data:
        f.write(f"{row[0]:.2f}," + ",".join(f"{v:.6f}" for v in row[1:]) + "\n")

print(f"Written {len(data)} rows → {out_path}")

# ── Preview plot ──────────────────────────────────────────────────────────────
try:
    if _args.no_preview:
        raise ImportError
    import matplotlib.pyplot as plt

    t = times

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

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

    ax2.step(t, V_RF3,              where="post", color="darkorange", lw=1.5,
             label=f"V_RF3 (set 3,  f={f_RF3:.0f} Hz)")
    ax2.step(t, V_dc_3_TL,          where="post", color="steelblue",  lw=1.2,
             label="V_dc_3_TL (5)")
    ax2.step(t, V_dc_3_TR,          where="post", color="steelblue",  lw=1.2, ls="--",
             label="V_dc_3_TR (6)")
    ax2.step(t, V_dc_3_BL,          where="post", color="navy",       lw=1.2,
             label="V_dc_3_BL (7)")
    ax2.step(t, V_dc_3_BR,          where="post", color="navy",       lw=1.2, ls="--",
             label="V_dc_3_BR (8)")
    ax2.step(t, V_endcap_optical_U, where="post", color="seagreen",   lw=1.5,
             label="V_endcap_optical_U (9)")
    ax2.step(t, V_endcap_optical_D, where="post", color="seagreen",   lw=1.5, ls="--",
             label="V_endcap_optical_D (10)")
    ax2.set_xlabel("Time (µs)")
    ax2.set_ylabel("Voltage (V)")
    ax2.set_title("Optical Paul trap (set 3 + optical endcaps)")
    ax2.legend(fontsize=8, ncol=4)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
except ImportError:
    pass
