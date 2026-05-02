"""
generate_voltages.py  –  Define DC electrode pulse sequences for the Paul trap.

Edit the SCHEDULE list below, then run:
    python generate_voltages.py

Output: voltages_<OUT_NUMBER>.csv  (loaded by SIMION when voltage_file_enable=1).
Set voltage_file_number in SIMION to match OUT_NUMBER.

Format: one row per time point; voltages are linearly interpolated between rows.
For a sharp step, add two rows at the same time (or 0.1 µs apart).

Electrodes controlled here:
    V_endcap  –  electrode 3 (left end cap)
    V_ring_L  –  electrode 6 (ring near gap left edge)
    V_ring_R  –  electrode 7 (ring near gap right edge)

RF frequency and amplitude envelope are also set here:
    f_RF      –  carrier frequency (Hz), written as a metadata comment
    V_RF      –  zero-to-peak amplitude vs time (can vary — enables adiabatic ramps)
"""

import numpy as np
import os

BASE       = os.path.dirname(os.path.abspath(__file__))
OUT_NUMBER = 1        # matches voltage_file_number in SIMION

# ── RF parameters ─────────────────────────────────────────────────────────────
f_RF = 1.5e4           # Hz — carrier frequency of the quadrupole RF

# ── Pulse sequence ────────────────────────────────────────────────────────────
# Each row: (time_us, V_endcap, V_ring_L, V_ring_R)
# The simulation starts at t=0; extend the last row to cover your full run time.

max_time = 2e5
times = np.linspace(0, max_time, 1000)
V_endcap = np.zeros_like(times)
V_endcap[times < 3e5] = 500
V_ring_L = -200 * np.ones_like(times)
t_L_start = 2.6e4
T_switch = 5e4
V_ring_L[times > t_L_start] = -200 * np.cos(2 * np.pi / T_switch * (times[times > t_L_start] - t_L_start))
V_ring_L[times > t_L_start + T_switch / 2] = 0
V_ring_R = -500 * np.ones_like(times)
t_R_start = 4e4
V_ring_R[times > t_R_start] = -500 * np.cos(2 * np.pi / T_switch * (times[times > t_R_start] - t_R_start))
V_ring_R[times > t_R_start + T_switch/4] = 0

# ── RF amplitude envelope ─────────────────────────────────────────────────────
# V_RF sets the zero-to-peak amplitude of the quadrupole RF at each time point.
# Linearly interpolated by SIMION between rows — same time axis as the DC voltages.
# Example: adiabatic ramp-down starting at t = 1e5 µs
#   V_RF = 100 * np.ones_like(times)
#   ramp_mask = times > 1e5
#   V_RF[ramp_mask] = 100 * np.exp(-(times[ramp_mask] - 1e5) / 3e4)
V_RF = 150 * np.ones_like(times)   # constant amplitude (edit to modulate)
# T_mod = 3e4
# mod_inds = (times > t_L_start) & (times <= t_L_start + T_mod)
# V_RF[mod_inds] *= 0.75 + 0.25 * np.cos(2 * np.pi / T_mod * (times[mod_inds] - t_L_start))

# ── Right end cap ─────────────────────────────────────────────────────────────
V_endcap_R = np.copy(V_endcap)   # right end cap DC voltage vs time
t_ec_R_start = 1e5
V_endcap_R[times < t_ec_R_start] *= -1

V_endcap_R = -500 * np.ones(len(times))
V_endcap = 500 * np.ones(len(times))
V_ring_L = np.zeros(len(times))
V_ring_R = np.zeros(len(times))

SCHEDULE = list(np.vstack((times, V_endcap, V_endcap_R, V_ring_L, V_ring_R, V_RF)).T)

# ── Write CSV ─────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE, f"voltages_{OUT_NUMBER}.csv")
with open(out_path, "w") as f:
    f.write(f"# f_RF_Hz={f_RF}\n")
    f.write("time_us,V_endcap,V_endcap_R,V_ring_L,V_ring_R,V_RF\n")
    for row in SCHEDULE:
        f.write(f"{row[0]:.2f},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f}\n")

print(f"Written {len(SCHEDULE)} rows → {out_path}")

# ── Quick preview plot ─────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    t   = np.array([r[0] for r in SCHEDULE])
    v3  = np.array([r[1] for r in SCHEDULE])
    v8  = np.array([r[2] for r in SCHEDULE])
    v6  = np.array([r[3] for r in SCHEDULE])
    v7  = np.array([r[4] for r in SCHEDULE])
    vrf = np.array([r[5] for r in SCHEDULE])

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.step(t, v3,  where="post", color="teal",      lw=1.5, label="End cap L (3)  [DC]")
    ax.step(t, v8,  where="post", color="teal",      lw=1.5, ls="--", label="End cap R (8)  [DC]")
    ax.step(t, v6,  where="post", color="goldenrod", lw=1.5, label="Ring L (6)  [DC]")
    ax.step(t, v7,  where="post", color="goldenrod", lw=1.5, ls="--", label="Ring R (7)  [DC]")
    ax.step(t, vrf, where="post", color="crimson",   lw=1.5, ls=":",  label="RF amplitude V₀")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"Voltage schedule — voltages_{OUT_NUMBER}.csv   (f_RF = {f_RF} Hz)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
except ImportError:
    pass
