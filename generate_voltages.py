"""
generate_voltages.py  –  Define DC electrode pulse sequences for the Paul trap.

Edit the SCHEDULE list below, then run:
    python generate_voltages.py

Output: voltages_<OUT_NUMBER>.csv  (loaded by SIMION when voltage_file_enable=1).
Set voltage_file_number in SIMION to match OUT_NUMBER.

Format: one row per time point; voltages are linearly interpolated between rows.
For a sharp step, add two rows at the same time (or 0.1 µs apart).

Electrodes controlled here (main Paul trap, axis along Z):
    V_endcap   –  electrode 3  (left end cap)
    V_endcap_R –  electrode 8  (right end cap)
    V_ring_L   –  electrode 6  (ring near gap left edge)
    V_ring_R   –  electrode 7  (ring near gap right edge)

Electrodes controlled here (perpendicular trap, axis along X):
    V_trap_lens –  electrode 11  (trapping_lens_holder)
    V_coll_lens –  electrode 12  (collection_lens_holder)

RF frequency and amplitude envelopes:
    f_RF       –  main trap carrier frequency (Hz), written as metadata comment
    V_RF       –  main trap zero-to-peak amplitude vs time
    f_RF2      –  perpendicular trap carrier frequency (Hz) [PLACEHOLDER]
    V_RF2      –  perpendicular trap zero-to-peak amplitude vs time [PLACEHOLDER]
"""

import numpy as np
import os

BASE       = os.path.dirname(os.path.abspath(__file__))
OUT_NUMBER = 1        # matches voltage_file_number in SIMION

# ── RF parameters ─────────────────────────────────────────────────────────────
f_RF  = 2e4          # Hz — carrier frequency of the main Paul trap (axis along Z)

# PLACEHOLDER: Set f_RF2 to the correct RF frequency for the perpendicular trap
# (axis along X).  Stability parameter q ≈ 4eV₀/(m ω² r₀²) — use the known r₀
# and target q < 0.908.
f_RF2 = 2e4          # Hz — PLACEHOLDER (currently same as f_RF; adjust as needed)

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
V_RF = 100 * np.ones_like(times)   # constant amplitude (edit to modulate)
# T_mod = 3e4
# mod_inds = (times > t_L_start) & (times <= t_L_start + T_mod)
# V_RF[mod_inds] *= 0.75 + 0.25 * np.cos(2 * np.pi / T_mod * (times[mod_inds] - t_L_start))

# ── Right end cap ─────────────────────────────────────────────────────────────
V_endcap_R = np.copy(V_endcap)   # right end cap DC voltage vs time
t_ec_R_start = 1e5
V_endcap_R[times < t_ec_R_start] *= -1

V_endcap_R = 100 * np.ones(len(times))
V_endcap = np.zeros(len(times))
V_ring_L = 50 * np.ones(len(times))
V_ring_R = -200 * np.ones(len(times))

# ── Perpendicular trap RF amplitude ───────────────────────────────────────────
# PLACEHOLDER: Set V_RF2 to the zero-to-peak RF voltage needed to trap a particle
# in the perpendicular trap.  Currently 0 V (rods are inactive).
V_RF2 = np.zeros_like(times)   # PLACEHOLDER — set to trapping amplitude

# ── Perpendicular trap DC voltages ────────────────────────────────────────────
# PLACEHOLDER: Set axial confinement voltages for the perpendicular trap.
# V_trap_lens (electrode 11) and V_coll_lens (electrode 12) act as end caps.
V_trap_lens = np.zeros_like(times)   # PLACEHOLDER — trapping_lens_holder bias (V)
V_coll_lens = np.zeros_like(times)   # PLACEHOLDER — collection_lens_holder bias (V)

SCHEDULE = list(np.vstack((
    times,
    V_endcap, V_endcap_R, V_ring_L, V_ring_R, V_RF,
    V_RF2, V_trap_lens, V_coll_lens,
)).T)

# ── Write CSV ─────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE, f"voltages_{OUT_NUMBER}.csv")
with open(out_path, "w") as f:
    f.write(f"# f_RF_Hz={f_RF}\n")
    f.write(f"# f_RF2_Hz={f_RF2}\n")
    f.write("time_us,V_endcap,V_endcap_R,V_ring_L,V_ring_R,V_RF,"
            "V_RF2,V_trap_lens,V_coll_lens\n")
    for row in SCHEDULE:
        f.write(f"{row[0]:.2f},"
                f"{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f},"
                f"{row[6]:.6f},{row[7]:.6f},{row[8]:.6f}\n")

print(f"Written {len(SCHEDULE)} rows → {out_path}")

# ── Quick preview plot ─────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    t    = np.array([r[0] for r in SCHEDULE])
    v3   = np.array([r[1] for r in SCHEDULE])
    v8   = np.array([r[2] for r in SCHEDULE])
    v6   = np.array([r[3] for r in SCHEDULE])
    v7   = np.array([r[4] for r in SCHEDULE])
    vrf  = np.array([r[5] for r in SCHEDULE])
    vrf2 = np.array([r[6] for r in SCHEDULE])
    v11  = np.array([r[7] for r in SCHEDULE])
    v12  = np.array([r[8] for r in SCHEDULE])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    ax1.step(t, v3,  where="post", color="teal",      lw=1.5, label="End cap L (3)  [DC]")
    ax1.step(t, v8,  where="post", color="teal",      lw=1.5, ls="--", label="End cap R (8)  [DC]")
    ax1.step(t, v6,  where="post", color="goldenrod", lw=1.5, label="Ring L (6)  [DC]")
    ax1.step(t, v7,  where="post", color="goldenrod", lw=1.5, ls="--", label="Ring R (7)  [DC]")
    ax1.step(t, vrf, where="post", color="crimson",   lw=1.5, ls=":",  label=f"RF V₀  (f={f_RF:.0f} Hz)")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(f"Main trap (Z-axis) — voltages_{OUT_NUMBER}.csv")
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    ax2.step(t, v11,  where="post", color="steelblue",  lw=1.5, label="Trap lens holder (11)  [DC]")
    ax2.step(t, v12,  where="post", color="steelblue",  lw=1.5, ls="--", label="Coll lens holder (12)  [DC]")
    ax2.step(t, vrf2, where="post", color="darkorange", lw=1.5, ls=":",  label=f"RF2 V₀  (f={f_RF2:.0f} Hz)")
    ax2.set_xlabel("Time (µs)")
    ax2.set_ylabel("Voltage (V)")
    ax2.set_title("Perpendicular trap (X-axis)")
    ax2.legend(fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
except ImportError:
    pass
