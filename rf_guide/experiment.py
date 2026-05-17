"""experiment.py  –  Per-experiment dynamics for trapsim.

Edit this file to change particles, voltages, triggers, and physics for a
given run.  All electrode names must match those declared in geometry.yaml.

Run with:
    python -m trapsim.fly
or
    python run.py
"""

import numpy as np

from trapsim.physics import Electrostatic, Gravity, EpsteinDrag, Langevin


# ── Particle ──────────────────────────────────────────────────────────────
particle = {
    "radius_m":     83e-9,     # 166 nm diameter
    "density_kgm3": 2200.0,    # fused silica
    "charge_e":     100,
}

particles = {
    "n": 20,
    "starts": [
        # Position in Fusion-world mm.  GEM offset is automatic.
        # sigma_mm is the 1-σ Gaussian spread per axis.
        {
            "position_mm": [0.0, 19.0, -98.12],
            "ke_ev":       0.0,
            "direction":   [0, 0, 1],          # +z (only relevant if ke_ev > 0)
            "sigma_mm":    [0.0, 0.0, 0.1],
        },
    ],
}

# ── Physics ───────────────────────────────────────────────────────────────
# Any physics object exposing accel/damping_rate/kick works; add your own.
physics = [
    Electrostatic(),
    Gravity(g_mm_us2=9.81e-9, axis="-y"),
    EpsteinDrag(
        pressure_pa=0.1, temperature_k=293, gas_mass_amu=28.0,
        pressure_ramp={"trigger":     "catch_sphere",
                       "p_final_pa":  100.0,
                       "duration_us": 5e5},
        scale=1.0,
    ),
    Langevin(temperature_k=293),
]

# ── Integrator ────────────────────────────────────────────────────────────
integrator = {
    "dt_init_us":    1.0,
    "dt_min_us":     0.01,
    "dt_max_us":     25.0,        # ≤ T_RF/20 at 2 kHz
    "atol":          1e-3,
    "rtol":          1e-4,
    "v_stop_mm_us":  1e-6,
    "record_stride": 20,
}

# ── Main voltage schedule ─────────────────────────────────────────────────
t = np.linspace(0, 5e5, 1000)
RF_FREQ_HZ = 2000.0
ones = np.ones_like(t)
zero = np.zeros_like(t)

main_schedule = {
    "time_us": t,
    "dc": {
        "endcap_load_U":     10.0 * ones,
        "endcap_load_D":    -10.0 * ones,
        "endcap_optical_U":  zero,
        "endcap_optical_D":  zero,
        "rod_3_TL": zero, "rod_3_TR": zero,
        "rod_3_BL": zero, "rod_3_BR": zero,
    },
    "rf": {
        "rf_loading":     {"amplitude":  10.0 * ones,
                            "frequency_hz": RF_FREQ_HZ, "phase_deg":   0},
        "rf_loading_inv": {"amplitude":  10.0 * ones,
                            "frequency_hz": RF_FREQ_HZ, "phase_deg": 180},
        "rod_3_TL":       {"amplitude": 300.0 * ones,
                            "frequency_hz": RF_FREQ_HZ, "phase_deg":   0},
        "rod_3_BR":       {"amplitude": 300.0 * ones,
                            "frequency_hz": RF_FREQ_HZ, "phase_deg":   0},
        "rod_3_TR":       {"amplitude": 300.0 * ones,
                            "frequency_hz": RF_FREQ_HZ, "phase_deg": 180},
        "rod_3_BL":       {"amplitude": 300.0 * ones,
                            "frequency_hz": RF_FREQ_HZ, "phase_deg": 180},
    },
}

# ── Triggers ──────────────────────────────────────────────────────────────
# Each trigger fires when pos[axis] >= threshold_mm (Fusion-world coords).
# When fired, its schedule overrides ONLY the listed electrodes from
# t_fire onward.  Other electrodes continue on main_schedule.

# Trigger 1: drop the loading-region downstream endcap when the ion has
# crossed the gate-valve gap.
t_drop = np.array([0.0, 200.0, 200.1, 1e6])
v_drop = np.array([-10.0, -10.0, 0.0, 0.0])

# Trigger 2: ramp set-3 DC trims into the trapping configuration once the
# ion arrives at the optical Paul trap.  Also pulses the optical endcaps.
t_rel  = np.arange(0.0, 1e6 + 5.0, 10.0)
ramp   = np.where(
    t_rel > 1e5,
    np.minimum((t_rel - 1e5) * 15.0 / 1e5, 15.0),
    0.0,
)

triggers = [
    {
        "name":         "throw_sphere",
        "axis":         "z",
        "threshold_mm": -83.52,
        "schedule": {
            "time_us": t_drop,
            "dc": {"endcap_load_D": v_drop},
        },
    },
    {
        "name":         "catch_sphere",
        "axis":         "z",
        "threshold_mm": 272.0,
        "schedule": {
            "time_us": t_rel,
            "dc": {
                "rod_3_TL": -ramp, "rod_3_TR":  ramp,
                "rod_3_BL": -ramp, "rod_3_BR":  ramp,
                "endcap_optical_U": 20.0 * np.ones_like(t_rel),
                "endcap_optical_D": 20.0 * np.ones_like(t_rel),
            },
        },
    },
]
