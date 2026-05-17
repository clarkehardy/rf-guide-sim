"""experiment.py  –  Per-experiment dynamics for trapsim.

Edit this file to change particles, voltages, triggers, and physics for a
given run.  All electrode names must match those declared in geometry.yaml.

Run with:
    python -m trapsim.fly
or
    python run.py
"""

import numpy as np

from trapsim.physics import Electrostatic, Gravity, ContinuumDrag, Langevin


# ── Particle ──────────────────────────────────────────────────────────────
particle = {
    "radius_m":     10e-6,     # 20 um diameter
    "density_kgm3": 789.0,    # ethanol
    "charge_e":     1e6,      # number of charges per ethanol droplet
}

particles = {
    "n": 20,
    "starts": [
        # Position in Fusion-world mm.  GEM offset is automatic.
        # sigma_mm is the 1-σ Gaussian spread per axis.
        {
            "position_mm": [0.0, 31.0, -98.12], # start the particles at the location of the inkjet tip
            "ke_ev":       0.0,
            "direction":   [0, 0, 1],          # +z (only relevant if ke_ev > 0)
            "sigma_mm":    [0.5, 0.5, 0.5],    # spread in initial position
        },
    ],
}

# ── Physics ───────────────────────────────────────────────────────────────
# Any physics object exposing accel/damping_rate/kick works; add your own.
physics = [
    Electrostatic(),
    Gravity(g_mm_us2=9.81e-9, axis="-y"),
    ContinuumDrag(rho_gas_kg_m3=1.225, eta_pa_s=1.81e-5), # drag at atmospheric pressure
    Langevin(temperature_k=293.),                         # FDT noise
]

# ── Integrator ────────────────────────────────────────────────────────────
integrator = {
    "dt_init_us":    1.0,
    "dt_min_us":     0.01,
    "dt_max_us":     25.0,        # ≤ T_RF/20 at 2 kHz
    "atol":          1e-3,
    "rtol":          1e-4,
    "v_stop_mm_us":  1e-12,
    "record_stride": 20,
}

# ── Main voltage schedule ─────────────────────────────────────────────────
t = np.linspace(0, 2e6, 1000) # 2 second simulation time
RF_FREQ_HZ = 1000.0
ones = np.ones_like(t)
zero = np.zeros_like(t)

main_schedule = {
    "time_us": t,
    "dc": {
        "endcap_load_U":     zero,
        "endcap_load_D":     zero,
    },
    "rf": {
        "rf_loading":     {"amplitude":  zero, # start the rf at zero; it turns on when triggered
                            "frequency_hz": RF_FREQ_HZ, "phase_deg":   0},
        "rf_loading_inv": {"amplitude":  zero,
                            "frequency_hz": RF_FREQ_HZ, "phase_deg": 180},
    },
}

# ── Triggers ──────────────────────────────────────────────────────────────
# Each trigger fires when pos[axis] >= threshold_mm (Fusion-world coords).
# When fired, its schedule overrides ONLY the listed electrodes from
# t_fire onward.  Other electrodes continue on main_schedule.

# Trigger 1: turn on the Paul trap when the droplet is between the rods.

triggers = [
    {
        "name":         "turn_on_rf",
        "axis":         "-y", # trigger threshold is in y in the negative-going (downward) direction
        "threshold_mm": 19.0,
        "schedule": {
            "time_us": t,
            # "dc": {
            #         "endcap_load_U":     10.0 * ones,
            #         "endcap_load_D":     10.0 * ones,
            #       },
            "rf": {
                    "rf_loading":     {"amplitude":  200 * ones, # set the rf to its nominal amplitude
                                        "frequency_hz": RF_FREQ_HZ, "phase_deg":   0},
                    "rf_loading_inv": {"amplitude":  200 * ones,
                                        "frequency_hz": RF_FREQ_HZ, "phase_deg": 180},
                  }
        },
    },
]
