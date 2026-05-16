# trap_config.py  –  Particle and trap configuration for fly.py
#
# Edit directly, then re-run fly.py (or run_simulation.py).
# This is a 1:1 translation of legacy/trap_config.lua into Python.
#
# fly.py imports this file and reads the `config` dict.

config = {

    # ── Gas ───────────────────────────────────────────────────────────────────
    "pressure_pa":          0.1,    # baseline pressure [Pa]  (100 Pa = 1 mbar)
    "temperature_k":        293,    # K
    "gas_molar_mass_amu":   28.0,   # amu  (28 = N2)

    # ── Pressure ramp (optional) ──────────────────────────────────────────────
    # Models opening a solenoid valve when the particle reaches a z threshold.
    # Pressure ramps linearly from pressure_pa to P_final_pa over duration_us,
    # starting at the moment the named trigger fires for that ion.
    # Set to None (or omit) to keep pressure constant.
    "pressure_ramp": {
        "trigger":      2,         # index (1-based) into the triggers list below
        "P_final_pa":   100.0,     # target pressure [Pa]
        "duration_us":  5e5,       # linear ramp duration [µs]
    },

    # ── Particle ──────────────────────────────────────────────────────────────
    "particle_radius_m":      83e-9,  # m  (166 nm diameter silica sphere)
    "particle_density_kgm3":  2200,   # kg/m³  (fused silica)

    # ── Drag / termination / recording ────────────────────────────────────────
    "drag_scale":     1.0,    # multiply Epstein drag rate; 0 disables drag (and noise)
    "langevin_noise": True,   # thermal noise paired with drag (fluctuation-dissipation)
    "v_stop_mm_us":   1e-6,   # terminate when speed < this [mm/µs]; 0 to disable
    "record_stride":  20,     # write trajectory row every N accepted steps; 0 to disable

    # ── Adaptive integrator ───────────────────────────────────────────────────
    "dt_init_us":  1.0,    # initial step size [µs]
    "dt_min_us":   0.01,   # minimum allowed step size [µs]
    "dt_max_us":   25.0,   # maximum allowed step size [µs]  (≤ T_RF/20 at 2 kHz)
    "atol":        1e-3,   # absolute tolerance [mm or mm/µs]
    "rtol":        1e-4,   # relative tolerance

    # ── Coordinate offsets: GEM → Fusion world (mm) ───────────────────────────
    # Must match the locate(tx, ty, tz) block in paulTrap.processed.gem.
    "gem_offset": {"x": 25.0, "y": 8.0, "z": 132.0},

    # ── Triggers ──────────────────────────────────────────────────────────────
    # Each trigger holds its listed electrodes on the main schedule until the
    # ion's Fusion-Z coordinate first reaches z_mm, then switches to the
    # post-trigger schedule (V_e{N}_trig columns in the voltage CSV).
    # Triggers are per-ion; state resets for each new ion.
    "triggers": [
        {"z_mm": -83.52, "electrodes": [4]},           # Trigger 1
        {"z_mm": 272.0,  "electrodes": [5, 6, 7, 8, 9, 10]},  # Trigger 2
    ],

    # ── Particle definitions ──────────────────────────────────────────────────
    # n:         number of particles to simulate
    # charge:    elementary charges
    # mass:      derived from particle_radius_m and particle_density_kgm3
    # Positions in Fusion world coordinates (mm); gem_offset added automatically.
    # ke_ev:     kinetic energy in eV  (0 = stationary)
    # az, el:    direction angles in degrees  (az=0,el=0 → +Z)
    # sigma_mm:  1-σ Gaussian spread per axis  (omit or zero for point source)
    # Multiple starts entries are assigned round-robin by ion number.
    "particles": {
        "n":      20,
        "charge": 100,
        "starts": [
            {
                "x_mm": 0, "y_mm": 19, "z_mm": -98.12,
                "ke_ev": 0,
                "sigma_mm": {"x": 0, "y": 0, "z": 0.1},
            },
        ],
    },
}
