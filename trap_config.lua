-- trap_config.lua
-- Configuration for paulTrap.lua.  Loaded once per run via dofile().
-- Edit here rather than inside the Lua program.

return {

  -- ── Gas ──────────────────────────────────────────────────────────────────
  pressure_pa        = 0.1,    -- baseline pressure [Pa]  (100 Pa = 1 mbar)
  temperature_k      = 293,    -- K
  gas_molar_mass_amu = 28.0,   -- amu  (28 = N2)

  -- ── Pressure ramp (optional, per-ion, fired by a trigger) ────────────────
  -- Models opening a solenoid valve when the particle reaches a z threshold:
  -- pressure ramps linearly from pressure_pa to P_final_pa over duration_us,
  -- starting at the moment the named trigger fires for that ion.  Before the
  -- trigger fires the pressure stays at pressure_pa.
  --
  -- Comment out this block (or omit `pressure_ramp` entirely) to keep
  -- pressure constant at pressure_pa for the whole simulation.
  pressure_ramp = {
    trigger     = 2,        -- index into the `triggers` list below
    P_final_pa  = 100.0,    -- target pressure [Pa]  (100 Pa = 1 mbar)
    duration_us = 5e5,      -- linear ramp duration [µs]
  },

  -- ── Particle ─────────────────────────────────────────────────────────────
  particle_radius_m     = 83e-9,  -- m  (166 nm diameter silica sphere)
  particle_density_kgm3 = 2200,   -- kg/m³  (fused silica)

  -- ── Drag / termination / recording ───────────────────────────────────────
  drag_scale     = 1.0,   -- multiply Epstein drag rate; 0 disables drag (and noise)
  langevin_noise = true,  -- thermal noise paired with drag (fluctuation-dissipation);
                          -- set false to recover deterministic drag-only behaviour
  v_stop_mm_us   = 1e-6,  -- terminate ion when speed < this [mm/us]; 0 to disable
  record_stride  = 20,    -- write trajectory row every N time steps; 0 to disable

  -- ── Coordinate offsets: GEM → Fusion world (mm) ──────────────────────────
  -- Must match the locate(tx, ty, tz) block in paulTrap.gem.
  gem_offset = { x = 25.0, y = 8.0, z = 132.0 },

  -- ── Triggers ─────────────────────────────────────────────────────────────
  -- Each trigger holds its listed electrodes at 0 V until the ion's Fusion-Z
  -- coordinate first reaches z_mm, then releases them to follow the normal
  -- voltage schedule.  Triggers reset for each ion (sequential simulation).
  --
  -- Only the optical-trap endcaps (electrodes 9, 10) are gated.  Sets 1+2 RF
  -- (1, 2), load endcaps (3, 4), and set-3 RF + DC trims (5–8) all follow the
  -- CSV schedule from absolute t = 0.
  --
  -- PLACEHOLDER: set z_mm to the Fusion-Z value at the centre of the optical
  -- Paul trap (between endcap_optical_U and endcap_optical_D).  Particles in
  -- the new geometry travel in -Z, so threshold should fire as the particle
  -- decelerates into the optical-trap volume.
  triggers = {
    { z_mm = -83.52, electrodes = {4} },     -- Trigger 1: flip the sign on the 
                                             -- downstream endcap to shoot the particles
    { z_mm = 272.0, electrodes = {5, 6, 7, 8, 9, 10} },  -- Trigger 2: switch rod DC trims + turn on optical trap endcaps
  },

  -- ── Particle definitions ──────────────────────────────────────────────────
  -- n:         number of ions to simulate; workbench must have at least this many
  -- charge:    elementary charges (e); overrides workbench value
  -- mass:      derived from particle_radius_m and particle_density_kgm3 above
  -- Positions are in Fusion world coordinates (mm); gem_offset is added automatically.
  -- ke_ev:     kinetic energy in eV  (0 = stationary)
  -- az, el:    direction angles in degrees (az=0,el=0 → +Z; az=90,el=0 → +X; el=90 → +Y)
  -- sigma_mm:  Gaussian 1-σ spread per axis {x,y,z} in mm; omit or zero for point source
  -- Multiple starts entries are assigned round-robin by ion number.
  particles = {
    n      = 5,
    charge = 100,
    starts = {
      -- PLACEHOLDER: set start position to inside the loading Paul trap (sets 1+2),
      -- between endcap_load_U (+z) and endcap_load_D (-z).  Coordinates are Fusion world (mm).
      { x_mm = 0, y_mm = 19, z_mm = -98.12, ke_ev = 0, -- -79.5, -98.12
        sigma_mm = { x = 0, y = 0, z = 0.1 } },
    },
  },

}
