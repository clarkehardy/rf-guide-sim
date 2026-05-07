-- trap_config.lua
-- Configuration for paulTrap.lua.  Loaded once per run via dofile().
-- Edit here rather than inside the Lua program.

return {

  -- ── Gas ──────────────────────────────────────────────────────────────────
  pressure_pa        = 0.1,    -- Pa  (100 Pa = 1 mbar)
  temperature_k      = 293,    -- K
  gas_molar_mass_amu = 28.0,   -- amu  (28 = N2)

  -- ── Particle ─────────────────────────────────────────────────────────────
  particle_radius_m     = 83e-9,  -- m  (166 nm diameter silica sphere)
  particle_density_kgm3 = 2200,   -- kg/m³  (fused silica)

  -- ── Drag / termination / recording ───────────────────────────────────────
  drag_scale   = 1.0,   -- multiply Epstein drag rate; 0 disables drag
  v_stop_mm_us = 1e-5,  -- terminate ion when speed < this [mm/us]; 0 to disable
  record_stride = 20,   -- write trajectory row every N time steps; 0 to disable

  -- ── Coordinate offsets: GEM → Fusion world (mm) ──────────────────────────
  -- Must match the locate(tx, ty, tz) block in paulTrap.gem.
  gem_offset = { x = 25.0, y = 8.0, z = 132.0 },

  -- ── Triggers ─────────────────────────────────────────────────────────────
  -- Each trigger holds its listed electrodes at 0 V until the ion's Fusion-Z
  -- coordinate first reaches z_mm, then releases them to follow the normal
  -- voltage schedule.  Triggers reset for each ion (sequential simulation).
  --
  -- Example — activate lens holders when ion enters the perpendicular trap:
  --   { z_mm = 200.0, electrodes = {11, 12} },
  triggers = {
    { z_mm = 276.0, electrodes = {9, 10, 11, 12} },
  },

  -- ── Particle definitions ──────────────────────────────────────────────────
  -- Positions are in Fusion world coordinates (mm); gem_offset is added automatically.
  -- charge:    elementary charges (e)
  -- ke_ev:     kinetic energy in eV
  -- az, el:    direction angles in degrees (az=0,el=0 → +Z; az=90,el=0 → +X; el=90 → +Y)
  -- n:         number of particles drawn from this start (default 1)
  -- sigma_mm:  Gaussian 1-σ spread per axis {x,y,z} in mm; omit for a point source
  particles = {
    charge = 100,
    starts = {
      { x_mm = 0, y_mm = 0, z_mm = -50, ke_ev = 0, n = 1,
        sigma_mm = { x = 0, y = 0, z = 0 } },
    },
  },

}
