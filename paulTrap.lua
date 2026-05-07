-- paulTrap.lua
-- User program for linear Paul trap RF guide + perpendicular retrapping trap.
-- Physics: RF trapping (fast adjust), Epstein drag (free molecular), gravity.
-- Gas, particle, drag, and trigger parameters are loaded from trap_config.lua.

simion.workbench_program()
simion.early_access(8.2)

local D = "C:\\users\\crossover\\Documents\\Research\\Nanospheres\\SIMION\\RF Guide\\"

-- ── Physical constants ────────────────────────────────────────────────────────
local kB  = 1.38065e-23   -- J/K
local amu = 1.66054e-27   -- kg per amu

-- ── Gravity ───────────────────────────────────────────────────────────────────
-- GEM Y increases upward, so gravity is −Y.  9.81 m/s² = 9.81e-9 mm/µs².
local g_simion = 9.81e-9  -- mm/µs²

-- ── RF defaults (overridden by voltages file metadata at run start) ───────────
local _rf_omega     = 2 * math.pi * 4900 * 1e-6  -- rad/µs  (fallback: 4900 Hz)
local _V0_default   = 100.0                        -- V, fallback amplitude main trap
local _rf_omega_2   = 2 * math.pi * 4900 * 1e-6  -- rad/µs  (fallback perp trap)
local _V0_2_default = 0.0                          -- V, fallback amplitude perp trap

-- ── Run-identity adjustables (flip in the SIMION Variables panel each run) ───
-- Everything else lives in trap_config.lua.
adjustable voltage_file_number = 1
adjustable run_number          = 1

-- ── State populated from trap_config.lua in initialize_run() ─────────────────
local gamma_drag     = 0.0              -- Epstein drag rate [µs⁻¹]
local _drag_scale    = 1.0
local _v_stop        = 1e-5             -- [mm/µs]
local _record_stride = 20
local _gem_off       = {x=25.0, y=8.0, z=132.0}  -- GEM → Fusion offsets [mm]
local _triggers      = {}              -- {z_mm, electrodes={...}} list
local _trigger_fire_time = {}          -- [i] = TOF when trigger i fired; nil = unfired

-- ── Voltage schedule tables (populated from CSV in initialize_run()) ──────────
local _vt, _v3, _v6, _v7, _v8, _v_rf = {}, {}, {}, {}, {}, {}
local _v11, _v12, _v13, _v_rf2, _v_dc2 = {}, {}, {}, {}, {}

local function _interp(t_tbl, v_tbl, t)
  if #t_tbl == 0 then return 0.0 end
  if t <= t_tbl[1] then return v_tbl[1] end
  if t >= t_tbl[#t_tbl] then return v_tbl[#t_tbl] end
  local lo, hi = 1, #t_tbl
  while hi - lo > 1 do
    local mid = math.floor((lo + hi) / 2)
    if t_tbl[mid] <= t then lo = mid else hi = mid end
  end
  local frac = (t - t_tbl[lo]) / (t_tbl[hi] - t_tbl[lo])
  return v_tbl[lo] + frac * (v_tbl[hi] - v_tbl[lo])
end

-- ── Trajectory file ───────────────────────────────────────────────────────────
local _traj_file = nil
local _traj_step = 0   -- per-ion step counter for stride


-- ─────────────────────────────────────────────────────────────────────────────
function segment.initialize_run()

  -- ── Load trap_config.lua ────────────────────────────────────────────────
  local ok, cfg = pcall(dofile, D .. "trap_config.lua")
  if not ok then
    simion.print("ERROR loading trap_config.lua: " .. tostring(cfg) .. "\n")
    simion.print("Falling back to built-in defaults.\n")
    cfg = {}
  end

  local T_gas  = cfg.temperature_k         or 293
  local P_gas  = cfg.pressure_pa           or 0.1
  local M_gas  = cfg.gas_molar_mass_amu    or 28.0
  local r_p    = cfg.particle_radius_m     or 83e-9
  local rho_p  = cfg.particle_density_kgm3 or 2200

  _drag_scale    = cfg.drag_scale    or 1.0
  _v_stop        = cfg.v_stop_mm_us  or 1e-5
  _record_stride = cfg.record_stride or 20
  _gem_off       = cfg.gem_offset    or {x=25.0, y=8.0, z=132.0}
  _triggers      = cfg.triggers      or {}

  -- Env vars SIMION_VOL_FILE / SIMION_RUN_NUM override the adjustable values.
  -- Set them in the shell (headless runs); leave unset to use the GUI panel.
  if os.getenv("SIMION_VOL_FILE") then
    voltage_file_number = tonumber(os.getenv("SIMION_VOL_FILE"))
  end
  if os.getenv("SIMION_RUN_NUM") then
    run_number = tonumber(os.getenv("SIMION_RUN_NUM"))
  end

  -- ── Compute Epstein drag rate ────────────────────────────────────────────
  local c_bar = math.sqrt(8 * kB * T_gas / (math.pi * M_gas * amu))
  local m_p   = (4/3) * math.pi * r_p^3 * rho_p
  local beta  = (8 * math.pi / 3) * r_p^2 * P_gas / c_bar
  gamma_drag  = beta / m_p * 1e-6   -- convert s⁻¹ → µs⁻¹

  simion.print(string.format(
    "Config:  P=%.3f Pa,  T=%.0f K,  M_gas=%.0f amu\n",
    P_gas, T_gas, M_gas))
  simion.print(string.format(
    "         r_p=%.0f nm,  rho_p=%.0f kg/m³,  m_p=%.3e kg\n",
    r_p*1e9, rho_p, m_p))
  simion.print(string.format(
    "         gamma=%.4e µs⁻¹,  drag_scale=%.1f\n",
    gamma_drag, _drag_scale))

  if #_triggers > 0 then
    for i, trig in ipairs(_triggers) do
      simion.print(string.format(
        "Trigger %d:  Z >= %.1f mm (Fusion)  →  electrodes {%s}\n",
        i, trig.z_mm, table.concat(trig.electrodes, ", ")))
    end
  end

  -- ── Define particles from config ─────────────────────────────────────────
  if cfg.particles then
    local p_cfg  = cfg.particles
    local charge = p_cfg.charge or 100
    local mass_amu = m_p / amu
    local F = simion.fly2
    local beams = {}
    for _, s in ipairs(p_cfg.starts or {}) do
      -- Convert az/el (degrees, SIMION convention) to unit vector for F.vector.
      -- az=0,el=0 → +Z; az=90,el=0 → +X; el=90 → +Y.
      local el_r = math.rad(s.el or 0)
      local az_r = math.rad(s.az or 0)
      local dx = math.cos(el_r) * math.sin(az_r)
      local dy = math.sin(el_r)
      local dz = math.cos(el_r) * math.cos(az_r)
      local ke = s.ke_ev or 0
      local beam_def = {
        n = 1, tob = 0,
        mass   = mass_amu,
        charge = charge,
        cwf = 1, color = 0,
        ke       = ke,
        position = F.vector(
          s.x_mm + _gem_off.x,
          s.y_mm + _gem_off.y,
          s.z_mm + _gem_off.z
        ),
      }
      if ke ~= 0 then
        beam_def.direction = F.cone_direction_distribution {
          axis = F.vector(dx, dy, dz), half_angle = 0, fill = true
        }
      end
      table.insert(beams, F.standard_beam(beam_def))
    end
    simion.experimental.add_particles { F.particles(beams) }
    simion.print(string.format(
      "Particles: %d defined from config  (charge=%de, mass=%.3e amu)\n",
      #beams, charge, mass_amu))
  end

  -- ── Clear and reload voltage schedule ───────────────────────────────────
  _vt, _v3, _v6, _v7, _v8, _v_rf = {}, {}, {}, {}, {}, {}
  _v11, _v12, _v13, _v_rf2, _v_dc2 = {}, {}, {}, {}, {}

  do
    local vpath = D .. "voltages_" .. math.floor(voltage_file_number) .. ".csv"
    local vf = io.open(vpath, "r")
    if vf then
      local line = vf:read("*l")
      local header_line
      while line and line:sub(1, 1) == "#" do
        local freq = line:match("f_RF_Hz=([%d%.eE%+%-]+)")
        if freq then
          _rf_omega = 2 * math.pi * tonumber(freq) * 1e-6
          simion.print("RF:   f = " .. freq .. " Hz\n")
        end
        local freq2 = line:match("f_RF2_Hz=([%d%.eE%+%-]+)")
        if freq2 then
          _rf_omega_2 = 2 * math.pi * tonumber(freq2) * 1e-6
          simion.print("RF2:  f = " .. freq2 .. " Hz\n")
        end
        line = vf:read("*l")
      end
      header_line = line

      local col_idx = {}
      if header_line then
        local i = 0
        for col in header_line:gmatch("[^,\r\n]+") do
          i = i + 1
          col_idx[col] = i
        end
      end

      local dest = {
        ["V_endcap"]     = _v3,
        ["V_endcap_R"]   = _v8,
        ["V_ring_L"]     = _v6,
        ["V_ring_R"]     = _v7,
        ["V_ring_brake"] = _v13,
        ["V_RF"]         = _v_rf,
        ["V_RF2"]        = _v_rf2,
        ["V_DC2"]        = _v_dc2,
        ["V_trap_lens"]  = _v11,
        ["V_coll_lens"]  = _v12,
      }

      for line in vf:lines() do
        local vals = {}
        local j = 0
        for v in line:gmatch("[^,\r\n]+") do
          j = j + 1
          vals[j] = tonumber(v)
        end
        local ti = col_idx["time_us"]
        if ti and vals[ti] then
          table.insert(_vt, vals[ti])
          for col, tbl in pairs(dest) do
            local idx = col_idx[col]
            if idx and vals[idx] then table.insert(tbl, vals[idx]) end
          end
        end
      end
      vf:close()
      simion.print("Loaded " .. #_vt .. " rows from " .. vpath .. "\n")
      local function _ch(name, tbl)
        if #tbl == 0 then
          simion.print("  " .. name .. ": NOT LOADED (column missing in CSV)\n")
        else
          simion.print(string.format("  %-14s %4d rows,  t=0: %+.1f V,  t_end: %+.1f V\n",
            name .. ":", #tbl, tbl[1], tbl[#tbl]))
        end
      end
      _ch("V_endcap",     _v3)
      _ch("V_endcap_R",   _v8)
      _ch("V_ring_L",     _v6)
      _ch("V_ring_R",     _v7)
      _ch("V_ring_brake", _v13)
      _ch("V_RF",         _v_rf)
      _ch("V_RF2",        _v_rf2)
      _ch("V_DC2",        _v_dc2)
      _ch("V_trap_lens",  _v11)
      _ch("V_coll_lens",  _v12)
    else
      simion.print("WARNING: voltage file not found: " .. vpath .. "\n")
    end
  end

  -- ── Open trajectory file ─────────────────────────────────────────────────
  _traj_file = nil
  if _record_stride > 0 then
    local path = D .. "trajectories_" .. math.floor(run_number) .. ".csv"
    _traj_file = io.open(path, "w")
    if _traj_file then
      _traj_file:write("ion,time_us,x_mm,y_mm,z_mm\n")
    else
      simion.print("WARNING: could not open trajectory file: " .. path .. "\n")
    end
  end
end


-- ─────────────────────────────────────────────────────────────────────────────
function segment.terminate_run()
  if _traj_file then _traj_file:close(); _traj_file = nil end
end


-- ─────────────────────────────────────────────────────────────────────────────
function segment.initialize()
  _traj_step = 0
  -- Reset trigger state for this ion (triggers are independent per ion).
  _trigger_fire_time = {}
end


-- ─────────────────────────────────────────────────────────────────────────────
-- segment.fast_adjust: set electrode voltages each time step.
-- ─────────────────────────────────────────────────────────────────────────────
function segment.fast_adjust()
  local t = ion_time_of_flight

  -- Main trap RF (electrodes 1, 2, 4, 5)
  local amp  = #_v_rf > 0 and _interp(_vt, _v_rf, t) or _V0_default
  local V_RF = amp * math.cos(_rf_omega * t)
  adj_elect[1] =  V_RF   -- rod pair 1, left   (+RF phase)
  adj_elect[2] = -V_RF   -- rod pair 2, left   (-RF phase)
  adj_elect[4] =  V_RF   -- rod pair 1, right  (+RF phase)
  adj_elect[5] = -V_RF   -- rod pair 2, right  (-RF phase)

  -- Main trap DC (electrodes 3, 6, 7, 8)
  if #_v3  > 0 then adj_elect[3]  = _interp(_vt, _v3,  t) end
  if #_v6  > 0 then adj_elect[6]  = _interp(_vt, _v6,  t) end
  if #_v7  > 0 then adj_elect[7]  = _interp(_vt, _v7,  t) end
  if #_v8  > 0 then adj_elect[8]  = _interp(_vt, _v8,  t) end

  -- Perpendicular trap RF + DC bias (electrodes 9, 10)
  local amp2  = #_v_rf2 > 0 and _interp(_vt, _v_rf2, t) or _V0_2_default
  local dc2   = #_v_dc2  > 0 and _interp(_vt, _v_dc2,  t) or 0.0
  local V_RF2 = amp2 * math.cos(_rf_omega_2 * t)
  adj_elect[9]  =  V_RF2 + dc2  -- trap rod pair 1, TL+BR
  adj_elect[10] = -V_RF2 + dc2  -- trap rod pair 2, TR+BL

  -- Perpendicular trap DC (electrodes 11, 12)
  if #_v11 > 0 then adj_elect[11] = _interp(_vt, _v11, t) end
  if #_v12 > 0 then adj_elect[12] = _interp(_vt, _v12, t) end

  -- Braking ring electrode (electrode 13)
  if #_v13 > 0 then adj_elect[13] = _interp(_vt, _v13, t) end

  -- Electrodes 14, 15 (glass lenses) are dielectric — not driven here.

  -- Trigger mask: electrodes belonging to unfired triggers are held at 0 V.
  -- Applied last so this overrides whatever the schedule set above.
  for i, trig in ipairs(_triggers) do
    if not _trigger_fire_time[i] then
      for _, en in ipairs(trig.electrodes) do
        adj_elect[en] = 0.0
      end
    end
  end
end


-- ─────────────────────────────────────────────────────────────────────────────
-- segment.accel_adjust: non-electric accelerations (Epstein drag + gravity).
-- Uses the finite-timestep correction factor to avoid underestimating drag
-- over long steps (see SIMION drag.lua example).
-- ─────────────────────────────────────────────────────────────────────────────
function segment.accel_adjust()
  if ion_time_step == 0 then return end

  local g     = _drag_scale * gamma_drag
  local tterm = ion_time_step * g

  if tterm < 1e-12 then
    -- Drag negligible: add gravity only
    ion_ay_mm = ion_ay_mm - g_simion
  else
    local factor = (1 - math.exp(-tterm)) / tterm
    ion_ax_mm = factor * (ion_ax_mm - g * ion_vx_mm)
    ion_ay_mm = factor * (ion_ay_mm - g * ion_vy_mm - g_simion)
    ion_az_mm = factor * (ion_az_mm - g * ion_vz_mm)
  end

  -- Trajectory recording (Fusion world coordinates)
  if _traj_file and _record_stride > 0 then
    _traj_step = _traj_step + 1
    if _traj_step % _record_stride == 0 then
      _traj_file:write(string.format("%d,%.4f,%.5f,%.5f,%.5f\n",
        ion_number,
        ion_time_of_flight,
        ion_px_mm - _gem_off.x,
        ion_py_mm - _gem_off.y,
        ion_pz_mm - _gem_off.z))
    end
  end
end


-- ─────────────────────────────────────────────────────────────────────────────
-- segment.other_actions: per-step logic that needs write access to ion_splat.
-- ─────────────────────────────────────────────────────────────────────────────
function segment.other_actions()
  -- Speed-based termination
  if _v_stop > 0 then
    local speed = math.sqrt(ion_vx_mm^2 + ion_vy_mm^2 + ion_vz_mm^2)
    if speed < _v_stop then ion_splat = 1 end
  end

  -- Trigger detection: fire when Fusion-Z first reaches the threshold.
  local z_fusion = ion_pz_mm - _gem_off.z
  for i, trig in ipairs(_triggers) do
    if not _trigger_fire_time[i] and z_fusion >= trig.z_mm then
      _trigger_fire_time[i] = ion_time_of_flight
      simion.print(string.format(
        "Trigger %d fired: ion %d at Z=%.2f mm, t=%.1f µs\n",
        i, ion_number, z_fusion, ion_time_of_flight))
    end
  end
end
