-- paulTrap.lua
-- User program for linear Paul trap RF guide + parallel-axis optical Paul trap.
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
local _rf_omega     = 2 * math.pi * 4900 * 1e-6  -- rad/µs  (fallback: main RF 4900 Hz)
local _V0_default   = 100.0                        -- V, fallback amplitude sets 1+2
local _rf_omega_3   = 2 * math.pi * 4900 * 1e-6  -- rad/µs  (fallback: optical Paul trap RF)
local _V0_3_default = 0.0                          -- V, fallback amplitude set 3

-- ── Run-identity adjustables (flip in the SIMION Variables panel each run) ───
-- Everything else lives in trap_config.lua.
adjustable voltage_file_number = 1
adjustable run_number          = 1

-- ── State populated from trap_config.lua in initialize_run() ─────────────────
local gamma_drag     = 0.0              -- Epstein drag rate at baseline pressure [µs⁻¹]
local _gamma_per_pa  = 0.0              -- drag rate per Pa of gas pressure [µs⁻¹ Pa⁻¹]
local _P_baseline    = 0.0              -- baseline gas pressure [Pa]
local _drag_scale    = 1.0
local _kT_over_m     = 0.0              -- kB*T/m_p in (m/s)², for Langevin noise amplitude
local _langevin_on   = true             -- thermal noise paired with Epstein drag (F-D theorem)
local _v_stop        = 1e-5             -- [mm/µs]
local _record_stride = 20
local _gem_off       = {x=25.0, y=8.0, z=132.0}  -- GEM → Fusion offsets [mm]
local _triggers           = {}         -- {z_mm, electrodes={...}} list
local _trig_for_electrode = {}         -- electrode_num → trigger index, built in initialize_run
-- Pressure-ramp config (set if cfg.pressure_ramp present)
local _ramp_enabled       = false
local _ramp_trigger_idx   = 1           -- which trigger fires the ramp
local _ramp_P_final       = 0.0         -- target pressure after ramp [Pa]
local _ramp_duration_us   = 0.0         -- linear ramp duration [µs]
-- Per-ion trigger state (keyed by ion_number so simultaneous ions don't share state)
local _ion_trig_fired     = {}         -- [ion_number][trig_idx] = true
local _ion_trig_fire_time = {}         -- [ion_number][trig_idx] = TOF at firing
local _particle_starts    = {}         -- [{x_mm,y_mm,z_mm,ke_ev,sigma_mm,...}] from config
local _particle_mass_amu  = 0.0        -- sphere mass in amu, written to ion_mass
local _particle_mass_kg   = 0.0        -- sphere mass in kg, for velocity ← KE conversion
local _particle_charge    = 100        -- elementary charges, written to ion_charge
local _particle_count     = 1          -- max ions to simulate; extras are splatted immediately
local _ion_traj_step      = {}         -- [ion_number] = per-ion accel_adjust call counter

-- ── Voltage schedule tables (populated from CSV in initialize_run()) ──────────
local _vt          = {}
local _v_rf        = {}   -- sets 1+2 RF amplitude envelope
local _v_rf3       = {}   -- set 3 RF amplitude envelope
local _v_ec_load_U = {}   -- electrode 3
local _v_ec_load_D = {}   -- electrode 4
local _v_dc_TL     = {}   -- electrode 5 DC trim (rod_3_TL)
local _v_dc_TR     = {}   -- electrode 6 DC trim (rod_3_TR)
local _v_dc_BL     = {}   -- electrode 7 DC trim (rod_3_BL)
local _v_dc_BR     = {}   -- electrode 8 DC trim (rod_3_BR)
local _v_ec_opt_U  = {}   -- electrode 9  (main schedule, coarse time axis)
local _v_ec_opt_D  = {}   -- electrode 10 (main schedule, coarse time axis)
-- Post-trigger fine-resolution schedule (independent time axis, time since trigger).
-- _v_trig[N] is the voltage array for triggered electrode N; built dynamically from
-- the triggers config in initialize_run so any electrode can have a trig schedule.
local _vt_trig = {}
local _v_trig  = {}

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

-- CSV tokenizer that correctly handles empty fields (consecutive commas).
-- tonumber("") returns nil, which is falsy, so callers treat nil as "not present".
local function _split_csv(line)
  local vals = {}
  local i = 1
  for field in (line .. ","):gmatch("([^,\r\n]*),") do
    vals[i] = tonumber(field)
    i = i + 1
  end
  return vals
end

-- Standard-normal sample via Box-Muller, using SIMION's built-in rand().
local function _randn()
  local u1
  repeat u1 = rand() until u1 > 1e-15
  return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * rand())
end

-- Returns the effective schedule time for electrode en for the current ion.
-- Not triggered → t.  Trigger fired → t − t_fire.  Not yet fired → false.
-- Uses ion_number so simultaneous ions each see their own independent trigger state.
local function _trig_t(en, t)
  local i = _trig_for_electrode[en]
  if not i then return t end
  local ft = _ion_trig_fired[ion_number]
  if not ft or not ft[i] then return false end
  return t - _ion_trig_fire_time[ion_number][i]
end

-- Returns the current gas pressure for the current ion, in Pa.
-- Before the ramp trigger fires (or if no ramp configured) → _P_baseline.
-- During ramp → linear interpolation baseline → P_final over duration.
-- After ramp → _ramp_P_final.
local function _current_pressure()
  if not _ramp_enabled then return _P_baseline end
  local fired = _ion_trig_fired[ion_number]
  if not fired or not fired[_ramp_trigger_idx] then return _P_baseline end
  local t_since = ion_time_of_flight - _ion_trig_fire_time[ion_number][_ramp_trigger_idx]
  if t_since <= 0 then return _P_baseline end
  if _ramp_duration_us <= 0 or t_since >= _ramp_duration_us then return _ramp_P_final end
  return _P_baseline + (t_since / _ramp_duration_us) * (_ramp_P_final - _P_baseline)
end

-- Returns the voltage for DC electrode `en` at absolute time `t_abs`.
-- Handles three cases uniformly:
--   • Not triggered (_trig_for_electrode[en] nil): main schedule, absolute t.
--   • Triggered, not yet fired: 0 V (gated).
--   • Triggered, fired: post-trigger schedule (V_e{N}_trig column) if loaded,
--     else main schedule, both using time-since-fire as the lookup key.
local function _volt_dc(en, t_abs, v_main)
  if not _trig_for_electrode[en] then
    return #v_main > 0 and _interp(_vt, v_main, t_abs) or 0
  end
  local t_rel = _trig_t(en, t_abs)
  if not t_rel then return 0 end
  local v_pt = _v_trig[en]
  if v_pt and #v_pt > 0 then return _interp(_vt_trig, v_pt, t_rel) end
  return #v_main > 0 and _interp(_vt, v_main, t_rel) or 0
end

-- ── Trajectory file ───────────────────────────────────────────────────────────
local _traj_file = nil


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
  _trig_for_electrode = {}
  for i, trig in ipairs(_triggers) do
    for _, en in ipairs(trig.electrodes) do _trig_for_electrode[en] = i end
  end

  -- Env vars SIMION_VOL_FILE / SIMION_RUN_NUM override the adjustable values.
  -- Set them in the shell (headless runs); leave unset to use the GUI panel.
  if os.getenv("SIMION_VOL_FILE") then
    voltage_file_number = tonumber(os.getenv("SIMION_VOL_FILE"))
  end
  if os.getenv("SIMION_RUN_NUM") then
    run_number = tonumber(os.getenv("SIMION_RUN_NUM"))
  end

  -- ── Compute Epstein drag rate ────────────────────────────────────────────
  -- γ = β/m = [(8π/3) r² P / c̄] / m  (in s⁻¹).  Linear in P, so precompute
  -- gamma_per_pa = γ / P; the actual γ at each timestep is gamma_per_pa * P_now.
  local c_bar = math.sqrt(8 * kB * T_gas / (math.pi * M_gas * amu))
  local m_p   = (4/3) * math.pi * r_p^3 * rho_p
  _gamma_per_pa = (8 * math.pi / 3) * r_p^2 / (m_p * c_bar) * 1e-6  -- µs⁻¹ Pa⁻¹
  _P_baseline   = P_gas
  gamma_drag    = _gamma_per_pa * P_gas   -- baseline rate for logging
  _kT_over_m    = kB * T_gas / m_p        -- (m/s)², fixed for the run (T doesn't ramp)
  _langevin_on  = cfg.langevin_noise ~= false  -- default true; set false to disable noise

  simion.print(string.format(
    "Config:  P=%.3f Pa,  T=%.0f K,  M_gas=%.0f amu\n",
    P_gas, T_gas, M_gas))
  simion.print(string.format(
    "         r_p=%.0f nm,  rho_p=%.0f kg/m^3,  m_p=%.3e kg\n",
    r_p*1e9, rho_p, m_p))
  simion.print(string.format(
    "         gamma=%.4e us^-1 (at baseline P),  drag_scale=%.1f\n",
    gamma_drag, _drag_scale))
  simion.print(string.format(
    "         Langevin noise: %s  (v_rms_1D = %.2e mm/us at baseline P)\n",
    _langevin_on and "on" or "off",
    math.sqrt(_kT_over_m) * 1e-3))

  -- ── Pressure ramp (triggered) ────────────────────────────────────────────
  local pr = cfg.pressure_ramp
  if pr then
    _ramp_enabled     = true
    _ramp_trigger_idx = pr.trigger     or 1
    _ramp_P_final     = pr.P_final_pa  or P_gas
    _ramp_duration_us = pr.duration_us or 0
    simion.print(string.format(
      "Pressure ramp: on trigger %d, P=%.3f Pa --> %.3f Pa over %.0f us\n",
      _ramp_trigger_idx, _P_baseline, _ramp_P_final, _ramp_duration_us))
  else
    _ramp_enabled = false
  end

  if #_triggers > 0 then
    for i, trig in ipairs(_triggers) do
      simion.print(string.format(
        "Trigger %d:  Z >= %.1f mm (Fusion)  -->  electrodes {%s}\n",
        i, trig.z_mm, table.concat(trig.electrodes, ", ")))
    end
  end

  -- ── Load particle parameters from config ─────────────────────────────────
  -- Position/velocity/mass/charge applied per-ion in segment.initialize().
  -- Ion count: workbench must have >= n ions; extras are splatted immediately.
  local pcfg         = cfg.particles or {}
  _particle_starts   = pcfg.starts  or {}
  _particle_mass_amu = m_p / amu
  _particle_mass_kg  = m_p
  _particle_charge   = pcfg.charge  or 100
  _particle_count    = pcfg.n       or 1
  simion.print(string.format(
    "Particles: n=%d, charge=%de, mass=%.3e amu,  %d start entry/entries\n",
    _particle_count, _particle_charge, _particle_mass_amu, #_particle_starts))

  -- ── Clear and reload voltage schedule ───────────────────────────────────
  _vt          = {}
  _v_rf        = {}
  _v_rf3       = {}
  _v_ec_load_U = {}
  _v_ec_load_D = {}
  _v_dc_TL     = {}
  _v_dc_TR     = {}
  _v_dc_BL     = {}
  _v_dc_BR     = {}
  _v_ec_opt_U  = {}
  _v_ec_opt_D  = {}
  _vt_trig = {}
  _v_trig  = {}

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
        local freq3 = line:match("f_RF3_Hz=([%d%.eE%+%-]+)")
        if freq3 then
          _rf_omega_3 = 2 * math.pi * tonumber(freq3) * 1e-6
          simion.print("RF3:  f = " .. freq3 .. " Hz\n")
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
        ["V_RF"]               = _v_rf,
        ["V_RF3"]              = _v_rf3,
        ["V_endcap_load_U"]    = _v_ec_load_U,
        ["V_endcap_load_D"]    = _v_ec_load_D,
        ["V_dc_3_TL"]          = _v_dc_TL,
        ["V_dc_3_TR"]          = _v_dc_TR,
        ["V_dc_3_BL"]          = _v_dc_BL,
        ["V_dc_3_BR"]          = _v_dc_BR,
        ["V_endcap_optical_U"] = _v_ec_opt_U,
        ["V_endcap_optical_D"] = _v_ec_opt_D,
      }
      -- Build post-trigger dest from the triggers config.
      -- Column name for electrode N is V_e{N}_trig.
      local trig_dest = {}
      for _, trig in ipairs(_triggers) do
        for _, en in ipairs(trig.electrodes) do
          if not _v_trig[en] then
            _v_trig[en] = {}
            trig_dest["V_e" .. en .. "_trig"] = _v_trig[en]
          end
        end
      end

      local ti  = col_idx["time_us"]
      local tti = col_idx["time_trig_us"]
      for line in vf:lines() do
        local vals = _split_csv(line)
        if ti and vals[ti] then
          table.insert(_vt, vals[ti])
          for col, tbl in pairs(dest) do
            local idx = col_idx[col]
            if idx and vals[idx] then table.insert(tbl, vals[idx]) end
          end
        end
        if tti and vals[tti] then
          table.insert(_vt_trig, vals[tti])
          for col, tbl in pairs(trig_dest) do
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
      _ch("V_RF",               _v_rf)
      _ch("V_RF3",              _v_rf3)
      _ch("V_endcap_load_U",    _v_ec_load_U)
      _ch("V_endcap_load_D",    _v_ec_load_D)
      _ch("V_dc_3_TL",          _v_dc_TL)
      _ch("V_dc_3_TR",          _v_dc_TR)
      _ch("V_dc_3_BL",          _v_dc_BL)
      _ch("V_dc_3_BR",          _v_dc_BR)
      _ch("V_endcap_optical_U", _v_ec_opt_U)
      _ch("V_endcap_optical_D", _v_ec_opt_D)
      if #_vt_trig > 0 then
        simion.print(string.format(
          "  Trigger schedule: %d rows,  t=0: %.4f us,  t_end: %.4f us\n",
          #_vt_trig, _vt_trig[1], _vt_trig[#_vt_trig]))
        for _, trig in ipairs(_triggers) do
          for _, en in ipairs(trig.electrodes) do
            local vt = _v_trig[en]
            if vt then _ch(string.format("V_e%d_trig", en), vt) end
          end
        end
      else
        simion.print("  Trigger schedule: not found (no V_e{N}_trig columns in CSV)\n")
      end
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
  _ion_traj_step[ion_number] = 0
  -- Allocate per-ion trigger tables (separate state for simultaneous ions).
  _ion_trig_fired[ion_number]     = {}
  _ion_trig_fire_time[ion_number] = {}

  -- Splat any ions beyond the configured count immediately (no flight).
  if ion_number > _particle_count then
    ion_splat = 1
    return
  end

  -- Set mass and charge from config, overriding workbench values.
  ion_mass   = _particle_mass_amu
  ion_charge = _particle_charge

  -- Override start position and velocity; cycles round-robin through starts entries.
  if #_particle_starts > 0 then
    local s   = _particle_starts[((ion_number - 1) % #_particle_starts) + 1]
    local sig = s.sigma_mm
    ion_px_mm = _gem_off.x + s.x_mm + (sig and sig.x or 0) * _randn()
    ion_py_mm = _gem_off.y + s.y_mm + (sig and sig.y or 0) * _randn()
    ion_pz_mm = _gem_off.z + s.z_mm + (sig and sig.z or 0) * _randn()
    local ke = s.ke_ev or 0
    if ke == 0 then
      ion_vx_mm = 0; ion_vy_mm = 0; ion_vz_mm = 0
    else
      local v    = math.sqrt(2 * ke * 1.602e-19 / _particle_mass_kg) * 1e-3
      local el_r = math.rad(s.el or 0)
      local az_r = math.rad(s.az or 0)
      ion_vx_mm = v * math.cos(el_r) * math.sin(az_r)
      ion_vy_mm = v * math.sin(el_r)
      ion_vz_mm = v * math.cos(el_r) * math.cos(az_r)
    end
  end
end


-- ─────────────────────────────────────────────────────────────────────────────
-- segment.fast_adjust: set electrode voltages each time step.
-- ─────────────────────────────────────────────────────────────────────────────
function segment.fast_adjust()
  local t = ion_time_of_flight

  -- Sets 1+2 RF (electrodes 1, 2) — always on, absolute TOF
  local amp  = #_v_rf > 0 and _interp(_vt, _v_rf, t) or _V0_default
  local V_RF = amp * math.cos(_rf_omega * t)
  adj_elect[1] =  V_RF   -- sets 1+2: TL + BR rods (+RF phase)
  adj_elect[2] = -V_RF   -- sets 1+2: TR + BL rods (-RF phase)

  -- Load endcaps (electrodes 3, 4)
  adj_elect[3] = _volt_dc(3, t, _v_ec_load_U)
  adj_elect[4] = _volt_dc(4, t, _v_ec_load_D)

  -- Set 3 (electrodes 5–8): shared V_RF3 with diagonal-pair phasing + per-rod DC trim.
  -- Always on, absolute TOF for both carrier and envelope.
  -- TL + BR get +RF phase; TR + BL get -RF phase.
  local amp3  = #_v_rf3 > 0 and _interp(_vt, _v_rf3, t) or _V0_3_default
  local V_RF3 = amp3 * math.cos(_rf_omega_3 * t)
  local dcTL  = #_v_dc_TL > 0 and _interp(_vt, _v_dc_TL, t) or 0
  local dcTR  = #_v_dc_TR > 0 and _interp(_vt, _v_dc_TR, t) or 0
  local dcBL  = #_v_dc_BL > 0 and _interp(_vt, _v_dc_BL, t) or 0
  local dcBR  = #_v_dc_BR > 0 and _interp(_vt, _v_dc_BR, t) or 0
  adj_elect[5] =  V_RF3 + dcTL   -- rod_3_TL
  adj_elect[6] = -V_RF3 + dcTR   -- rod_3_TR
  adj_elect[7] = -V_RF3 + dcBL   -- rod_3_BL
  adj_elect[8] =  V_RF3 + dcBR   -- rod_3_BR

  -- Optical endcaps (electrodes 9, 10)
  adj_elect[9]  = _volt_dc(9,  t, _v_ec_opt_U)
  adj_elect[10] = _volt_dc(10, t, _v_ec_opt_D)

  -- Dielectric volumes (trapping_lens, collection_lens, lens_holder) are not driven here.
end


-- ─────────────────────────────────────────────────────────────────────────────
-- segment.accel_adjust: non-electric accelerations (Epstein drag + gravity).
-- Uses the finite-timestep correction factor to avoid underestimating drag
-- over long steps (see SIMION drag.lua example).
-- ─────────────────────────────────────────────────────────────────────────────
function segment.accel_adjust()
  if ion_time_step == 0 then return end

  -- Drag rate at the current per-ion gas pressure (pressure may be ramping
  -- after a trigger fire; falls back to baseline when no ramp is configured).
  local g     = _drag_scale * _gamma_per_pa * _current_pressure()
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
    _ion_traj_step[ion_number] = (_ion_traj_step[ion_number] or 0) + 1
    if _ion_traj_step[ion_number] % _record_stride == 0 then
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
  -- Terminate ions beyond the configured count.
  -- ion_splat must be set here (other_actions) to actually take effect.
  if ion_number > _particle_count then
    ion_splat = 1
    return
  end

  -- Langevin thermal noise: stochastic velocity kick, one per time step.
  -- σ² = (kB·T/m)·(1 − exp(−2·γ·dt)) is the exact OU fluctuation-dissipation
  -- complement to the deterministic drag applied in accel_adjust.
  if _langevin_on and _drag_scale > 0 and ion_time_step > 0 then
    local g  = _drag_scale * _gamma_per_pa * _current_pressure()
    local dt = ion_time_step
    if g * dt > 1e-12 then
      local sigma = math.sqrt(_kT_over_m * (1 - math.exp(-2 * g * dt))) * 1e-3
      ion_vx_mm = ion_vx_mm + sigma * _randn()
      ion_vy_mm = ion_vy_mm + sigma * _randn()
      ion_vz_mm = ion_vz_mm + sigma * _randn()
    end
  end

  -- Speed-based termination
  if _v_stop > 0 then
    local speed = math.sqrt(ion_vx_mm^2 + ion_vy_mm^2 + ion_vz_mm^2)
    if speed < _v_stop then ion_splat = 1 end
  end

  -- Trigger detection: fire when Fusion-Z first reaches the threshold.
  local z_fusion = ion_pz_mm - _gem_off.z
  local fired    = _ion_trig_fired[ion_number]
  for i, trig in ipairs(_triggers) do
    if not fired[i] and z_fusion >= trig.z_mm then
      fired[i]                           = true
      _ion_trig_fire_time[ion_number][i] = ion_time_of_flight
      simion.print(string.format(
        "Trigger %d fired: ion %d at Z=%.2f mm, t=%.1f us\n",
        i, ion_number, z_fusion, ion_time_of_flight))
    end
  end
end
