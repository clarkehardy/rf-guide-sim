-- refine_with_dielectric.lua
--
-- Re-refines each electrode fast-adjust PA (pa1–pa10) with the dielectric
-- permittivity array so that the lenses and lens holder correctly modify
-- the field.
--
-- Run AFTER:
--   1. Normal GEM Refine in SIMION (creates pa0–pa10 without dielectrics).
--   2. generate_dielectric_pa.py (creates paulTrap-dielectric.pa).
--
-- Run this script from within SIMION via:
--   File → Run Lua Script → refine_with_dielectric.lua
-- or from the SIMION command line:
--   simion --nogui lua refine_with_dielectric.lua
--
-- When finished, the fast-adjust PAs incorporate dielectric effects and
-- SIMION can be used as normal (Fly'm, etc.).

local D = "C:\\users\\crossover\\Documents\\Research\\Nanospheres\\SIMION\\RF Guide\\"

local N_ELECTRODES = 10   -- electrodes 1–10 (dielectrics are not in the electric PA)
local CONVERGENCE  = 1e-7 -- refine convergence criterion

-- ── Open the dielectric permittivity array ────────────────────────────────────
local di_path = D .. "paulTrap-dielectric.pa"
print("Opening dielectric PA: " .. di_path)
local di = simion.pas:open(di_path)
if not di then
  error("Could not open dielectric PA.  Run generate_dielectric_pa.py first.")
end

-- ── Re-refine each electrode PA ───────────────────────────────────────────────
for e = 1, N_ELECTRODES do
  local pa_path = D .. string.format("paulTrap.pa%d", e)
  local fh = io.open(pa_path, "rb")
  if fh then
    fh:close()
    print(string.format("  Refining pa%d ...", e))
    local pa = simion.pas:open(pa_path)
    pa:refine { convergence = CONVERGENCE, permittivity = di }
    pa:save()
    print(string.format("  pa%d done.", e))
  else
    print(string.format("  pa%d not found — skipping.", e))
  end
end

print("All electrode PAs re-refined with dielectric.  Fast-adjust is ready.")
