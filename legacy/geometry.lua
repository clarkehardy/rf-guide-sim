-- geometry.lua
-- Potential array geometry for linear Paul trap
-- Array: 2548 x 94 x 94, grid spacing 0.15 mm
-- Coordinate mapping: SIMION xi -> Fusion Z (trap axis)
--                     SIMION yi -> Fusion X (horizontal)
--                     SIMION zi -> Fusion Y (vertical)

local stl = require "simion.stl"

local GRID = 0.15   -- mm per grid step

-- PA corner in Fusion world coordinates (mm)
local Z0 = -131.0   -- Fusion Z at xi=0
local X0 =   -7.0   -- Fusion X at yi=0
local Y0 =   12.05  -- Fusion Y at zi=0

-- Load electrode meshes
-- Electrode 1: rod pair 1 (+RF)
local p1 = {}
for _, f in ipairs({
    "rod_P1_L1a.stl", "rod_P1_L1b.stl",
    "rod_P1_L2a.stl", "rod_P1_L2b.stl",
    "rod_P1_R1.stl",  "rod_P1_R2.stl",
}) do p1[#p1+1] = stl.open(f) end

-- Electrode 2: rod pair 2 (-RF)
local p2 = {}
for _, f in ipairs({
    "rod_P2_L1a.stl", "rod_P2_L1b.stl",
    "rod_P2_L2a.stl", "rod_P2_L2b.stl",
    "rod_P2_R1.stl",  "rod_P2_R2.stl",
}) do p2[#p2+1] = stl.open(f) end

-- Electrode 3: left end cap
local ec_L = stl.open("endcap_L.stl")

-- Called for each grid point. Returns electrode number (1-based) or 0.
-- End cap checked first; rods checked after so they take priority at overlaps.
function pa_electrode(xi, yi, zi)
    local fus_z = Z0 + xi * GRID
    local fus_x = X0 + yi * GRID
    local fus_y = Y0 + zi * GRID

    if ec_L:inside(fus_x, fus_y, fus_z) then return 3 end

    for _, m in ipairs(p1) do
        if m:inside(fus_x, fus_y, fus_z) then return 1 end
    end
    for _, m in ipairs(p2) do
        if m:inside(fus_x, fus_y, fus_z) then return 2 end
    end

    return 0
end