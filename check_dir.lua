-- check_dir.lua
-- Run via: Settings and Utilities > Run Lua Program
-- Reports SIMION's working directory and tests file visibility.

local function try(f) local ok, v = pcall(f); return ok and v or nil end

-- Get working directory via Windows 'cd' command
local cwd = try(function()
    local p = io.popen("cd"); local s = p:read("*l"); p:close(); return s
end) or "(io.popen unavailable)"

-- Check whether the STL files are visible from the current directory
local gem_visible = io.open("paulTrap.gem",  "r") and "YES" or "NO"
local stl_visible = io.open("rod_P1_L1a.stl","r") and "YES" or "NO"

-- Check some candidate absolute paths for the project folder
local candidates = {
    "Z:\\Users\\clarke\\Documents\\Research\\Nanospheres\\SIMION\\RF Guide\\rod_P1_L1a.stl",
    "C:\\Users\\crossover\\Documents\\Research\\Nanospheres\\SIMION\\RF Guide\\rod_P1_L1a.stl",
}
local found = "(none worked)"
for _, p in ipairs(candidates) do
    if io.open(p, "r") then found = p; break end
end

print("=== SIMION working directory ===")
print("CWD: " .. cwd)
print("paulTrap.gem visible from CWD: " .. gem_visible)
print("rod_P1_L1a.stl visible from CWD: " .. stl_visible)
print("Absolute path that worked: " .. found)
