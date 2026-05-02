"""
sanity_check.py -- GEM/STL geometry verification for paulTrap.gem
Checks:
  1. Bounding boxes of each STL in Fusion world coordinates
  2. Whether each GEM seed point is inside its STL body (ray casting)
  3. Whether all geometry falls within the PA bounds
  4. Rod radii and transverse axis positions
  5. End cap seed point plausibility
"""

import struct, math, os

BASE = os.path.dirname(os.path.abspath(__file__))

# PA bounds in Fusion world coordinates (derived from GEM locate + pa_define)
# pa_define: 14x14x382 mm at dx=0.15, with locate(7, -12.05, 131)
# GEM x = Fusion X + 7   →  Fusion X ∈ [-7, 7]
# GEM y = Fusion Y - 12.05 → Fusion Y ∈ [12.05, 26.05]
# GEM z = Fusion Z + 131  →  Fusion Z ∈ [-131, 251]
PA_X = (-7.0, 7.0)
PA_Y = (12.05, 26.05)
PA_Z = (-131.0, 251.0)

# Seed points from GEM file (Fusion world coords, i.e. values passed to stl() before locate())
SEEDS = {
    "rod_P1_L1a": ((-2.1082, 21.1582, -87.4141), 1),
    "rod_P1_L1b": ((-2.1082, 21.1582,   8.5471), 1),
    "rod_P1_L2a": (( 2.1082, 16.9672, -87.4141), 1),
    "rod_P1_L2b": (( 2.1082, 16.9672,   8.5471), 1),
    "rod_P1_R1":  ((-2.1082, 21.1582, 169.2021), 1),
    "rod_P1_R2":  (( 2.1082, 16.9672, 169.2021), 1),
    "rod_P2_L1a": (( 2.1082, 21.1582, -87.4141), 2),
    "rod_P2_L1b": (( 2.1082, 21.1582,   8.5471), 2),
    "rod_P2_L2a": ((-2.1082, 16.9672, -87.4141), 2),
    "rod_P2_L2b": ((-2.1082, 16.9672,   8.5471), 2),
    "rod_P2_R1":  (( 2.1082, 21.1582, 169.2021), 2),
    "rod_P2_R2":  ((-2.1082, 16.9672, 169.2021), 2),
    "endcap_L":   (( 0.0000, 19.0500, 109.2200), 3),
}

# ── STL parsing ──────────────────────────────────────────────────────────────

def read_stl_binary(path):
    """Return list of triangles as ((v0,v1,v2), normal) where each vertex is (x,y,z)."""
    triangles = []
    with open(path, "rb") as f:
        f.read(80)  # header
        n, = struct.unpack("<I", f.read(4))
        for _ in range(n):
            nx, ny, nz = struct.unpack("<fff", f.read(12))
            v0 = struct.unpack("<fff", f.read(12))
            v1 = struct.unpack("<fff", f.read(12))
            v2 = struct.unpack("<fff", f.read(12))
            f.read(2)  # attribute byte count
            triangles.append(((v0, v1, v2), (nx, ny, nz)))
    return triangles

def bbox(triangles):
    xs = [v[0] for tri, _ in triangles for v in tri]
    ys = [v[1] for tri, _ in triangles for v in tri]
    zs = [v[2] for tri, _ in triangles for v in tri]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))

# ── Ray-triangle intersection (Möller-Trumbore) ──────────────────────────────

EPS = 1e-9

def ray_triangle_intersect(orig, direction, v0, v1, v2):
    """Return t (distance along ray) or None if no intersection."""
    e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
    e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
    dx, dy, dz = direction
    h = (dy*e2[2]-dz*e2[1], dz*e2[0]-dx*e2[2], dx*e2[1]-dy*e2[0])
    a = e1[0]*h[0] + e1[1]*h[1] + e1[2]*h[2]
    if abs(a) < EPS:
        return None
    f = 1.0 / a
    s = (orig[0]-v0[0], orig[1]-v0[1], orig[2]-v0[2])
    u = f * (s[0]*h[0] + s[1]*h[1] + s[2]*h[2])
    if u < 0.0 or u > 1.0:
        return None
    q = (s[1]*e1[2]-s[2]*e1[1], s[2]*e1[0]-s[0]*e1[2], s[0]*e1[1]-s[1]*e1[0])
    v = f * (dx*q[0] + dy*q[1] + dz*q[2])
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * (e2[0]*q[0] + e2[1]*q[1] + e2[2]*q[2])
    return t if t > EPS else None

def point_in_mesh(point, triangles, ray=(1.0, 0.0, 0.0)):
    """Return True if point is inside a closed mesh (parity test, three ray directions)."""
    counts = []
    for direction in [(1,0,0), (0,1,0), (0,0,1)]:
        count = 0
        for (v0, v1, v2), _ in triangles:
            t = ray_triangle_intersect(point, direction, v0, v1, v2)
            if t is not None:
                count += 1
        counts.append(count % 2 == 1)  # odd = inside
    return sum(counts) >= 2  # majority vote across 3 ray directions

# ── Analysis ─────────────────────────────────────────────────────────────────

def fmt_range(lo, hi):
    return f"[{lo:8.3f}, {hi:8.3f}]"

print("=" * 70)
print("SIMION Paul Trap — STL Sanity Checks")
print("=" * 70)

all_ok = True
rod_axes = []  # (name, x_center, y_center) in Fusion world coords

for name, (seed, elec) in sorted(SEEDS.items()):
    path = os.path.join(BASE, name + ".stl")
    if not os.path.exists(path):
        print(f"\n[MISSING] {name}.stl")
        all_ok = False
        continue

    tris = read_stl_binary(path)
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bbox(tris)
    cx, cy, cz = (xlo+xhi)/2, (ylo+yhi)/2, (zlo+zhi)/2
    rx, ry = (xhi-xlo)/2, (yhi-ylo)/2

    inside = point_in_mesh(seed, tris)
    sx, sy, sz = seed

    # PA bounds check
    in_pa = (PA_X[0] <= xlo and xhi <= PA_X[1] and
             PA_Y[0] <= ylo and yhi <= PA_Y[1] and
             PA_Z[0] <= zlo and zhi <= PA_Z[1])

    seed_in_pa = (PA_X[0] <= sx <= PA_X[1] and
                  PA_Y[0] <= sy <= PA_Y[1] and
                  PA_Z[0] <= sz <= PA_Z[1])

    status_inside = "OK  " if inside else "FAIL"
    status_pa     = "OK  " if in_pa  else "WARN"
    status_seed   = "OK  " if seed_in_pa else "WARN"

    print(f"\n── {name}  (electrode {elec}) ─────────────────────────────")
    print(f"  Bounding box X: {fmt_range(xlo, xhi)}  center={cx:7.3f}  half={rx:.3f}")
    print(f"  Bounding box Y: {fmt_range(ylo, yhi)}  center={cy:7.3f}  half={ry:.3f}")
    print(f"  Bounding box Z: {fmt_range(zlo, zhi)}  center={cz:7.3f}")
    print(f"  Seed point    : ({sx:.4f}, {sy:.4f}, {sz:.4f})")
    print(f"  Seed inside mesh : [{status_inside}]")
    print(f"  Mesh within PA   : [{status_pa}]")
    print(f"  Seed within PA   : [{status_seed}]")

    if not inside:
        all_ok = False
    if not in_pa:
        all_ok = False

    # Collect rod cross-section info (not end cap)
    if "rod" in name:
        rod_axes.append((name, cx, cy, (xhi-xlo)/2, (yhi-ylo)/2))

# ── Rod geometry summary ─────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Rod cross-section summary (Fusion world X/Y, from bbox center)")
print(f"  Expected rod radius   : ~1.5 mm")
print(f"  Expected |X| from axis: ~2.1 mm   (rod center-to-center = 6 mm → r0=3 mm,")
print(f"                                      rod center at ~r0 ± r_rod)")
print(f"  Expected Y centers    : top row ~21.2 mm, bottom row ~17.0 mm")
print()
print(f"  {'Name':<18} {'X_center':>9} {'Y_center':>9} {'r_x':>6} {'r_y':>6}")
for name, cx, cy, rx, ry in sorted(rod_axes):
    flag = ""
    if abs(rx - 1.5) > 0.3 or abs(ry - 1.5) > 0.3:
        flag = "  ← radius mismatch?"
    print(f"  {name:<18} {cx:9.4f} {cy:9.4f} {rx:6.3f} {ry:6.3f}{flag}")

# ── End cap specific check ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("End cap seed point check")
endcap_path = os.path.join(BASE, "endcap_L.stl")
if os.path.exists(endcap_path):
    tris = read_stl_binary(endcap_path)
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bbox(tris)
    seed = SEEDS["endcap_L"][0]
    print(f"  End cap bbox Z: [{zlo:.3f}, {zhi:.3f}]  (center = {(zlo+zhi)/2:.3f})")
    print(f"  GEM seed Z    : {seed[2]:.3f}")
    print(f"  Known Fusion Z: -116.002 mm (center of end cap from assembly)")
    if seed[2] > zhi or seed[2] < zlo:
        print(f"  [FAIL] Seed Z={seed[2]:.3f} is OUTSIDE the end cap bbox!")
        print(f"         The seed point is almost certainly wrong.")
        print(f"         Correct seed should be near Z≈{(zlo+zhi)/2:.3f} mm (bbox center)")
    else:
        print(f"  [OK] Seed Z is within bbox.")

print("\n" + "=" * 70)
print("Overall: " + ("All checks passed." if all_ok else "One or more issues found — see above."))
