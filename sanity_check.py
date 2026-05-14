"""
sanity_check.py  –  GEM/STL geometry verification for paulTrap.gem

Checks for every STL referenced in the GEM (and for the dielectric volumes):
  1. Bounding box in Fusion world coordinates
  2. Whether the GEM seed point lies inside the mesh (ray-casting parity test)
  3. Whether the mesh falls within the PA bounds
  4. Per-group geometry summaries: rod cross-sections by set, endcap Z positions,
     dielectric body positions

Electrodes checked
  1  rod_1_TL, rod_1_BR, rod_2_TL, rod_2_BR     sets 1+2 +RF
  2  rod_1_TR, rod_1_BL, rod_2_TR, rod_2_BL     sets 1+2 −RF
  3  endcap_load_U                              load Paul trap +z endcap
  4  endcap_load_D                              load Paul trap -z endcap
  5  rod_3_TL                                   optical Paul trap +RF3
  6  rod_3_TR                                   optical Paul trap −RF3
  7  rod_3_BL                                   optical Paul trap −RF3
  8  rod_3_BR                                   optical Paul trap +RF3
  9  endcap_optical_U                           optical Paul trap +z endcap
 10  endcap_optical_D                           optical Paul trap -z endcap
  –  trapping_lens                              dielectric (not in electric PA)
  –  collection_lens                            dielectric (not in electric PA)
  –  lens_holder                                dielectric (not in electric PA)
"""

import struct, os

BASE = os.path.dirname(os.path.abspath(__file__))

# ── PA bounds in Fusion world coordinates ─────────────────────────────────────
# PLACEHOLDER: update to match new pa_define + locate() in paulTrap.gem.
PA_X = (-25.0,  40.0)
PA_Y = ( -8.0,  37.0)
PA_Z = (-132.0, 295.0)

# ── Electrodes: (stl_stem, seed_xyz, electrode_number, short_description) ─────
# Seed points are Fusion world coordinates, exactly as written in paulTrap.gem.
# Electrode 0 = not in the electric PA (dielectric only).
# PLACEHOLDER: replace every seed coordinate with the actual Fusion centroid
# from the re-exported STL.  Use bbox-centre as a reliable default.
ELECTRODES = [
    # Sets 1 + 2 (loading + RF guide, wired in parallel)
    ("rod_1_TL",         (0.0, 0.0, 0.0), 1, "set 1 TL  +RF"),
    ("rod_1_BR",         (0.0, 0.0, 0.0), 1, "set 1 BR  +RF"),
    ("rod_2_TL",         (0.0, 0.0, 0.0), 1, "set 2 TL  +RF"),
    ("rod_2_BR",         (0.0, 0.0, 0.0), 1, "set 2 BR  +RF"),
    ("rod_1_TR",         (0.0, 0.0, 0.0), 2, "set 1 TR  -RF"),
    ("rod_1_BL",         (0.0, 0.0, 0.0), 2, "set 1 BL  -RF"),
    ("rod_2_TR",         (0.0, 0.0, 0.0), 2, "set 2 TR  -RF"),
    ("rod_2_BL",         (0.0, 0.0, 0.0), 2, "set 2 BL  -RF"),
    # Load Paul trap endcaps
    ("endcap_load_U",    (0.0, 0.0, 0.0), 3, "load endcap U (+z)"),
    ("endcap_load_D",    (0.0, 0.0, 0.0), 4, "load endcap D (-z)"),
    # Set 3 (optical Paul trap, 4 independent rods)
    ("rod_3_TL",         (0.0, 0.0, 0.0), 5, "rod_3_TL  +RF3 + V_dc_3_TL"),
    ("rod_3_TR",         (0.0, 0.0, 0.0), 6, "rod_3_TR  -RF3 + V_dc_3_TR"),
    ("rod_3_BL",         (0.0, 0.0, 0.0), 7, "rod_3_BL  -RF3 + V_dc_3_BL"),
    ("rod_3_BR",         (0.0, 0.0, 0.0), 8, "rod_3_BR  +RF3 + V_dc_3_BR"),
    # Optical Paul trap endcaps
    ("endcap_optical_U", (0.0, 0.0, 0.0), 9,  "optical endcap U (+z)"),
    ("endcap_optical_D", (0.0, 0.0, 0.0), 10, "optical endcap D (-z)"),
    # Dielectric volumes — not in the electric PA
    ("trapping_lens",    (0.0, 0.0, 0.0), 0,  "dielectric (lens)"),
    ("collection_lens",  (0.0, 0.0, 0.0), 0,  "dielectric (lens)"),
    ("lens_holder",      (0.0, 0.0, 0.0), 0,  "dielectric (single uniform holder)"),
]

# Rod groupings for summary tables (positions in mm, Fusion world).
RODS_BY_SET = {
    1: ["rod_1_TL", "rod_1_TR", "rod_1_BL", "rod_1_BR"],
    2: ["rod_2_TL", "rod_2_TR", "rod_2_BL", "rod_2_BR"],
    3: ["rod_3_TL", "rod_3_TR", "rod_3_BL", "rod_3_BR"],
}
ENDCAPS    = ["endcap_load_U", "endcap_load_D", "endcap_optical_U", "endcap_optical_D"]
DIELECTRICS = ["trapping_lens", "collection_lens", "lens_holder"]

# ── STL parsing ───────────────────────────────────────────────────────────────

def read_stl_binary(path):
    triangles = []
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        for _ in range(n):
            struct.unpack("<fff", f.read(12))          # normal (unused)
            v0 = struct.unpack("<fff", f.read(12))
            v1 = struct.unpack("<fff", f.read(12))
            v2 = struct.unpack("<fff", f.read(12))
            f.read(2)
            triangles.append((v0, v1, v2))
    return triangles

def bbox(triangles):
    xs = [v[0] for tri in triangles for v in tri]
    ys = [v[1] for tri in triangles for v in tri]
    zs = [v[2] for tri in triangles for v in tri]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))

# ── Ray–triangle intersection (Möller–Trumbore) ───────────────────────────────

_EPS = 1e-9

def _ray_tri(orig, direction, v0, v1, v2):
    e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
    e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
    dx, dy, dz = direction
    h  = (dy*e2[2]-dz*e2[1], dz*e2[0]-dx*e2[2], dx*e2[1]-dy*e2[0])
    a  = e1[0]*h[0]+e1[1]*h[1]+e1[2]*h[2]
    if abs(a) < _EPS:
        return None
    f  = 1.0/a
    s  = (orig[0]-v0[0], orig[1]-v0[1], orig[2]-v0[2])
    u  = f*(s[0]*h[0]+s[1]*h[1]+s[2]*h[2])
    if u < 0.0 or u > 1.0:
        return None
    q  = (s[1]*e1[2]-s[2]*e1[1], s[2]*e1[0]-s[0]*e1[2], s[0]*e1[1]-s[1]*e1[0])
    v  = f*(dx*q[0]+dy*q[1]+dz*q[2])
    if v < 0.0 or u+v > 1.0:
        return None
    t  = f*(e2[0]*q[0]+e2[1]*q[1]+e2[2]*q[2])
    return t if t > _EPS else None

def point_in_mesh(point, triangles):
    """Majority vote across three ray directions — robust for imperfect meshes."""
    results = []
    for direction in [(1,0,0), (0,1,0), (0,0,1)]:
        count = sum(
            1 for tri in triangles
            if _ray_tri(point, direction, *tri) is not None
        )
        results.append(count % 2 == 1)
    return sum(results) >= 2

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_range(lo, hi):
    return f"[{lo:8.3f}, {hi:8.3f}]"

def within(val, lo, hi):
    return lo <= val <= hi

# ── Main loop ─────────────────────────────────────────────────────────────────

print("=" * 72)
print("paulTrap GEM — STL Sanity Check")
print(f"PA bounds (Fusion): X {fmt_range(*PA_X)}  Y {fmt_range(*PA_Y)}  Z {fmt_range(*PA_Z)}")
print("=" * 72)

all_ok  = True
results = {}

for (stem, seed, elec, desc) in ELECTRODES:
    path = os.path.join(BASE, stem + ".stl")
    sx, sy, sz = seed

    if not os.path.exists(path):
        print(f"\n[MISSING] {stem}.stl  (electrode {elec}: {desc})")
        all_ok = False
        results[stem] = None
        continue

    tris = read_stl_binary(path)
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bbox(tris)
    cx, cy, cz = (xlo+xhi)/2, (ylo+yhi)/2, (zlo+zhi)/2
    rx, ry, rz = (xhi-xlo)/2, (yhi-ylo)/2, (zhi-zlo)/2

    inside     = point_in_mesh(seed, tris)
    in_pa      = (PA_X[0] <= xlo and xhi <= PA_X[1] and
                  PA_Y[0] <= ylo and yhi <= PA_Y[1] and
                  PA_Z[0] <= zlo and zhi <= PA_Z[1])
    seed_in_pa = (within(sx, *PA_X) and within(sy, *PA_Y) and within(sz, *PA_Z))

    ok_seed = "OK  " if inside     else "FAIL"
    ok_pa   = "OK  " if in_pa      else "WARN"
    ok_spa  = "OK  " if seed_in_pa else "WARN"

    elec_label = f"electrode {elec}" if elec > 0 else "dielectric"
    print(f"\n── {stem}  ({elec_label}: {desc})")
    print(f"  Bbox X: {fmt_range(xlo,xhi)}  ctr={cx:7.3f}  half={rx:.3f}")
    print(f"  Bbox Y: {fmt_range(ylo,yhi)}  ctr={cy:7.3f}  half={ry:.3f}")
    print(f"  Bbox Z: {fmt_range(zlo,zhi)}  ctr={cz:7.3f}  half={rz:.3f}")
    print(f"  Seed   ({sx:.4f}, {sy:.4f}, {sz:.4f})")
    print(f"  Seed inside mesh : [{ok_seed}]   Mesh in PA: [{ok_pa}]   Seed in PA: [{ok_spa}]")

    if not inside:
        print(f"  *** SEED IS OUTSIDE MESH — SIMION will fill incorrect region ***")
        print(f"      Suggest seed: ({cx:.4f}, {cy:.4f}, {cz:.4f})  (bbox centre)")
        all_ok = False
    if not in_pa:
        print(f"  *** MESH EXTENDS OUTSIDE PA — increase pa_define dimensions ***")
        all_ok = False

    results[stem] = dict(
        xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, zlo=zlo, zhi=zhi,
        cx=cx, cy=cy, cz=cz, rx=rx, ry=ry, rz=rz,
        inside=inside, in_pa=in_pa,
    )

# ── Rod cross-section summary by set ──────────────────────────────────────────

for set_num, rod_stems in RODS_BY_SET.items():
    print("\n" + "=" * 72)
    print(f"Set {set_num} rod cross-section summary  (quadrupole axis along Z)")
    print(f"  {'Name':<14} {'X_ctr':>8} {'Y_ctr':>8} {'Z_ctr':>8} {'r_X':>6} {'r_Y':>6}")
    for stem in rod_stems:
        r = results.get(stem)
        if r is None:
            print(f"  {stem:<14}  (missing)")
            continue
        print(f"  {stem:<14} {r['cx']:8.3f} {r['cy']:8.3f} {r['cz']:8.3f} {r['rx']:6.3f} {r['ry']:6.3f}")

    # Compute r_0 (half-distance between opposite rods) within this set.
    cxs = [results[s]["cx"] for s in rod_stems if results.get(s)]
    cys = [results[s]["cy"] for s in rod_stems if results.get(s)]
    if len(cxs) == 4 and len(cys) == 4:
        r0_x = (max(cxs) - min(cxs)) / 2
        r0_y = (max(cys) - min(cys)) / 2
        print(f"\n  X half-gap r0_X ≈ {r0_x:.3f} mm,  Y half-gap r0_Y ≈ {r0_y:.3f} mm")
        if abs(r0_x - r0_y) > 0.3:
            print("  *** X and Y half-gaps differ — trap may not be square ***")

# ── End-cap Z positions ───────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("End-cap Z positions  (Fusion world)")
for stem in ENDCAPS:
    r = results.get(stem)
    if r is None:
        print(f"  {stem:<22}  (missing)")
        continue
    print(f"  {stem:<22}  Z_ctr = {r['cz']:.3f} mm   X_ctr = {r['cx']:.3f}   Y_ctr = {r['cy']:.3f}")

# ── Dielectric body positions ─────────────────────────────────────────────────

print("\n" + "=" * 72)
print("Dielectric body positions  (lenses + lens holder)")
for stem in DIELECTRICS:
    r = results.get(stem)
    if r is None:
        print(f"  {stem:<22}  (missing)")
        continue
    print(f"  {stem:<22}  ctr = ({r['cx']:.3f}, {r['cy']:.3f}, {r['cz']:.3f})")

# ── Final verdict ─────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
if all_ok:
    print("Overall: All checks passed.")
else:
    print("Overall: One or more issues found — see FAIL/WARN lines above.")
    print("  FAIL = seed point outside mesh → SIMION will fill wrong region.")
    print("  WARN = geometry outside PA bounds → increase pa_define dimensions.")
print("=" * 72)
