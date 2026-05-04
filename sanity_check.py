"""
sanity_check.py  –  GEM/STL geometry verification for paulTrap.gem

Checks for every STL referenced in the GEM (and for the dielectric lenses):
  1. Bounding box in Fusion world coordinates
  2. Whether the GEM seed point lies inside the mesh (ray-casting parity test)
  3. Whether the mesh falls within the PA bounds
  4. Per-group geometry summaries: main-trap rod axes, perp-trap rod axes,
     ring electrode Z positions, end-cap Z positions, lens holder positions

Electrodes checked
  1  rod_P1_L1, rod_P1_L2        main trap rod pair 1 (+RF)
  2  rod_P2_L1, rod_P2_L2        main trap rod pair 2 (−RF)
  3  endcap_L                    left end cap
  4  rod_P1_R1, rod_P1_R2        right rod pair 1 (+RF)
  5  rod_P2_R1, rod_P2_R2        right rod pair 2 (−RF)
  6  ring_L                      ring electrode, left
  7  ring_R                      ring electrode, right
  8  endcap_R                    right end cap
  9  trap_rod_TL, trap_rod_BR    perp-trap rod pair 1 (+RF2)
 10  trap_rod_TR, trap_rod_BL    perp-trap rod pair 2 (−RF2)
 11  trapping_lens_holder        perp-trap DC electrode
 12  collection_lens_holder      perp-trap DC electrode
  –  trapping_lens               glass dielectric (not in electric PA)
  –  collection_lens             glass dielectric (not in electric PA)
"""

import struct, math, os

BASE = os.path.dirname(os.path.abspath(__file__))

# ── PA bounds in Fusion world coordinates ─────────────────────────────────────
# pa_define{59×45×427 mm, dx=0.5}, locate(25, 8, 132)
# Fusion = GEM - (25, 8, 132)
PA_X = (-25.0,  40.0)
PA_Y = ( -8.0,  37.0)
PA_Z = (-132.0, 295.0)

# ── Electrodes: (stl_stem, seed_xyz, electrode_number, short_description) ─────
# Seed points are Fusion world coordinates, exactly as written in paulTrap.gem.
# Electrode 0 = not in the electric PA (dielectric only).
ELECTRODES = [
    # Main Paul trap — axis along Z
    ("rod_P1_L1",             (-2.1082, 21.1582,  -20.6470),  1, "rod pair 1 left  +RF"),
    ("rod_P1_L2",             ( 2.1082, 16.9672,  -20.6470),  1, "rod pair 1 left  +RF"),
    ("rod_P2_L1",             ( 2.1082, 21.1582,  -20.6470),  2, "rod pair 2 left  −RF"),
    ("rod_P2_L2",             (-2.1082, 16.9672,  -20.6470),  2, "rod pair 2 left  −RF"),
    ("endcap_L",              ( 7.0000, 19.0600, -114.9950),  3, "left end cap"),
    ("rod_P1_R1",             (-2.1082, 21.1582,  169.2021),  4, "rod pair 1 right +RF"),
    ("rod_P1_R2",             ( 2.1082, 16.9672,  169.2021),  4, "rod pair 1 right +RF"),
    ("rod_P2_R1",             ( 2.1082, 21.1582,  169.2021),  5, "rod pair 2 right −RF"),
    ("rod_P2_R2",             (-2.1082, 16.9672,  169.2021),  5, "rod pair 2 right −RF"),
    ("ring_L",                (10.0000, 19.0600,   67.3860),  6, "ring left"),
    ("ring_R",                (10.0000, 19.0600,  110.3760),  7, "ring right"),
    ("endcap_R",              ( 7.0000, 19.0600,  -81.2450),  8, "right end cap"),
    # Perpendicular Paul trap — axis along X
    ("trap_rod_TL",           (15.3960, 25.9140,  269.6560),  9, "perp rod pair 1 +RF2"),
    ("trap_rod_BR",           (15.3960, 13.2140,  282.3560),  9, "perp rod pair 1 +RF2"),
    ("trap_rod_TR",           (15.3960, 25.9140,  282.3560), 10, "perp rod pair 2 −RF2"),
    ("trap_rod_BL",           (15.3960, 13.2140,  269.6560), 10, "perp rod pair 2 −RF2"),
    ("trapping_lens_holder",  ( 4.8300, 13.9130,  275.9940), 11, "trapping lens holder DC"),
    ("collection_lens_holder",(-6.0730, 15.1270,  275.9690), 12, "collection lens holder DC"),
    # Dielectric lenses — not in the electric PA; checked for generate_dielectric_pa.py
    ("trapping_lens",         ( 4.0420, 19.5860,  275.9840),  0, "dielectric (not in electric PA)"),
    ("collection_lens",       (-6.5820, 19.5300,  275.9330),  0, "dielectric (not in electric PA)"),
]

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
    xs = [v[i] for tri in triangles for v in tri for i in [0]]
    ys = [v[i] for tri in triangles for v in tri for i in [1]]
    zs = [v[i] for tri in triangles for v in tri for i in [2]]
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

all_ok     = True
results    = {}   # stl_stem → dict of computed values

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

    ok_seed    = "OK  " if inside     else "FAIL"
    ok_pa      = "OK  " if in_pa      else "WARN"
    ok_spa     = "OK  " if seed_in_pa else "WARN"

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

# ── Main-trap rod cross-section summary ───────────────────────────────────────

MAIN_RODS = ["rod_P1_L1","rod_P1_L2","rod_P2_L1","rod_P2_L2",
             "rod_P1_R1","rod_P1_R2","rod_P2_R1","rod_P2_R2"]

print("\n" + "=" * 72)
print("Main-trap rod cross-section summary  (quadrupole axis along Z)")
print("  Expected: rod radius ≈ 1.5 mm, |X| from trap axis ≈ 2.1 mm")
print(f"  {'Name':<18} {'X_ctr':>7} {'Y_ctr':>7} {'r_X':>6} {'r_Y':>6}")
for stem in MAIN_RODS:
    r = results.get(stem)
    if r is None:
        print(f"  {stem:<18}  (missing)")
        continue
    flag = ""
    if abs(r["rx"] - 1.5) > 0.4 or abs(r["ry"] - 1.5) > 0.4:
        flag = "  ← radius unexpected"
    print(f"  {stem:<18} {r['cx']:7.3f} {r['cy']:7.3f} {r['rx']:6.3f} {r['ry']:6.3f}{flag}")

# ── Perp-trap rod cross-section summary ───────────────────────────────────────

PERP_RODS = ["trap_rod_TL","trap_rod_TR","trap_rod_BL","trap_rod_BR"]

print("\n" + "=" * 72)
print("Perp-trap rod cross-section summary  (quadrupole axis along X)")
print("  Rod long axis is X; quadrupole cross-section is in Y-Z plane.")
print("  Expected: TL/TR Y ≈ 25.9 mm, BL/BR Y ≈ 13.2 mm")
print("  Expected: TL/BL Z ≈ 269.7 mm, TR/BR Z ≈ 282.4 mm")
print(f"  {'Name':<18} {'Y_ctr':>7} {'Z_ctr':>7} {'r_Y':>6} {'r_Z':>6}")

perp_y_top, perp_y_bot, perp_z_L, perp_z_R = [], [], [], []
for stem in PERP_RODS:
    r = results.get(stem)
    if r is None:
        print(f"  {stem:<18}  (missing)")
        continue
    flag = ""
    if abs(r["ry"] - 2.87) > 0.5 or abs(r["rz"] - 2.87) > 0.5:
        flag = "  ← radius unexpected"
    print(f"  {stem:<18} {r['cy']:7.3f} {r['cz']:7.3f} {r['ry']:6.3f} {r['rz']:6.3f}{flag}")
    if "TL" in stem or "TR" in stem:
        perp_y_top.append(r["cy"])
    else:
        perp_y_bot.append(r["cy"])
    if "TL" in stem or "BL" in stem:
        perp_z_L.append(r["cz"])
    else:
        perp_z_R.append(r["cz"])

if perp_y_top and perp_y_bot:
    y_sep = sum(perp_y_top)/len(perp_y_top) - sum(perp_y_bot)/len(perp_y_bot)
    z_sep = sum(perp_z_R)/len(perp_z_R)     - sum(perp_z_L)/len(perp_z_L)
    r0_y  = y_sep / 2
    r0_z  = z_sep / 2
    print(f"\n  Y rod-centre separation: {y_sep:.3f} mm  →  Y half-gap r0_Y ≈ {r0_y:.3f} mm")
    print(f"  Z rod-centre separation: {z_sep:.3f} mm  →  Z half-gap r0_Z ≈ {r0_z:.3f} mm")
    if abs(y_sep - z_sep) > 0.5:
        print("  *** Y and Z separations differ — trap may not be square ***")

# ── Ring electrode positions ──────────────────────────────────────────────────

print("\n" + "=" * 72)
print("Ring electrode Z positions  (should be in gap region, Fusion Z ≈ 67–110 mm)")
GAP_Z = (67.0, 111.0)
for stem in ["ring_L", "ring_R"]:
    r = results.get(stem)
    if r is None:
        continue
    flag = "" if (GAP_Z[0] <= r["cz"] <= GAP_Z[1]) else "  ← outside expected gap region"
    print(f"  {stem:<12}  Z_ctr = {r['cz']:.3f} mm{flag}")

# ── End-cap Z positions ───────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("End-cap Z positions")
expected = {"endcap_L": -115.0, "endcap_R": -81.24}
for stem, exp_z in expected.items():
    r = results.get(stem)
    if r is None:
        continue
    err = r["cz"] - exp_z
    flag = "" if abs(err) < 2.0 else f"  ← {err:+.2f} mm from expected {exp_z} mm"
    print(f"  {stem:<14}  Z_ctr = {r['cz']:.3f} mm  (expected ≈ {exp_z} mm){flag}")

# ── Lens holder and lens positions ────────────────────────────────────────────

print("\n" + "=" * 72)
print("Perp-trap lens positions  (should cluster near Fusion Z ≈ 276 mm)")
LENS_STEMS = ["trapping_lens_holder", "collection_lens_holder",
              "trapping_lens",        "collection_lens"]
for stem in LENS_STEMS:
    r = results.get(stem)
    if r is None:
        continue
    in_lens_region = (260 <= r["cz"] <= 292)
    flag = "" if in_lens_region else "  ← outside expected Z region"
    print(f"  {stem:<28}  ctr = ({r['cx']:.2f}, {r['cy']:.2f}, {r['cz']:.2f}){flag}")

# ── Final verdict ─────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
if all_ok:
    print("Overall: All checks passed.")
else:
    print("Overall: One or more issues found — see FAIL/WARN lines above.")
    print("  FAIL = seed point outside mesh → SIMION will fill wrong region.")
    print("  WARN = geometry outside PA bounds → increase pa_define dimensions.")
print("=" * 72)
