"""
generate_dielectric_pa.py  –  Build the SIMION dielectric permittivity array
for the perpendicular-trap glass lenses.

The dielectric PA stores relative permittivity (ε_r) at each grid-cell centre.
It is consumed by refine_with_dielectric.lua, which re-refines every electrode
PA with it so that the glass lenses correctly modify the electric field.

Grid conventions (must match paulTrap.gem):
  Electric PA dimensions : NX=119, NY=91, NZ=855  (pa_define 59×45×427 mm, dx=0.5)
  Dielectric PA dimensions: NX-1, NY-1, NZ-1 = 118×90×854
  Cell (i,j,k) centre (Fusion mm) :
      x = (i + 0.5)*DX + FUSION_X_MIN
      y = (j + 0.5)*DX + FUSION_Y_MIN
      z = (k + 0.5)*DX + FUSION_Z_MIN

Binary format (same as SIMION charge-density / space-charge PA):
  56-byte header (identical fields to the electric PA header)
  nx_d * ny_d * nz_d  float64 values, z outermost / x innermost  →  V[iz,iy,ix]
  Values represent ε_r directly (1.0 = vacuum, EPSILON_GLASS = glass).

Usage:
    ~/.venvs/mesh/bin/python3 generate_dielectric_pa.py
Requires: trimesh  (pip install trimesh in ~/.venvs/mesh)
"""

import os, struct
import numpy as np

try:
    import trimesh
except ImportError:
    raise SystemExit(
        "trimesh is required.  Install it with:\n"
        "  ~/.venvs/mesh/bin/pip install trimesh"
    )

# ── Configuration ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# Dielectric constant of the glass lenses.
# PLACEHOLDER: replace with the correct value for your glass.
# Common values: fused silica ≈ 3.82 (DC), BK7 ≈ 7.1 (DC), N-BK7 ≈ 7.1 (DC).
# For optical-frequency fields the refractive-index squared is used instead
# (fused silica n≈1.46 → ε_r≈2.13 at 1064 nm).
EPSILON_GLASS = 3.0   # PLACEHOLDER

# STL files that define glass dielectric volumes (Fusion world coordinates)
LENS_STLS = [
    "trapping_lens.stl",
    "collection_lens.stl",
]

# PA grid — must match pa_define in paulTrap.gem
NX, NY, NZ = 119, 91, 855    # electric PA grid points
DX = 0.5                      # mm

# GEM → Fusion offset: Fusion = GEM_coord - (tx, ty, tz)
TX, TY, TZ = 25, 8, 132

# Fusion-world origin of GEM index (0,0,0)
FUSION_X_MIN = -TX   # = -25 mm
FUSION_Y_MIN = -TY   # =  -8 mm
FUSION_Z_MIN = -TZ   # = -132 mm

# Output filename
OUT_PA = os.path.join(BASE, "paulTrap-dielectric.pa")


# ── Build cell-centre coordinate arrays ───────────────────────────────────────

NX_D = NX - 1   # 118
NY_D = NY - 1   # 90
NZ_D = NZ - 1   # 854

# Cell centres along each axis (Fusion mm)
x_c = (np.arange(NX_D) + 0.5) * DX + FUSION_X_MIN
y_c = (np.arange(NY_D) + 0.5) * DX + FUSION_Y_MIN
z_c = (np.arange(NZ_D) + 0.5) * DX + FUSION_Z_MIN

print(f"Dielectric array: {NX_D} × {NY_D} × {NZ_D}  ({NX_D*NY_D*NZ_D:,} cells)")
print(f"Fusion extent: X [{x_c[0]:.2f}, {x_c[-1]:.2f}]  "
      f"Y [{y_c[0]:.2f}, {y_c[-1]:.2f}]  "
      f"Z [{z_c[0]:.2f}, {z_c[-1]:.2f}]  (cell centres, mm)")


# ── Initialise dielectric array to vacuum ─────────────────────────────────────

epsilon = np.ones((NZ_D, NY_D, NX_D), dtype=np.float64)  # V[iz, iy, ix] = 1.0


# ── Mark lens voxels ──────────────────────────────────────────────────────────

for stl_name in LENS_STLS:
    stl_path = os.path.join(BASE, stl_name)
    if not os.path.exists(stl_path):
        print(f"  WARNING: {stl_name} not found — skipping")
        continue

    mesh = trimesh.load_mesh(stl_path)
    print(f"  {stl_name}: {mesh.vertices.shape[0]} verts, "
          f"bbox {mesh.bounds[0]} → {mesh.bounds[1]}")

    # Restrict to cells whose centres fall within the mesh bounding box
    # (with a small margin) to avoid testing every cell in the array.
    bb_lo, bb_hi = mesh.bounds
    ix_lo = max(0, int(np.floor((bb_lo[0] - FUSION_X_MIN) / DX - 1)))
    ix_hi = min(NX_D, int(np.ceil( (bb_hi[0] - FUSION_X_MIN) / DX + 1)))
    iy_lo = max(0, int(np.floor((bb_lo[1] - FUSION_Y_MIN) / DX - 1)))
    iy_hi = min(NY_D, int(np.ceil( (bb_hi[1] - FUSION_Y_MIN) / DX + 1)))
    iz_lo = max(0, int(np.floor((bb_lo[2] - FUSION_Z_MIN) / DX - 1)))
    iz_hi = min(NZ_D, int(np.ceil( (bb_hi[2] - FUSION_Z_MIN) / DX + 1)))

    print(f"    Candidate cell range: ix [{ix_lo},{ix_hi}]  "
          f"iy [{iy_lo},{iy_hi}]  iz [{iz_lo},{iz_hi}]")

    # Build all candidate cell-centre positions as an (N,3) array
    xi = x_c[ix_lo:ix_hi]
    yi = y_c[iy_lo:iy_hi]
    zi = z_c[iz_lo:iz_hi]
    XX, YY, ZZ = np.meshgrid(xi, yi, zi, indexing='ij')   # (nx_sub, ny_sub, nz_sub)
    pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    # Point-in-mesh test (ray casting via trimesh)
    inside = mesh.contains(pts)                            # bool (N,)
    inside_3d = inside.reshape(
        len(xi), len(yi), len(zi))                         # [ix_sub, iy_sub, iz_sub]

    # Write into epsilon array (stored as [iz, iy, ix])
    n_inside = inside.sum()
    print(f"    Cells inside mesh: {n_inside}")
    for dix in range(len(xi)):
        for diy in range(len(yi)):
            for diz in range(len(zi)):
                if inside_3d[dix, diy, diz]:
                    epsilon[iz_lo + diz,
                            iy_lo + diy,
                            ix_lo + dix] = EPSILON_GLASS


# ── Write dielectric PA binary file ──────────────────────────────────────────

n_pts   = NX_D * NY_D * NZ_D
n_glass = int((epsilon > 1.0).sum())
print(f"\nGlass cells: {n_glass}  ({100*n_glass/n_pts:.3f}% of array)")
print(f"Writing {OUT_PA} ...")

# Flatten to 1-D in SIMION storage order: z outermost, x innermost → C order
flat = epsilon.flatten(order='C')   # [iz, iy, ix] C-order → z slowest, x fastest ✓

with open(OUT_PA, 'wb') as f:
    # Header (56 bytes)
    f.write(struct.pack('<i', -2))           # [0:4]  version flag
    f.write(struct.pack('<i',  1))           # [4:8]
    f.write(struct.pack('<d',  1.0))         # [8:16] scale_ref = 1.0 (ε_r is dimensionless)
    f.write(struct.pack('<i', NX_D))         # [16:20]
    f.write(struct.pack('<i', NY_D))         # [20:24]
    f.write(struct.pack('<i', NZ_D))         # [24:28]
    f.write(struct.pack('<i', 1600))         # [28:32] (standard SIMION field)
    f.write(struct.pack('<d', DX))           # [32:40] dx mm
    f.write(struct.pack('<d', DX))           # [40:48] dy mm
    f.write(struct.pack('<d', DX))           # [48:56] dz mm
    # Data
    f.write(flat.astype('<f8').tobytes())

expected_size = 56 + n_pts * 8
actual_size   = os.path.getsize(OUT_PA)
print(f"Written {actual_size:,} bytes  (expected {expected_size:,})  "
      f"{'OK' if actual_size == expected_size else 'MISMATCH!'}")
print("Done.")
