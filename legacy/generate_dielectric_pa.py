"""
generate_dielectric_pa.py  –  Build the SIMION dielectric permittivity array
for the optical-region glass lenses and the lens holder.

The dielectric PA stores relative permittivity (ε_r) at each grid-cell centre.
It is consumed by refine_with_dielectric.lua, which re-refines every electrode
PA with it so that the dielectric volumes correctly modify the electric field.

Grid conventions (must match paulTrap.gem):
  Electric PA dimensions  : NX, NY, NZ — auto-detected from paulTrap.pa0
                            header, so a change to `dx` in paulTrap.gem
                            takes effect after the next Refine.  If
                            paulTrap.pa0 is missing the script falls back
                            to hardcoded defaults with a warning.
  Dielectric PA dimensions: NX-1, NY-1, NZ-1
  Cell (i,j,k) centre (Fusion mm) :
      x = (i + 0.5)*DX + FUSION_X_MIN
      y = (j + 0.5)*DX + FUSION_Y_MIN
      z = (k + 0.5)*DX + FUSION_Z_MIN

Memory note: trimesh.contains() can allocate many GB internally for
complex non-convex meshes (e.g. a lens holder with through-holes for the
rods).  This script processes one Z-slice at a time so peak memory is
bounded by nx_sub × ny_sub points regardless of mesh complexity.

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

# Dielectric constant applied to every mesh in DIELECTRIC_STLS.
# PLACEHOLDER: replace with the correct value for your materials.  Fused silica
# ≈ 3.82 (DC); PEEK ≈ 3.2.  These are close enough that a single ε_r covers
# both lenses and the PEEK holder.  If a per-mesh value is needed later,
# replace DIELECTRIC_STLS with a {filename: ε_r} dict and update the loop.
EPSILON_GLASS = 3.0   # PLACEHOLDER

# STL files that define dielectric volumes (Fusion world coordinates).
# Includes both lenses and the single uniform lens holder.
DIELECTRIC_STLS = [
    "trapping_lens.stl",
    "collection_lens.stl",
    "lens_holder.stl",
]

# PA grid — auto-detected from paulTrap.pa0 header below.  If paulTrap.pa0
# is missing (no Refine has been run yet), these defaults are used and a
# warning is printed.  After the first Refine, the script reads the actual
# NX, NY, NZ, DX out of the PA header so the dielectric PA stays aligned
# with whatever dx is currently set in paulTrap.gem.
NX, NY, NZ = 131, 91, 855
DX = 0.5

PA_HEADER_PATH = os.path.join(BASE, "paulTrap.pa0")
if os.path.exists(PA_HEADER_PATH):
    with open(PA_HEADER_PATH, "rb") as _fh:
        _hdr = _fh.read(56)
    NX = struct.unpack_from("<i", _hdr, 16)[0]
    NY = struct.unpack_from("<i", _hdr, 20)[0]
    NZ = struct.unpack_from("<i", _hdr, 24)[0]
    DX = struct.unpack_from("<d", _hdr, 32)[0]
    print(f"Detected electric PA grid from paulTrap.pa0: "
          f"NX={NX}, NY={NY}, NZ={NZ}, dx={DX:.4g} mm")
else:
    print(f"  WARNING: {PA_HEADER_PATH} not found.  Falling back to "
          f"hardcoded NX={NX}, NY={NY}, NZ={NZ}, DX={DX} mm — these MUST "
          f"match pa_define in paulTrap.gem or the dielectric PA will not "
          f"align with the electric PAs.")

# GEM → Fusion offset: Fusion = GEM_coord - (tx, ty, tz).
# Must match the locate(tx, ty, tz) block in paulTrap.gem.  (Not stored in
# the PA header, so still set manually here.)
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


# ── Mark dielectric voxels ────────────────────────────────────────────────────

for stl_name in DIELECTRIC_STLS:
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

    nx_sub = ix_hi - ix_lo
    ny_sub = iy_hi - iy_lo
    nz_sub = iz_hi - iz_lo
    print(f"    Candidate cell range: ix [{ix_lo},{ix_hi}]  "
          f"iy [{iy_lo},{iy_hi}]  iz [{iz_lo},{iz_hi}]  "
          f"({nx_sub * ny_sub * nz_sub:,} cells)")

    # Process one Z-slice at a time.  trimesh.contains() on millions of points
    # in one call allocates several × the input size internally; per-slice
    # keeps peak memory bounded by nx_sub * ny_sub points (~10s of MB max
    # even on a fine grid).  The mesh's ray intersector is cached after the
    # first call so per-call overhead is small.
    xi = x_c[ix_lo:ix_hi]
    yi = y_c[iy_lo:iy_hi]
    zi = z_c[iz_lo:iz_hi]
    XX_2d, YY_2d = np.meshgrid(xi, yi, indexing='ij')      # (nx_sub, ny_sub)
    xy_flat = np.column_stack([XX_2d.ravel(), YY_2d.ravel()])

    pts = np.empty((nx_sub * ny_sub, 3), dtype=np.float64)
    pts[:, 0:2] = xy_flat
    n_inside = 0
    for diz, z_val in enumerate(zi):
        pts[:, 2] = z_val
        inside_flat = mesh.contains(pts)                   # bool (nx_sub*ny_sub,)
        if not inside_flat.any():
            continue
        inside_2d = inside_flat.reshape(nx_sub, ny_sub)
        ix_hits, iy_hits = np.where(inside_2d)
        epsilon[iz_lo + diz,
                iy_lo + iy_hits,
                ix_lo + ix_hits] = EPSILON_GLASS
        n_inside += int(inside_flat.sum())

    print(f"    Cells inside mesh: {n_inside}")


# ── Write dielectric PA binary file ──────────────────────────────────────────

n_pts   = NX_D * NY_D * NZ_D
n_diel = int((epsilon > 1.0).sum())
print(f"\nDielectric cells: {n_diel}  ({100*n_diel/n_pts:.3f}% of array)")
print(f"Writing {OUT_PA} ...")

# SIMION storage order is z outermost / x innermost, which is exactly the
# C-order of our [iz, iy, ix] array — no flatten or copy needed.  We require
# native little-endian float64 (which is what numpy uses on every platform
# this is likely to run on, but assert just in case).
assert epsilon.dtype == np.float64 and epsilon.flags["C_CONTIGUOUS"]

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
    # Data — write the array's buffer directly to avoid the 2× memory cost
    # of materialising a full bytes copy.
    epsilon.tofile(f)

expected_size = 56 + n_pts * 8
actual_size   = os.path.getsize(OUT_PA)
print(f"Written {actual_size:,} bytes  (expected {expected_size:,})  "
      f"{'OK' if actual_size == expected_size else 'MISMATCH!'}")
print("Done.")
