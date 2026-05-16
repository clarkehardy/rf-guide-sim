"""
rasterize_pa.py  –  Build a SIMION pa# geometry file by point-in-mesh testing,
bypassing SIMION's GEM voxelizer.

Why this exists
---------------
SIMION 8.2's dielectric solver is incompatible with surface='fractional'.
Without fractional surfaces, the GEM-to-pa# voxelizer marks surface voxels
then flood-fills from a seed point.  For meshes whose surfaces graze the grid
at shallow angles (rods of any orientation, curved endcaps, the lens holder),
the marked surface cells can have sub-grid gaps that the flood-fill walks
through, leaking out and filling the entire free space.  Repairing the STLs
doesn't fix this — the problem is geometric, not mesh-quality.

This script sidesteps the voxelizer entirely.  For every grid cell, it asks
trimesh.contains(cell_centre) for each electrode's STL — a true point-in-mesh
test using the embree ray engine (via embreex).  Cells inside an electrode
mesh get marked with that electrode's pa# value; everything else stays 0.
No surface tracking, no flood-fill, no leak paths.

This is an *alternative* to the regular workflow, not a replacement.  Run
SIMION's GEM Refine once first (it sets up paulTrap.pa# at your chosen dx);
this script then writes paulTrap_rasterized.pa#.  Swap them in/out of place
when you want to switch workflows.

Suggested cross-check workflow
------------------------------
  1.  In SIMION: load paulTrap.gem, Refine.  Produces paulTrap.pa# at the
      grid dx you want (and pa0..pa10 too, but we'll overwrite those).
  2.  python rasterize_pa.py
        → writes paulTrap_rasterized.pa#  (leak-free electrode markers).
  3.  cp paulTrap.pa# paulTrap.pa#.simion-backup
      cp paulTrap_rasterized.pa# paulTrap.pa#
  4.  In SIMION: Refine *again*.  This time SIMION reads the corrected pa#
      and just runs the relaxation solver — no re-voxelization — producing
      clean pa1..pa10.
  5.  python generate_dielectric_pa.py    (unchanged)
  6.  In SIMION: run refine_with_dielectric.lua (unchanged).

When you want to go back to the SIMION-voxelized + fractional-surface workflow:
  cp paulTrap.pa#.simion-backup paulTrap.pa#
  ... then a normal Refine.

Cell encoding (verified against an existing paulTrap.pa#)
---------------------------------------------------------
  Header (56 bytes)  : identical to the paN format used by plot_field.py.
  Free cell          : 0.0
  Electrode N cell   : 200000.0 + float(N)            (= 2 * scale_ref + N)
  Indexing           : V[iz, iy, ix], z outermost, x innermost.
  Cell coordinates   : node-centred (Fusion x = ix*DX + FUSION_X_MIN),
                        NOT cell-centred — pa# uses node values.

The header from the source pa# is copied verbatim so scale_ref, mirroring,
and any other SIMION-specific flags are preserved.
"""

import argparse
import os
import struct
import time

import numpy as np

try:
    import trimesh
except ImportError:
    raise SystemExit(
        "trimesh required.\n"
        "  ~/.venvs/mesh/bin/pip install trimesh embreex"
    )

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Electrode → list of STL files ────────────────────────────────────────────
# Must match the e() { ... } blocks in paulTrap.gem.
ELECTRODE_STLS = {
    1:  ["rod_1_TL.stl", "rod_1_BR.stl", "rod_2_TL.stl", "rod_2_BR.stl"],
    2:  ["rod_1_TR.stl", "rod_1_BL.stl", "rod_2_TR.stl", "rod_2_BL.stl"],
    3:  ["endcap_load_U.stl"],
    4:  ["endcap_load_D.stl"],
    5:  ["rod_3_TL.stl"],
    6:  ["rod_3_TR.stl"],
    7:  ["rod_3_BL.stl"],
    8:  ["rod_3_BR.stl"],
    9:  ["endcap_optical_U.stl"],
    10: ["endcap_optical_D.stl"],
}

# ── Coordinate transform: Fusion → GEM ───────────────────────────────────────
# Must match locate(tx, ty, tz) in paulTrap.gem and gem_offset in trap_config.lua.
TX, TY, TZ = 25, 8, 132
FUSION_X_MIN = -TX
FUSION_Y_MIN = -TY
FUSION_Z_MIN = -TZ

# Electrode-marker encoding for SIMION pa# files
SCALE_REF      = 100000.0
ELECTRODE_BASE = 2.0 * SCALE_REF     # electrode N → value (ELECTRODE_BASE + N)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--source-pa", default="paulTrap.pa#",
        help="Existing PA file to read grid dimensions and header bytes from "
             "(default: paulTrap.pa#).  Run SIMION's GEM Refine once first "
             "to create this file at the desired grid size.")
    ap.add_argument(
        "--out", default="paulTrap_rasterized.pa#",
        help="Output pa# file (default: paulTrap_rasterized.pa#).  Pass "
             "--out paulTrap.pa# to overwrite the SIMION-generated file "
             "directly (back it up first).")
    ap.add_argument(
        "--verify", action="store_true",
        help="After writing, compare per-electrode cell counts against the "
             "source pa# and print a diff.  Useful to confirm that the "
             "rasterizer agrees with SIMION's voxelizer in non-leaking regions.")
    args = ap.parse_args()

    source_pa = os.path.join(BASE, args.source_pa)
    out_path  = os.path.join(BASE, args.out)

    # ── Header + grid dims from source PA ────────────────────────────────────
    if not os.path.exists(source_pa):
        raise SystemExit(
            f"{source_pa} not found.  Run SIMION's GEM Refine once first.")
    with open(source_pa, "rb") as f:
        hdr_src = f.read(56)
    NX  = struct.unpack_from("<i", hdr_src, 16)[0]
    NY  = struct.unpack_from("<i", hdr_src, 20)[0]
    NZ  = struct.unpack_from("<i", hdr_src, 24)[0]
    DX  = struct.unpack_from("<d", hdr_src, 32)[0]
    scale = struct.unpack_from("<d", hdr_src, 8)[0]
    print(f"Source PA: {args.source_pa}")
    print(f"  NX, NY, NZ = {NX}, {NY}, {NZ}  (dx = {DX} mm)")
    print(f"  scale_ref  = {scale}  (electrode marker = {ELECTRODE_BASE + 1:.0f} for elec 1)")
    print(f"  ε array    = {NX*NY*NZ:,} cells "
          f"({NX*NY*NZ*8/1e9:.2f} GB float64)")

    # ── Node coordinates along each axis (Fusion mm) ──────────────────────────
    x_n = np.arange(NX) * DX + FUSION_X_MIN
    y_n = np.arange(NY) * DX + FUSION_Y_MIN
    z_n = np.arange(NZ) * DX + FUSION_Z_MIN
    print(f"  Fusion extent:  X [{x_n[0]:.2f}, {x_n[-1]:.2f}]  "
          f"Y [{y_n[0]:.2f}, {y_n[-1]:.2f}]  "
          f"Z [{z_n[0]:.2f}, {z_n[-1]:.2f}]  mm")

    # ── Allocate PA array (free space = 0) ───────────────────────────────────
    pa = np.zeros((NZ, NY, NX), dtype=np.float64)

    # ── Rasterize each electrode ─────────────────────────────────────────────
    print("\nRasterizing electrodes:")
    t0 = time.time()
    per_elec_counts = {}
    for elec_num, stl_list in ELECTRODE_STLS.items():
        marker = ELECTRODE_BASE + float(elec_num)
        n_total = 0
        for stl_name in stl_list:
            n_total += rasterize_one(pa, stl_name, marker,
                                     x_n, y_n, z_n, NX, NY, NZ, DX)
        per_elec_counts[elec_num] = n_total
        print(f"  elec {elec_num:2d}  →  marker {marker:>8.0f}  "
              f"({n_total:>9,} cells across {len(stl_list)} STL{'s' if len(stl_list)>1 else ''})")
    print(f"\nRasterization done in {time.time() - t0:.1f} s")

    # ── Write PA file (copy source header verbatim, then data) ───────────────
    print(f"\nWriting {out_path}")
    with open(out_path, "wb") as f:
        f.write(hdr_src)
        pa.tofile(f)
    expected = 56 + NX * NY * NZ * 8
    actual   = os.path.getsize(out_path)
    print(f"  Wrote {actual:,} bytes  (expected {expected:,})  "
          f"{'OK' if actual == expected else 'MISMATCH'}")
    n_filled = int((pa > 0).sum())
    print(f"  Total filled cells: {n_filled:,}  "
          f"({100*n_filled/pa.size:.4f}% of grid)")

    # ── Optional: compare against the source pa# ────────────────────────────
    if args.verify:
        verify_against_source(source_pa, per_elec_counts, NX, NY, NZ)


def rasterize_one(pa, stl_name, marker, x_n, y_n, z_n, NX, NY, NZ, DX):
    """Stamp `marker` into `pa` at every grid node inside the STL mesh.
    Operates one Z-slice at a time so peak memory is bounded regardless of
    mesh size.  Returns the count of cells stamped."""
    path = os.path.join(BASE, stl_name)
    if not os.path.exists(path):
        print(f"      [WARN] {stl_name}  MISSING — skipped")
        return 0
    mesh = trimesh.load_mesh(path)
    bb_lo, bb_hi = mesh.bounds

    # Bounding-box index range in pa (with 1-cell margin).
    ix_lo = max(0,  int(np.floor((bb_lo[0] - x_n[0]) / DX) - 1))
    ix_hi = min(NX, int(np.ceil ((bb_hi[0] - x_n[0]) / DX) + 1))
    iy_lo = max(0,  int(np.floor((bb_lo[1] - y_n[0]) / DX) - 1))
    iy_hi = min(NY, int(np.ceil ((bb_hi[1] - y_n[0]) / DX) + 1))
    iz_lo = max(0,  int(np.floor((bb_lo[2] - z_n[0]) / DX) - 1))
    iz_hi = min(NZ, int(np.ceil ((bb_hi[2] - z_n[0]) / DX) + 1))

    nx_sub = ix_hi - ix_lo
    ny_sub = iy_hi - iy_lo

    xi = x_n[ix_lo:ix_hi]
    yi = y_n[iy_lo:iy_hi]
    zi = z_n[iz_lo:iz_hi]
    XX2, YY2 = np.meshgrid(xi, yi, indexing="ij")
    xy_flat  = np.column_stack([XX2.ravel(), YY2.ravel()])

    pts = np.empty((nx_sub * ny_sub, 3), dtype=np.float64)
    pts[:, 0:2] = xy_flat
    n_inside = 0
    for diz, z_val in enumerate(zi):
        pts[:, 2] = z_val
        inside_flat = mesh.contains(pts)
        if not inside_flat.any():
            continue
        inside_2d = inside_flat.reshape(nx_sub, ny_sub)
        ix_hits, iy_hits = np.where(inside_2d)
        pa[iz_lo + diz,
           iy_lo + iy_hits,
           ix_lo + ix_hits] = marker
        n_inside += int(inside_flat.sum())
    return n_inside


def verify_against_source(source_pa, per_elec_counts, NX, NY, NZ):
    """Compare per-electrode cell counts in the rasterized array against
    the source pa#.  If the source had a leak, the source will have many
    extra "electrode" cells; the rasterized version should have fewer (but
    the same order of magnitude) for the same body."""
    print("\nComparing against source pa#:")
    with open(source_pa, "rb") as f:
        f.read(56)
        src = np.frombuffer(f.read(), dtype="<f8")
    src_counts = {n: int(np.count_nonzero(np.isclose(src, ELECTRODE_BASE + n)))
                  for n in per_elec_counts}
    print(f"  {'elec':<6}{'rasterized':>14}{'SIMION pa#':>14}{'diff':>14}")
    for n in sorted(per_elec_counts):
        r, s = per_elec_counts[n], src_counts[n]
        diff = r - s
        flag = ""
        if s > 0:
            if abs(diff) / max(s, 1) > 0.10:
                flag = "   ← large difference; likely leak in source"
        print(f"  {n:<6}{r:>14,}{s:>14,}{diff:>+14,}{flag}")


if __name__ == "__main__":
    main()
