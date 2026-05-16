"""
voxelize.py  –  Convert STL electrode/dielectric meshes to voxel grids.

Reads the electrode-to-STL mapping from paulTrap.processed.gem (in legacy/),
builds one binary mask per electrode and a dielectric permittivity array,
then writes them to solver/ for the C++ Laplace solver to consume.

Output files (in solver/ subdirectory):
    grid.txt          — "NX NY NZ DX TX TY TZ" on one line
    mask_1.raw        — flat uint8 array, shape NZ×NY×NX; 1 = inside electrode 1
    ...
    mask_10.raw
    epsilon.raw       — flat float64 array, shape (NZ-1)×(NY-1)×(NX-1); ε_r values

Usage:
    python voxelize.py [--gem legacy/paulTrap.processed.gem] [--out-dir solver]

Requires: trimesh  (pip install trimesh in ~/.venvs/mesh)
"""

import argparse
import os
import re
import struct

import numpy as np

try:
    import trimesh
except ImportError:
    raise SystemExit(
        "trimesh is required.  Install it with:\n"
        "  ~/.venvs/mesh/bin/pip install trimesh"
    )

BASE     = os.path.dirname(os.path.abspath(__file__))
GEM_FILE = os.path.join(BASE, "legacy", "paulTrap.processed.gem")

# ── Grid defaults (must match paulTrap.processed.gem: pa_define(131,91,855,...,0.5,...))
NX_DEFAULT, NY_DEFAULT, NZ_DEFAULT = 131, 91, 855
DX_DEFAULT = 0.5   # mm
# locate(TX, TY, TZ) in the GEM file — offset from GEM index (0,0,0) to Fusion world
TX_DEFAULT, TY_DEFAULT, TZ_DEFAULT = 25.0, 8.0, 132.0

# ── Dielectric configuration (matches generate_dielectric_pa.py) ──────────────
EPSILON_GLASS = 3.0   # ε_r for lenses and lens holder (fused silica + PEEK ≈ 3.0-3.8)
DIELECTRIC_STLS = [
    "trapping_lens.stl",
    "collection_lens.stl",
    "lens_holder.stl",
]


def parse_gem(gem_path):
    """Return (NX, NY, NZ, DX, TX, TY, TZ, electrode_stls).

    electrode_stls: dict {electrode_number: [stl_path, ...]} (1-indexed).
    Parses only pa_define(...) and e(N) { stl(...) } blocks.
    """
    NX, NY, NZ = NX_DEFAULT, NY_DEFAULT, NZ_DEFAULT
    DX         = DX_DEFAULT
    TX, TY, TZ = TX_DEFAULT, TY_DEFAULT, TZ_DEFAULT
    electrode_stls = {}

    if not os.path.exists(gem_path):
        print(f"WARNING: GEM file not found at {gem_path}; using defaults.")
        return NX, NY, NZ, DX, TX, TY, TZ, electrode_stls

    with open(gem_path) as f:
        text = f.read()

    # pa_define(NX, NY, NZ, ..., DX, ...)
    m = re.search(r'pa_define\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)[^)]*?,\s*([\d.]+)', text)
    if m:
        NX, NY, NZ = int(m.group(1)), int(m.group(2)), int(m.group(3))
        DX = float(m.group(4))

    # locate(TX, TY, TZ)
    m = re.search(r'locate\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', text)
    if m:
        TX, TY, TZ = float(m.group(1)), float(m.group(2)), float(m.group(3))

    # e(N) { stl("path") stl("path") ... }
    for block in re.finditer(r'e\s*\(\s*(\d+)\s*\)\s*\{([^}]*)\}', text, re.DOTALL):
        en   = int(block.group(1))
        body = block.group(2)
        stls = re.findall(r'stl\s*\(\s*"([^"]+)"', body)
        # Normalise Windows paths to local paths
        local_stls = [
            os.path.join(BASE, os.path.basename(s.replace("\\", "/")))
            for s in stls
        ]
        electrode_stls[en] = local_stls

    return NX, NY, NZ, DX, TX, TY, TZ, electrode_stls


def voxelize_stl(mesh, NX, NY, NZ, DX, TX, TY, TZ, label=""):
    """Return a flat uint8 array (1 = inside mesh) with shape NZ×NY×NX.

    Processes one Z-slice at a time to keep peak memory bounded.
    Fusion world origin of node (0,0,0) is (-TX, -TY, -TZ) mm.
    """
    mask   = np.zeros(NZ * NY * NX, dtype=np.uint8)
    x_node = np.arange(NX) * DX - TX
    y_node = np.arange(NY) * DX - TY
    z_node = np.arange(NZ) * DX - TZ

    bb_lo, bb_hi = mesh.bounds
    # Restrict to nodes that could be inside the mesh bounding box
    ix_lo = max(0, int(np.floor((bb_lo[0] + TX) / DX)))
    ix_hi = min(NX, int(np.ceil( (bb_hi[0] + TX) / DX)) + 1)
    iy_lo = max(0, int(np.floor((bb_lo[1] + TY) / DX)))
    iy_hi = min(NY, int(np.ceil( (bb_hi[1] + TY) / DX)) + 1)
    iz_lo = max(0, int(np.floor((bb_lo[2] + TZ) / DX)))
    iz_hi = min(NZ, int(np.ceil( (bb_hi[2] + TZ) / DX)) + 1)

    xi = x_node[ix_lo:ix_hi]
    yi = y_node[iy_lo:iy_hi]
    nx_sub = len(xi)
    ny_sub = len(yi)
    if nx_sub == 0 or ny_sub == 0:
        print(f"    {label}: bounding box outside grid — skipped")
        return mask

    XX, YY = np.meshgrid(xi, yi, indexing='ij')
    pts = np.empty((nx_sub * ny_sub, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    n_inside = 0
    for diz, iz in enumerate(range(iz_lo, iz_hi)):
        pts[:, 2] = z_node[iz]
        inside = mesh.contains(pts)
        if not inside.any():
            continue
        inside_2d = inside.reshape(nx_sub, ny_sub)
        ix_hits, iy_hits = np.where(inside_2d)
        flat_idx = iz * NY * NX + (iy_lo + iy_hits) * NX + (ix_lo + ix_hits)
        mask[flat_idx] = 1
        n_inside += int(inside.sum())

    print(f"    {label}: bbox {bb_lo} → {bb_hi}  |  {n_inside} voxels inside")
    return mask


def build_electrode_masks(electrode_stls, NX, NY, NZ, DX, TX, TY, TZ):
    """Return list of 10 flat uint8 arrays (one per electrode, 1-indexed)."""
    n_elec  = max(electrode_stls.keys()) if electrode_stls else 10
    masks   = [np.zeros(NX * NY * NZ, dtype=np.uint8) for _ in range(n_elec + 1)]

    for en in sorted(electrode_stls.keys()):
        stl_paths = electrode_stls[en]
        print(f"  Electrode {en}: {len(stl_paths)} STL(s)")
        for stl_path in stl_paths:
            if not os.path.exists(stl_path):
                print(f"    WARNING: {stl_path} not found — skipping")
                continue
            mesh = trimesh.load_mesh(stl_path)
            m    = voxelize_stl(mesh, NX, NY, NZ, DX, TX, TY, TZ,
                                label=os.path.basename(stl_path))
            masks[en] |= m

    return masks[1:]   # return indices 0..n_elec-1  (electrode 1 at index 0)


def build_epsilon(NX, NY, NZ, DX, TX, TY, TZ):
    """Return flat float64 array of shape (NZ-1)×(NY-1)×(NX-1) with ε_r values."""
    NXc, NYc, NZc = NX - 1, NY - 1, NZ - 1
    epsilon = np.ones(NZc * NYc * NXc, dtype=np.float64)

    x_c = (np.arange(NXc) + 0.5) * DX - TX
    y_c = (np.arange(NYc) + 0.5) * DX - TY
    z_c = (np.arange(NZc) + 0.5) * DX - TZ

    for stl_name in DIELECTRIC_STLS:
        stl_path = os.path.join(BASE, stl_name)
        if not os.path.exists(stl_path):
            print(f"  Dielectric WARNING: {stl_name} not found — skipping")
            continue

        mesh = trimesh.load_mesh(stl_path)
        bb_lo, bb_hi = mesh.bounds
        print(f"  Dielectric {stl_name}: {mesh.vertices.shape[0]} verts  "
              f"bbox {bb_lo} → {bb_hi}")

        ix_lo = max(0, int(np.floor((bb_lo[0] + TX) / DX - 1)))
        ix_hi = min(NXc, int(np.ceil( (bb_hi[0] + TX) / DX + 1)))
        iy_lo = max(0, int(np.floor((bb_lo[1] + TY) / DX - 1)))
        iy_hi = min(NYc, int(np.ceil( (bb_hi[1] + TY) / DX + 1)))
        iz_lo = max(0, int(np.floor((bb_lo[2] + TZ) / DX - 1)))
        iz_hi = min(NZc, int(np.ceil( (bb_hi[2] + TZ) / DX + 1)))

        xi = x_c[ix_lo:ix_hi]
        yi = y_c[iy_lo:iy_hi]
        nx_sub = len(xi)
        ny_sub = len(yi)
        if nx_sub == 0 or ny_sub == 0:
            continue

        XX, YY = np.meshgrid(xi, yi, indexing='ij')
        pts = np.empty((nx_sub * ny_sub, 3), dtype=np.float64)
        pts[:, 0] = XX.ravel()
        pts[:, 1] = YY.ravel()

        n_inside = 0
        for iz in range(iz_lo, iz_hi):
            pts[:, 2] = z_c[iz]
            inside = mesh.contains(pts)
            if not inside.any():
                continue
            inside_2d = inside.reshape(nx_sub, ny_sub)
            ix_hits, iy_hits = np.where(inside_2d)
            flat_idx = iz * NYc * NXc + (iy_lo + iy_hits) * NXc + (ix_lo + ix_hits)
            epsilon[flat_idx] = EPSILON_GLASS
            n_inside += int(inside.sum())
        print(f"    {n_inside} cells marked ε_r = {EPSILON_GLASS}")

    return epsilon


def main():
    ap = argparse.ArgumentParser(description="Voxelize STL meshes for the Laplace solver.")
    ap.add_argument("--gem",     default=GEM_FILE, help="Path to paulTrap.processed.gem")
    ap.add_argument("--out-dir", default=os.path.join(BASE, "solver"), help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Parsing GEM file ...")
    NX, NY, NZ, DX, TX, TY, TZ, electrode_stls = parse_gem(args.gem)
    print(f"  Grid: {NX}×{NY}×{NZ}  DX={DX} mm  locate=({TX},{TY},{TZ})")
    print(f"  Electrodes defined: {sorted(electrode_stls.keys())}")

    # Write grid.txt
    grid_path = os.path.join(args.out_dir, "grid.txt")
    with open(grid_path, "w") as f:
        f.write(f"{NX} {NY} {NZ} {DX} {TX} {TY} {TZ}\n")
    print(f"Written {grid_path}")

    # Build and write electrode masks
    print("\nVoxelizing electrodes ...")
    masks = build_electrode_masks(electrode_stls, NX, NY, NZ, DX, TX, TY, TZ)
    for e_idx, mask in enumerate(masks):
        path = os.path.join(args.out_dir, f"mask_{e_idx+1}.raw")
        mask.tofile(path)
        n_set = int(mask.sum())
        print(f"  Written {path}  ({n_set} voxels)")

    # Build and write dielectric array
    print("\nVoxelizing dielectrics ...")
    epsilon = build_epsilon(NX, NY, NZ, DX, TX, TY, TZ)
    eps_path = os.path.join(args.out_dir, "epsilon.raw")
    epsilon.tofile(eps_path)
    n_diel = int((epsilon > 1.0).sum())
    print(f"Written {eps_path}  ({n_diel} cells with ε_r > 1)")

    print("\nVoxelization complete.")


if __name__ == "__main__":
    main()
