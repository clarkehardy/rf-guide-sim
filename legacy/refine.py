"""
refine.py  –  Orchestrate voxelization + C++ Laplace solve → PA files.

Replaces: SIMION Refine + refine_with_dielectric.lua

Steps:
  1. If mask files are stale (or --force-voxelize), run voxelize.py.
  2. Compile solver/laplace if the binary is absent or stale.
  3. Run the C++ solver for all 10 electrodes, writing paulTrap.pa1–pa10.

Usage:
    python refine.py [--force-voxelize] [--omega 1.99] [--max-iter 3000] [--tol 1e-5]
                     [--solver-dir solver] [--out-dir .]
"""

import argparse
import glob
import os
import subprocess
import sys
import time

BASE       = os.path.dirname(os.path.abspath(__file__))
PYTHON     = sys.executable
N_ELEC     = 10


def stl_mtime(base):
    """Return the newest mtime among all .stl files in base."""
    stls = glob.glob(os.path.join(base, "*.stl"))
    return max((os.path.getmtime(p) for p in stls), default=0.0)


def masks_stale(solver_dir, base):
    """True if any mask_*.raw is absent or older than the newest STL."""
    stl_t = stl_mtime(base)
    for e in range(1, N_ELEC + 1):
        p = os.path.join(solver_dir, f"mask_{e}.raw")
        if not os.path.exists(p) or os.path.getmtime(p) < stl_t:
            return True
    return False


def ensure_compiled(solver_dir):
    """Compile solver/laplace if absent or older than laplace.cpp."""
    src = os.path.join(solver_dir, "laplace.cpp")
    exe = os.path.join(solver_dir, "laplace")
    if (not os.path.exists(exe) or
            os.path.getmtime(exe) < os.path.getmtime(src)):
        print("Compiling solver/laplace ...")
        result = subprocess.run(["make", "-C", solver_dir], capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            sys.exit(f"ERROR: compilation failed (return code {result.returncode})")
        print("  Compiled OK.")
    else:
        print("solver/laplace is up-to-date.")


def main():
    ap = argparse.ArgumentParser(description="Refine Paul trap potential arrays.")
    ap.add_argument("--force-voxelize", action="store_true",
                    help="Re-voxelize even if mask files are current")
    ap.add_argument("--omega",    type=float, default=1.99,
                    help="SOR relaxation parameter (default: 1.99)")
    ap.add_argument("--max-iter", type=int,   default=3000,
                    help="Maximum SOR iterations per electrode (default: 3000)")
    ap.add_argument("--tol",      type=float, default=1e-5,
                    help="Convergence tolerance — max |Δφ| (default: 1e-5)")
    ap.add_argument("--solver-dir", default=os.path.join(BASE, "solver"),
                    help="Directory containing laplace.cpp / laplace binary")
    ap.add_argument("--out-dir",    default=BASE,
                    help="Directory to write paulTrap.pa1..pa10 (default: .)")
    args = ap.parse_args()

    solver_dir = args.solver_dir
    out_dir    = args.out_dir

    # ── Step 1: voxelize ──────────────────────────────────────────────────────
    if args.force_voxelize or masks_stale(solver_dir, BASE):
        print("─── Voxelizing STL meshes ───")
        t0 = time.time()
        result = subprocess.run(
            [PYTHON, os.path.join(BASE, "voxelize.py"),
             "--out-dir", solver_dir],
            check=True
        )
        print(f"Voxelization done in {time.time()-t0:.1f} s\n")
    else:
        print("Mask files are current — skipping voxelization  (use --force-voxelize to override)\n")

    # ── Step 2: compile ───────────────────────────────────────────────────────
    ensure_compiled(solver_dir)
    print()

    # ── Step 3: solve ─────────────────────────────────────────────────────────
    print("─── Running Laplace solver ───")
    grid_file = os.path.join(solver_dir, "grid.txt")
    eps_file  = os.path.join(solver_dir, "epsilon.raw")
    exe       = os.path.join(solver_dir, "laplace")
    mask_args = [os.path.join(solver_dir, f"mask_{e}.raw") for e in range(1, N_ELEC+1)]

    for f in [grid_file, eps_file] + mask_args:
        if not os.path.exists(f):
            sys.exit(f"ERROR: required file not found: {f}\n"
                     "Run voxelize.py first (or pass --force-voxelize).")

    cmd = [exe, grid_file, eps_file, out_dir,
           str(args.omega), str(args.max_iter), str(args.tol)] + mask_args

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        sys.exit(f"ERROR: Laplace solver exited with code {result.returncode}")

    elapsed = time.time() - t0
    print(f"\nSolver finished in {elapsed:.1f} s")

    # Quick size check
    expected = 56 + 131 * 91 * 855 * 8   # 56-byte header + NX*NY*NZ float64
    ok = True
    for e in range(1, N_ELEC + 1):
        pa = os.path.join(out_dir, f"paulTrap.pa{e}")
        if not os.path.exists(pa):
            print(f"  WARNING: {pa} not found")
            ok = False
        else:
            sz = os.path.getsize(pa)
            status = "OK" if sz == expected else f"SIZE MISMATCH (got {sz}, expected {expected})"
            print(f"  paulTrap.pa{e}: {sz:,} bytes  {status}")

    if ok:
        print("\n─── Refine complete ───")
    else:
        print("\n─── Refine completed with warnings ───")


if __name__ == "__main__":
    main()
