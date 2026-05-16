#!/usr/bin/env python3
"""
run_simulation.py  –  Full pipeline: generate voltages → fly → animate → visualize

Usage:
    python run_simulation.py                        # voltages_1.csv, trajectories_1.csv
    python run_simulation.py --vol 2 --run 3        # voltages_2.csv, trajectories_3.csv
    python run_simulation.py --no-animate           # skip the animation window
    python run_simulation.py --no-visualize         # skip the 3-D visualize window
    python run_simulation.py --no-fly               # skip particle integration
    python run_simulation.py --no-refine            # skip potential-array refinement
    python run_simulation.py --preview-voltages     # show the voltage preview plot
    python run_simulation.py --workers N            # use N worker processes for fly.py
    python run_simulation.py --refine               # force a full refine before flying

Replaces run_simulation.sh from the legacy/ directory.
"""

import argparse
import os
import subprocess
import sys

BASE   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

def run(cmd, **kwargs):
    print("$", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(f"Command failed (exit {result.returncode})")

def main():
    ap = argparse.ArgumentParser(description="Paul trap simulation pipeline.")
    ap.add_argument("--vol",             type=int, default=1,
                    help="Voltage file number N (writes voltages_N.csv)")
    ap.add_argument("--run",             type=int, default=1,
                    help="Run number N (writes trajectories_N.csv)")
    ap.add_argument("--workers",         type=int, default=None,
                    help="Worker processes for fly.py (default: all CPUs)")
    ap.add_argument("--no-animate",      action="store_true")
    ap.add_argument("--no-visualize",    action="store_true")
    ap.add_argument("--no-fly",          action="store_true",
                    help="Skip particle integration (use existing trajectories_N.csv)")
    ap.add_argument("--no-refine",       action="store_true",
                    help="Skip refine step (PA files assumed current)")
    ap.add_argument("--refine",          action="store_true",
                    help="Force a full refine (voxelize + solve) before flying")
    ap.add_argument("--preview-voltages", action="store_true")
    args = ap.parse_args()

    print(f"━━━ run_simulation.py  vol={args.vol}  run={args.run} ━━━\n")

    # ── Step 0: Refine (optional) ─────────────────────────────────────────────
    if args.refine and not args.no_refine:
        print("── Step 0: Refine potential arrays")
        refine_cmd = [PYTHON, os.path.join(BASE, "refine.py")]
        if args.refine:
            refine_cmd.append("--force-voxelize")
        run(refine_cmd)
        print()

    # ── Step 1: Generate voltage schedule ────────────────────────────────────
    print(f"── Step 1: Generate voltages_{args.vol}.csv")
    volt_cmd = [PYTHON, os.path.join(BASE, "generate_voltages.py"), "--out", str(args.vol)]
    if not args.preview_voltages:
        volt_cmd.append("--no-preview")
    run(volt_cmd)
    print()

    # ── Step 2: Fly particles ─────────────────────────────────────────────────
    if not args.no_fly:
        print(f"── Step 2: Fly  (voltages_{args.vol}.csv → trajectories_{args.run}.csv)")
        fly_cmd = [PYTHON, os.path.join(BASE, "fly.py"),
                   "--vol", str(args.vol), "--run", str(args.run)]
        if args.workers is not None:
            fly_cmd += ["--workers", str(args.workers)]
        run(fly_cmd)
        print()

    # ── Step 3: Animate ───────────────────────────────────────────────────────
    if not args.no_animate:
        traj = os.path.join(BASE, f"trajectories_{args.run}.csv")
        volt = os.path.join(BASE, f"voltages_{args.vol}.csv")
        print("── Step 3: Animate")
        if os.path.exists(traj):
            run([PYTHON, os.path.join(BASE, "animate.py"),
                 "--traj", traj, "--volt", volt])
        else:
            print(f"WARNING: trajectory file not found: {traj}")
        print()

    # ── Step 4: Visualize ─────────────────────────────────────────────────────
    if not args.no_visualize:
        traj = os.path.join(BASE, f"trajectories_{args.run}.csv")
        print("── Step 4: Visualize")
        if os.path.exists(traj):
            run([PYTHON, os.path.join(BASE, "visualize.py"), "--traj", traj])
        else:
            print(f"WARNING: trajectory file not found: {traj}")
        print()

    print("━━━ Done ━━━")

if __name__ == "__main__":
    main()
