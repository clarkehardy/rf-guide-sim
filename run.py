#!/usr/bin/env python3
"""run.py  –  Thin entry point that delegates to trapsim.run.

    python run.py                   # default: geometry.yaml, experiment.py, run=1
    python run.py --run 2 --refine  # forward any trapsim.run CLI flags

For programmatic access, prefer the package directly:

    from trapsim import load_geometry, load_experiment
    from trapsim.run import run
"""

from trapsim.run import main

if __name__ == "__main__":
    main()
