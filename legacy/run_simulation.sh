#!/usr/bin/env bash
# run_simulation.sh — full pipeline: generate voltages → SIMION fly → animate → visualize
#
# Usage:
#   ./run_simulation.sh                        # voltages_1.csv, trajectories_1.csv
#   ./run_simulation.sh --vol 2 --run 3        # voltages_2.csv, trajectories_3.csv
#   ./run_simulation.sh --no-animate           # skip the animation window
#   ./run_simulation.sh --no-visualize         # skip the 3-D visualize window
#   ./run_simulation.sh --preview-voltages     # show the voltage preview plot
#   ./run_simulation.sh --keep-tmp             # don't delete SIMION's trj*.tmp scratch files
#
# The voltage schedule is always regenerated from generate_voltages.py before
# flying.  Edit that script to change RF frequencies, amplitudes, and DC pulses.
# Edit trap_config.lua to change gas, particle, drag, and trigger parameters.

set -euo pipefail

# ── Fixed paths ───────────────────────────────────────────────────────────────
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$HOME/.venvs/mesh/bin/python3"
WINE="$HOME/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin/wine"
WINEPREFIX="$HOME/Library/Application Support/CrossOver/Bottles/Windows 10 64-bit"
SIMION_EXE='C:\Program Files\SIMION-2024\simion.exe'
IOB='C:\users\crossover\Documents\Research\Nanospheres\SIMION\RF Guide\paulTrap.iob'

# ── Defaults ──────────────────────────────────────────────────────────────────
VOL_FILE=1
RUN_NUM=1
ANIMATE=1
VISUALIZE=1
PREVIEW_VOLTAGES=0
KEEP_TMP=0

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
  grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,1\}//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --vol)             VOL_FILE=$2;      shift 2 ;;
    --run)             RUN_NUM=$2;       shift 2 ;;
    --no-animate)      ANIMATE=0;        shift   ;;
    --no-visualize)    VISUALIZE=0;      shift   ;;
    --preview-voltages) PREVIEW_VOLTAGES=1; shift ;;
    --keep-tmp)        KEEP_TMP=1;       shift   ;;
    -h|--help)         usage ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

# ── Cleanup of SIMION scratch files ───────────────────────────────────────────
# SIMION caches in-memory trajectory data to trj<hex>.tmp files during a fly
# and is supposed to clean them up afterwards.  In --nogui mode it doesn't,
# so we sweep them ourselves on exit.  We snapshot which trj*.tmp existed
# before this run started so we only delete files this run created — any
# trj*.tmp from a parallel SIMION process (if you ever run two at once) is
# left alone.
shopt -s nullglob
TRJ_BEFORE=("$DIR"/trj*.tmp)
shopt -u nullglob

cleanup_trj() {
  [[ $KEEP_TMP -eq 1 ]] && return
  shopt -s nullglob
  local after=("$DIR"/trj*.tmp)
  shopt -u nullglob
  # bash 3.2-compatible "is this file in the before-snapshot?" check via
  # substring match.  trj<hex>.tmp filenames contain no spaces, so the
  # space-padded match is safe.  The `:-` default keeps `set -u` happy
  # when the snapshot is empty (the usual case after a successful run).
  local before_str=" ${TRJ_BEFORE[*]:-} "
  local removed=0
  for f in "${after[@]}"; do
    if [[ "$before_str" != *" $f "* ]]; then
      rm -f "$f" && removed=$((removed + 1))
    fi
  done
  [[ $removed -gt 0 ]] && echo "   Cleaned up $removed SIMION trj*.tmp scratch file(s)"
}
trap cleanup_trj EXIT

echo "━━━ run_simulation.sh  vol=${VOL_FILE}  run=${RUN_NUM} ━━━"

# ── Step 1: Generate voltage schedule ─────────────────────────────────────────
echo ""
echo "── Step 1: Generate voltages_${VOL_FILE}.csv"
PREVIEW_FLAG="--no-preview"
[[ $PREVIEW_VOLTAGES -eq 1 ]] && PREVIEW_FLAG=""
"$PYTHON" "$DIR/generate_voltages.py" --out "$VOL_FILE" $PREVIEW_FLAG

# ── Step 2: Run SIMION headless ───────────────────────────────────────────────
echo ""
echo "── Step 2: SIMION fly  (voltages_${VOL_FILE}.csv → trajectories_${RUN_NUM}.csv)"
env SIMION_VOL_FILE="$VOL_FILE" \
    SIMION_RUN_NUM="$RUN_NUM" \
    CX_BOTTLE="Windows 10 64-bit" \
    WINEPREFIX="$WINEPREFIX" \
  "$WINE" "$SIMION_EXE" --nogui --noprompt --quiet fly "$IOB" 2>&1

# ── Step 3: Animate ───────────────────────────────────────────────────────────
if [[ $ANIMATE -eq 1 ]]; then
  echo ""
  echo "── Step 3: Animate"
  TRAJ="$DIR/trajectories_${RUN_NUM}.csv"
  VOLT="$DIR/voltages_${VOL_FILE}.csv"
  if [[ -f "$TRAJ" ]]; then
    "$PYTHON" "$DIR/animate.py" --traj "$TRAJ" --volt "$VOLT"
  else
    echo "WARNING: trajectory file not found: $TRAJ"
  fi
fi

# ── Step 4: Visualize (3-D) ───────────────────────────────────────────────────
if [[ $VISUALIZE -eq 1 ]]; then
  echo ""
  echo "── Step 4: Visualize"
  TRAJ="$DIR/trajectories_${RUN_NUM}.csv"
  if [[ -f "$TRAJ" ]]; then
    "$PYTHON" "$DIR/visualize.py" --traj "$TRAJ"
  else
    echo "WARNING: trajectory file not found: $TRAJ"
  fi
fi

echo ""
echo "━━━ Done ━━━"
