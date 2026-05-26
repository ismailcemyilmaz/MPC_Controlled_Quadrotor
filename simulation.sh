#!/bin/bash

# ============================================================
# Assignment 07-mpc — simulation.sh
# Quadrotor MPC — Flip / Loop
# ============================================================

set -e

usage() {
    echo ""
    echo "Usage: $(basename $0)"
    echo ""
    exit 1
}

# Default settings
middleware=pocolibs
world="${TK3LAB_WS}/gazebo/worlds/quad.world"

# Components WITHOUT nhfc
components="
  rotorcraft
  pom
  optitrack
"

pids=

cleanup() {
    trap - 0 INT CHLD

    set +e

    echo ""
    echo "=== Cleaning up ==="

    kill $pids 2>/dev/null
    wait

    h2 end 2>/dev/null

    exit 0
}

trap cleanup 0 INT

echo "=== Starting MPC simulation stack ==="

# ------------------------------------------------
# Initialize middleware
# ------------------------------------------------

h2 init

# ------------------------------------------------
# Start genomixd
# ------------------------------------------------

pkill genomixd 2>/dev/null || true
sleep 0.3

genomixd &
pids="$pids $!"

sleep 1

# ------------------------------------------------
# Start GenoM3 components
# ------------------------------------------------

for c in $components; do
    echo "Starting $c-$middleware"
    $c-$middleware &
    pids="$pids $!"
    sleep 1
done

# ------------------------------------------------
# Check world file
# ------------------------------------------------

if [ ! -f "$world" ]; then
    echo "Cannot find world file:"
    echo "$world"
    exit 1
fi

# ------------------------------------------------
# Start Gazebo
# ------------------------------------------------

echo "Starting Gazebo..."
gz sim "$world" &
pids="$pids $!"

echo ""
echo "=== Stack ready ==="
echo ""
echo "Terminal 2:"
echo "python3 -i quadrotor_mpc_client.py"
echo ""
echo "Then:"
echo ">>> simulation('backflip')"
echo ""

trap cleanup CHLD
wait