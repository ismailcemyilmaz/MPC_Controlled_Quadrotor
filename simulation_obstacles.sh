#!/bin/bash

# ============================================================
# Quadrotor MPC — Obstacle Avoidance Simulation
# ============================================================

set -e

middleware=pocolibs
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
world="${1:-${SCRIPT_DIR}/worlds/quad_obstacles.world}"

export GZ_SIM_RESOURCE_PATH="${SCRIPT_DIR}/model:${GZ_SIM_RESOURCE_PATH:-}"

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

echo "=== Starting obstacle avoidance simulation ==="

h2 init

pkill genomixd 2>/dev/null || true
sleep 0.3

genomixd &
pids="$pids $!"
sleep 1

for c in $components; do
    echo "Starting $c-$middleware"
    $c-$middleware &
    pids="$pids $!"
    sleep 1
done

if [ ! -f "$world" ]; then
    echo "Cannot find world file: $world"
    exit 1
fi

echo "Starting Gazebo..."
gz sim "$world" &
pids="$pids $!"

echo ""
echo "=== Stack ready (obstacle world) ==="
echo ""
echo "Terminal 2:"
echo "  python3 -i quadrotor_mpc_client_v3.py"
echo ""
echo "Then:"
echo "  >>> setup()"
echo "  >>> slalom()"
echo ""

trap cleanup CHLD
wait
