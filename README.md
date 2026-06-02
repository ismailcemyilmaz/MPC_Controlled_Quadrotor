# Quadrotor NMPC — Obstacle-Aware Local Planner

An **acados SQP-RTI** based Nonlinear MPC controller for a quadrotor.

Includes a 5th-order polynomial global trajectory generator, obstacle avoidance with soft constraints, and Gazebo simulation integration.

**Politecnico di Milano — Aerial Robotics 2025-26**

---

## Table of Contents

* [Overview](#overview)
* [System Requirements](#system-requirements)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Configuration](#configuration)
* [Testing and Usage](#testing-and-usage)
* [Obstacle Avoidance](#obstacle-avoidance)
* [Log Analysis](#log-analysis)

---

## Overview

```
State  x ∈ R^13  :  [px, py, pz,  vx, vy, vz,  qw, qx, qy, qz,  p, q, r]
Input  u ∈ R^4   :  [f_total, τx, τy, τz]
```

| Component | Description |
| --- | --- |
| `local_planner_mpc.py` | Obstacle-aware NMPC (acados SQP-RTI, N=20, Ts=50ms) |
| `mpc_solver.py` | Core QuadrotorMPC (landing cone constraints) |
| `quadrotor_model.py` | CasADi/acados dynamic model (quaternion-based) |
| `global_planner.py` | WaypointTrajectory, BackflipTrajectory, APFTrajectory |
| `perception.py` | PerceptionManager (Gazebo GT / 2D Lidar / Static) |
| `quadrotor_mpc_client_v3.py` | Main control loop and public API |
| `plot_mpc_log.py` | Log visualization and auto-save to `plots/` |
| `plot_backflip_paper.py` | Paper-ready backflip analysis plots |
| `plot_apf_field.py` | APF force field + potential visualization |

**Features:**

* Quaternion-based NMPC (no Euler angle singularities)
* 1.0s prediction horizon (N=20, Ts=50ms), SQP-RTI single iteration
* APF-based obstacle avoidance: offline path planning + MPC tracking
* Backflip: Lupashin 5-phase bang-coast-bang with rate-kill recovery
* Landing cone constraint: vz + α·z ≥ 0 (prevents hard landings)
* "+" configuration motor mixer with feasibility checking
* Quintic polynomial multi-waypoint trajectory generation
* Shared log session: takeoff + flight + landing in a single `.npz` file

---

## System Requirements

| Requirement | Version |
| --- | --- |
| Docker Engine | ≥ 24 |
| Python | ≥ 3.10 |
| acados | ≥ 0.3 |
| CasADi | ≥ 3.6 |
| NumPy | ≥ 1.24 |
| Gazebo | Ionic (tk3lab Docker image) |
| GenoM3 | pocolibs middleware |

Docker image: `art/tk3lab:ionic-0.2`

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url> mpc-quadrotor
cd mpc-quadrotor
```

### 2. Start the Docker environment (tk3lab)

The project runs inside the tk3lab Docker container. The host `~/tk3lab-ws/` maps to `/shared-workspace/` inside Docker.

### 3. Install Dependencies (first time only)

Dependencies are installed to `/shared-workspace/` so they persist across Docker restarts.

```bash
cd /shared-workspace/src/mpc-quadrotor
bash setup_deps.sh install
```

This installs CasADi, NumPy, and acados into `/shared-workspace/pip-packages/` and `/shared-workspace/acados/`.

### 4. Environment Setup (after each Docker restart)

Run this once per Docker session, before using the MPC project:

```bash
source /shared-workspace/src/mpc-quadrotor/env_setup.sh
```

---

## Project Structure

```
mpc-quadrotor/
├── quadrotor_mpc_client_v3.py   # Main API (hover, slalom, slalom_reactive, backflip)
├── local_planner_mpc.py         # Obstacle-aware LocalPlannerMPC
├── mpc_solver.py                # Core QuadrotorMPC
├── quadrotor_model.py           # acados dynamic model
├── global_planner.py            # WaypointTrajectory, BackflipTrajectory, APFTrajectory
├── perception.py                # PerceptionManager (3 levels)
├── plot_mpc_log.py              # Log visualizer (auto-saves to plots/)
├── plot_apf_field.py            # APF force field + potential visualization
├── plot_backflip.py             # Backflip analysis plots (detailed)
├── plot_backflip_paper.py       # Backflip paper-ready plots (single column)
├── simulation.sh                # Basic simulation stack (no obstacles)
├── simulation_obstacles.sh      # Obstacle avoidance simulation stack
├── worlds/
│   ├── quad.world               # Empty world
│   ├── quad_obstacles.world     # 3 cylindrical obstacles
│   └── quad_obstacles_dense.world  # 5-obstacle alternating slalom
├── model/
│   ├── mrsim-quadrotor-lidar/
│   │   ├── model.sdf            # Quadrotor with Lidar (+ configuration)
│   │   └── model.config
│   └── mrsim-rotor/
│       ├── model.sdf
│       └── model.config
├── plots/                       # Auto-saved plot images
├── acados_generated/            # Auto-generated solver code
└── logs/mpc/                    # Test logs (.npz)
```

## Architecture

<img width="2720" height="2320" alt="architecture_diagram" src="https://github.com/user-attachments/assets/fd65240d-b5bc-498c-8834-c995fa556fd6" />
<img width="2720" height="2400" alt="control_loop_dataflow" src="https://github.com/user-attachments/assets/4bb141d7-4a50-4ee8-90d7-7f662623546c" />

---

## Configuration

### Physical Constants

```python
MASS    = 1.280        # kg (base 1.0 + 4 rotors × 0.07)
I_DIAG  = (22.916e-3, 22.916e-3, 22.132e-3)  # kg·m²
ARM_LEN = 0.23         # m
KF      = 6.5e-4       # N/(rad/s)²
KM      = 1e-5         # Nm/(rad/s)²
```

### MPC Parameters

```python
MPC_N  = 20            # Prediction horizon steps
MPC_TS = 0.05          # Sampling time [s] → 1.0s horizon

_LOCAL_MPC_KWARGS = dict(
    n_obs_max=5,                     # Max simultaneous obstacles
    R_drone=0.30,                    # Drone collision radius [m]
    W_obs=10000.0,                   # Obstacle slack penalty weight
)
```

### Hover Gains

Validated for stable hover (z std=0.21m, zero negative omega²):

```python
Q_pos=5.0,  Q_vel=3.0,  Q_att=1.5,  Q_omega=25.0,  Q_omega_r=6.0,
P_scale=5.0, R_f=0.01,  R_tau=0.10, R_tau_z=0.20,
tau_max=0.20, tau_z_max=0.06, f_min=0.40*MASS*G, f_max_scale=2.5,
alpha_land=2.0, W_land=500.0
```

### Slalom / Waypoint Gains (current)

Validated for obstacle avoidance slalom (3 obstacles, max_vel=1.5, qw>0.91, 68% torque sat):

```python
Q_pos=5.0,  Q_vel=3.0,  Q_att=1.5,  Q_omega=25.0,  Q_omega_r=6.0,
P_scale=5.0, R_f=0.01,  R_tau=0.10, R_tau_z=0.20,
tau_max=0.25, tau_z_max=0.06, f_min=0.40*MASS*G, f_max_scale=2.5,
alpha_land=2.0, W_land=500.0
```

Difference from hover: `tau_max=0.25` (was 0.20) — 25% more torque authority for turns.

### Perception Levels

| Level | Source | Description |
| --- | --- | --- |
| 1 | Gazebo ground truth | Reads obstacle poses from Gazebo (requires model names) |
| 2 | 2D Lidar | Clusters lidar scan into obstacles |
| 3 | Static | Manually registered obstacle positions |

---

## Testing and Usage

### Terminal 1 — Start Simulation Stack

```bash
cd ~/tk3lab-ws/src/mpc-quadrotor

# Basic (no obstacles):
bash simulation.sh

# 3-obstacle world:
bash simulation_obstacles.sh

# 5-obstacle dense slalom:
bash simulation_obstacles.sh worlds/quad_obstacles_dense.world
```

### Terminal 2 — Python Control

```bash
cd ~/tk3lab-ws/src/mpc-quadrotor
python3 -i quadrotor_mpc_client_v3.py
```

### Hover Test

```python
>>> setup()
>>> hover(0, 0, 4, T_hover=10, log_tag='hover')
```

### Waypoint Following

```python
>>> setup()
>>> follow_waypoints([[0,0,3], [4,0.5,3], [8,-0.5,3], [12,0,3]],
...                   max_vel=0.8, log_tag='waypoints')
```

### Obstacle Avoidance — Reactive APF (recommended)

```python
# 3-obstacle world (simulation_obstacles.sh):
>>> setup()
>>> slalom_reactive()

# 5-obstacle dense world with lidar perception:
>>> setup(perception_level=2)
>>> slalom_reactive(use_perception=True, max_vel=2.5)
```

Real-time APF at each MPC step. No pre-planned waypoints — drone reacts to obstacles online.

With `use_perception=True`, obstacles are detected via 2D lidar (LaserScan → DBSCAN clustering → nearest-neighbor data association) instead of hardcoded positions. Requires `perception_level=2` in `setup()` to enable lidar-based perception.

### Obstacle Avoidance — Offline APF

```python
# Use simulation_obstacles.sh world
>>> setup()
>>> slalom()
```

Pre-plans full path with APF, fits quintic polynomials, MPC tracks. Faster but less reliable on sharp turns.

### Backflip

```python
# Use simulation.sh (no obstacles, needs altitude clearance)
>>> setup()
>>> backflip()          # basic version
>>> backflip_ilc()      # tuned version with PD position-feedback recovery
```

Lupashin 5-phase bang-coast-bang backflip:
1. MPC climb to 10m and hover
2. Open-loop pop-up impulse (2.0×mg for 0.40s)
3. Open-loop flip: accel(+τ) → coast(freefall) → decel(-τ), body-rate integrated angle tracking with dynamic decel start
4. SO(3) quaternion-based recovery with velocity + position damping
5. MPC return to origin and landing

`backflip_ilc()` uses tuned parameters (f_accel=8.77N, f_decel=8.81N) and adds position feedback (K_pos=0.05) to the SO(3) recovery phase, pulling the drone back toward its pre-flip hover position.

**Typical results (good flip exit, qw > 0.95):**

| Metric | Value |
| --- | --- |
| Flip duration | ~0.75s |
| Peak pitch rate | ~9.3 rad/s |
| Attitude error at exit | 14–24° |
| Max lateral drift | 2–3m |
| Settled drift | 0.5–0.8m |
| Altitude loss | 0.5–1.0m |
| Recovery time | 3–5s |

### Manual Control

```python
>>> setup()
>>> start()
>>> set_position(0, 0, 2, T_hold=5)
>>> set_position(3, 1, 2, T_hold=5)
>>> landing()
```

### Emergency Stop

```python
>>> stop()
```

---

## Obstacle Avoidance

### Architecture: APF + MPC

Two-layer approach combining Artificial Potential Fields (APF) for path planning with NMPC for trajectory tracking (Khatib 1986, Ge & Cui 2000):

**Reactive mode (`slalom_reactive`)** — recommended:
```
Every 50ms:  Current Position → APF Force → Velocity Reference → MPC Horizon → Motor Commands
```
No pre-planned path. APF computes velocity direction at each MPC step. Extends to lidar-based perception.

**Offline mode (`slalom`):**
```
APF Path (offline) → Downsample → Quintic Polynomial → MPC Tracker (online)
```
Pre-plans full path, fits smooth trajectory, MPC tracks. Faster peak speed but less reliable on sharp turns.

### APF Force Model

Attractive force pulls toward goal, repulsive force pushes away from obstacles with a rotational tangent component:

```
F_att = k_att × (goal - pos) / ||goal - pos||        (capped at k_att)
F_rep = k_rep × (1/margin - 1/d0) × (1/margin²) × (radial + 0.5 × tangent)
```

Tangent direction per obstacle: `sign(cross(line_dir, obs_offset))` — obstacle left of start→goal line → pass right, and vice versa. Creates natural slalom pattern.

### World Layouts

**Original (`quad_obstacles.world`) — 3 obstacles:**

```
                       obs2(6, 1.5)
Start(0,0)  →  obs1(3,0)  ────────────────  obs3(9,-1)  →  Goal(12,0)
                 🔴 r=0.4      🟠 r=0.4        🔵 r=0.4
```

**Dense (`quad_obstacles_dense.world`) — 5 obstacles, alternating slalom:**

```
     y
 1.5┊       ○2(6)          ○4(12)
    ┊
  0 ─S┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄G(18,0)→ x
    ┊
-1.5┊    ○1(3)       ○3(9)        ○5(15)
    ┊
    0  3  6  9  12  15  18
```

All obstacles: radius=0.4m, height=3.0m. Obstacles alternate y=±1.5 to force zigzag weaving. Margin = 1.5 - 0.4 - 0.5 = 0.60m per side.

### APF Parameters

| Parameter | Original (3-obs) | Dense (5-obs) |
| --- | --- | --- |
| k_att | 1.0 | 1.0 |
| k_rep | 0.8 | 0.8 |
| d0 | 2.5 m | 2.5 m |
| R_drone (APF) | 0.65 m | 0.50 m |
| tangent_weight | 0.5 | 0.5 |
| max_vel | 2.5 m/s | 2.5 m/s |
| goal | (12, 0) | (18, 0) |

### Reactive Slalom Results — Original World

| Metric | Value |
| --- | --- |
| Duration | 21.0s |
| MPC solve time | 8.7ms avg |
| Obstacles avoided | 3 |
| Min distance to obs1 | 1.28m |
| Min distance to obs2 | 2.47m |
| Min distance to obs3 | 1.42m |
| Max speed | 2.83 m/s |
| Torque saturation | 49% |
| qw min | 0.987 (stable) |

### Reactive Slalom Results — Dense World (Lidar Perception)

| Metric | Value |
| --- | --- |
| Duration | ~25s |
| Obstacles | 5 (detected via 2D lidar) |
| Avg speed | 1.28 m/s |
| Obstacle margins | 0.9–1.9m |
| Perception | DBSCAN clustering + nearest-neighbor association |

### Perception Pipeline (Lidar Mode)

```
LaserScan (360° 2D) → Polar-to-Cartesian → DBSCAN Clustering → (position, radius)
                                                ↓
                               Nearest-Neighbor Data Association (1.5m threshold)
                                                ↓
                                    APF Sign Stability (cross-product)
```

Filtering: radius > 1.0m rejected, obstacles behind drone (x < drone_x - 1.5) ignored, |y| > 5.0m rejected.

### Visualization

APF force field and potential surface visualization:

```bash
python3 plot_apf_field.py                                         # dense world, lidar log
python3 plot_apf_field.py --world original                        # original 3-obstacle world
python3 plot_apf_field.py logs/mpc/slalom/mpc_log.npz --world original  # offline log
```

---

## Log Analysis

Logs saved to `logs/mpc/<log_tag>/mpc_log.npz`.

```python
import numpy as np

data = np.load('logs/mpc/slalom/mpc_log.npz')
t        = data['t']          # (N,)     Time [s]
x        = data['x']          # (N, 13)  State vector
u        = data['u']          # (M, 4)   Control input [f, τx, τy, τz]
xref     = data['xref']       # (N, 13)  Reference trajectory
mpc_times= data['mpc_times']  # (M,)     MPC solver times [ms]
n_obs    = data['n_obs']      # (M,)     Number of active obstacles
```

### Plot Generation

```bash
# Inside Docker:
python3 plot_mpc_log.py logs/mpc/slalom/mpc_log.npz

# From host (plots auto-saved to mpc-quadrotor/plots/):
python3 plot_mpc_log.py --save logs/mpc/slalom/mpc_log.npz
```

---

## References

1. O. Khatib, "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots," *The International Journal of Robotics Research*, vol. 5, no. 1, pp. 90–98, 1986. [DOI: 10.1177/027836498600500106](https://doi.org/10.1177/027836498600500106)

2. S. S. Ge and Y. J. Cui, "New Potential Functions for Mobile Robot Path Planning," *IEEE Transactions on Robotics and Automation*, vol. 16, no. 5, pp. 615–620, 2000. [DOI: 10.1109/70.880813](https://doi.org/10.1109/70.880813)

3. S. Lupashin, A. Schöllig, M. Sherback, and R. D'Andrea, "A Simple Learning Strategy for High-Speed Quadrocopter Multi-Flips," in *Proc. IEEE International Conference on Robotics and Automation (ICRA)*, pp. 1642–1648, 2010. [DOI: 10.1109/ROBOT.2010.5509452](https://doi.org/10.1109/ROBOT.2010.5509452)

4. R. Verschueren, G. Frison, D. Kouzoupis, et al., "acados — A Modular Open-Source Framework for Fast Embedded Optimal Control," *Mathematical Programming Computation*, vol. 14, no. 1, pp. 147–183, 2022. [DOI: 10.1007/s12532-021-00208-8](https://doi.org/10.1007/s12532-021-00208-8)

---

## Contributing

Politecnico di Milano — Aerial Robotics 2025-26.

For questions or issues, please open a GitHub issue.
