# Quadrotor NMPC — Obstacle-Aware Local Planner

An **acados SQP-RTI** based Nonlinear Model Predictive Control (NMPC) framework for quadrotor autonomous flight.

Includes quaternion-based attitude tracking, obstacle avoidance via Artificial Potential Fields (APF) or in-MPC keep-out constraints, 5th-order polynomial trajectory generation, and Gazebo simulation integration.

**Politecnico di Milano — Aerial Robotics 2025-26**

---

## Table of Contents

* [Overview](#overview)
* [System Requirements](#system-requirements)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Architecture](#architecture)
* [Configuration](#configuration)
* [Testing and Usage](#testing-and-usage)
* [Obstacle Avoidance](#obstacle-avoidance)
* [Log Analysis](#log-analysis)
* [References](#references)

---

## Overview

```
State  x ∈ R^13  :  [px, py, pz,  vx, vy, vz,  qw, qx, qy, qz,  p, q, r]
Input  u ∈ R^4   :  [f_total, τx, τy, τz]
```

| Component | Description |
| --- | --- |
| `quadrotor_mpc_client_v3.py` | Main control loop and public API |
| `local_planner_mpc.py` | Obstacle-aware NMPC — primary solver (acados SQP-RTI, N=20, Ts=50ms) |
| `mpc_solver.py` | Core QuadrotorMPC — landing cone constraint only, no obstacle avoidance |
| `quadrotor_model.py` | CasADi/acados dynamic model (quaternion-based) |
| `global_planner.py` | `WaypointTrajectory`, `BackflipTrajectory`, `APFTrajectory` |
| `perception.py` | `PerceptionManager` — three-level perception pipeline |
| `plot_mpc_log.py` | Log visualization and auto-save to `plots/` |
| `plot_backflip_paper.py` | Paper-ready backflip analysis plots (4-panel) |
| `plot_backflip.py` | Detailed backflip analysis plots |
| `plot_apf_field.py` | APF force field + potential surface visualization |
| `plot_slalom_paper.py` | Slalom analysis / controls paper plot (per world) |
| `plot_compare_paper.py` | In-MPC vs APF comparison figures |

> **Note:** `mpc_solver.py` (`QuadrotorMPC`) is not used directly by the main client. `LocalPlannerMPC` in `local_planner_mpc.py` is the active solver; it supersedes `QuadrotorMPC` and adds online obstacle-avoidance constraints.

**Features:**

* Quaternion-based NMPC (no Euler-angle singularities)
* 1.0s prediction horizon (N=20, Ts=50ms), SQP-RTI single iteration per step
* APF-based reactive obstacle avoidance: online horizon building at each MPC step
* Offline APF path planning: pre-plans full path, fits quintic polynomials, MPC tracks
* In-MPC soft keep-out constraints: obstacle avoidance embedded directly in the OCP
* Backflip: Lupashin 6-phase bang-coast-bang with NMPC post-flip recovery
* Landing cone soft constraint: `vz + α·z ≥ 0` (prevents hard landings)
* `+`-configuration motor mixer with negative-omega² detection and clamping
* Quintic polynomial multi-waypoint trajectory generation (C2-continuous)
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
├── quadrotor_mpc_client_v3.py      # Main API (hover, slalom_reactive, slalom_mpc_avoid,
│                                   #   slalom_mpc_homotopy, backflip, backflip_mpc_recovery)
├── local_planner_mpc.py            # LocalPlannerMPC — active NMPC solver with obstacle constraints
├── mpc_solver.py                   # QuadrotorMPC — landing-cone-only solver (reference)
├── quadrotor_model.py              # acados dynamic model (CasADi / quaternion)
├── global_planner.py               # WaypointTrajectory, BackflipTrajectory, APFTrajectory
├── perception.py                   # PerceptionManager (3 levels)
├── simulation.sh                   # Basic simulation stack (no obstacles)
├── simulation_obstacles.sh         # Obstacle avoidance simulation stack
├── setup_deps.sh                   # Dependency installer (first time only)
├── env_setup.sh                    # Environment variables (source each session)
├── plot_mpc_log.py                 # Log visualizer (auto-saves to plots/)
├── plot_apf_field.py               # APF force field + potential visualization
├── plot_backflip.py                # Backflip analysis plots (detailed)
├── plot_backflip_paper.py          # Backflip paper plot (4-panel, single column)
├── plot_slalom_paper.py            # Slalom analysis/controls paper plot (per world)
├── plot_compare_paper.py           # In-MPC vs APF comparison figures
├── worlds/
│   ├── quad.world                  # Empty world
│   ├── quad_obstacles.world        # 3 cylindrical obstacles
│   ├── quad_obstacles_dense.world  # 5-obstacle alternating slalom
│   └── quad_obstacles_line.world   # 5 obstacles ON the start→goal line
├── model/
│   ├── mrsim-quadrotor-lidar/
│   │   ├── model.sdf               # Quadrotor with 2D Lidar
│   │   └── model.config
│   └── mrsim-rotor/
│       ├── model.sdf
│       └── model.config
├── plots/                          # Auto-saved plot images
├── acados_generated/               # Auto-generated solver C code
└── logs/mpc/                       # Flight logs (.npz)
```

---

## Architecture

### System Data Flow

Every 50 ms the following pipeline executes:

```
Gazebo Simulation
      │
      ▼
  pom.frame('robot')
      │  position, velocity, quaternion, angular rates
      ▼
  pom_to_state()
      │  normalize quaternion, antipodal fix (qw < 0 → flip sign)
      │  → x ∈ R^13
      ▼
  PerceptionManager.get_obstacles()          ← Level 1 / 2 / 3
      │  [(p_obs, R_obs), ...]
      │
      ├── [Reactive APF] _apf_horizon()      ← APF force at current pos,
      │       integrate N+1 steps → x_ref    ← integrated forward Ts each step
      │
      ├── [Goal-only]    _goal_horizon()     ← straight-line to goal
      │
      └── [Homotopy]     _homotopy_horizon() ← goal + pass-side tangent hint
      │
      ▼
  LocalPlannerMPC.solve(x0, x_ref_horizon, obstacles)
      │  acados SQP-RTI, HPIPM QP
      │  cost:       ||[x;u] − [x_ref; u_hover]||²_W
      │  soft h_obs: ||p − p_obs_i||² − (R_obs_i + R_drone)² ≥ 0
      │  soft h_land: vz + α·z ≥ 0
      │  input bounds: f ∈ [0.4mg, 2.5mg], |τxy| ≤ 0.25, |τz| ≤ 0.06 Nm
      │  → u_opt = [f, τx, τy, τz]
      ▼
  wrench_to_rotorcraft(f, τx, τy, τz)
      │  allocation matrix B⁻¹ (+ configuration)
      │  → Ω² = B⁻¹ · [f, τx, τy, τz]
      │  clip Ω² ≥ 0,  Ω ≤ 1200 rad/s
      ▼
  rotorcraft.set_velocity([Ω0, Ω1, Ω2, Ω3, ...])
      │
      ▼
  Session Logger (_session dict)
      → logs/mpc/<tag>/mpc_log.npz
```

### Trajectory / Horizon Builders

| Builder | Used by | Description |
| --- | --- | --- |
| `_apf_horizon()` | `slalom_reactive` | APF force integrated N+1 steps; reactive to current obstacles |
| `_goal_horizon()` | `slalom_mpc_avoid` | Straight-line to goal; MPC keep-out does all avoidance |
| `_homotopy_horizon()` | `slalom_mpc_homotopy` | Goal + signed tangent hint; MPC keep-out enforces clearance |
| `WaypointTrajectory` | `hover`, `set_position`, `landing` | Quintic polynomial, offline |
| `APFTrajectory` | `slalom` | APF path computed offline → quintic fit → MPC tracks |
| `BackflipTrajectory` | `backflip` (reference only) | Ballistic arc + 360° quintic angle profile |

### Perception Levels

| Level | Class | Source | Latency |
| --- | --- | --- | --- |
| 1 | `GazeboGroundTruth` | gz transport / gz CLI | < 10 ms |
| 2 | `Lidar2DPerception` | `/lidar/scan` → DBSCAN | < 1 ms |
| 3 | `StaticObstacles` | Manually registered positions | 0 ms |

### Backflip — 6-Phase Sequence

```
Phase 1  Climb      NMPC active; ascend to target altitude (default 10 m)
Phase 2  Pop-up     Open-loop thrust impulse: f = 2·mg for 0.40 s
Phase 3  Flip       Open-loop feedforward:
                      Accel  →  Coast (near-zero thrust)  →  Decel
                    Quaternion pitch integrated for phase transitions.
                    Full 360° completes in ≈0.73 s, peak pitch rate ≈9.5 rad/s.
Phase 4a Rate-kill  P-controller on body rates (open-loop, 0.8 s max)
                    Damps residual angular momentum to bring rates into
                    NMPC's convergence basin.
Phase 4b Recovery   NMPC reactivated with boosted attitude/rate damping weights.
                    f_min = 0 N (allows near-zero thrust for inversion recovery).
                    Stabilises from ≈1.7 m peak lateral drift back toward hover.
Phase 5  Return     NMPC tracks trajectory back to pre-flip hover point.
Phase 6  Landing    Vertical descent; auto-detect obstacle height below drone.
```

---

## Configuration

### Physical Constants

```python
MASS    = 1.280        # kg
I_DIAG  = (22.916e-3, 22.916e-3, 22.132e-3)  # kg·m² (Ixx, Iyy, Izz)
ARM_LEN = 0.23         # m
KF      = 6.5e-4       # N/(rad/s)²  thrust coefficient
KM      = 1e-5         # Nm/(rad/s)² moment coefficient
```

### MPC Parameters (as set in `quadrotor_mpc_client_v3.py`)

```python
MPC_N  = 20            # Prediction horizon steps
MPC_TS = 0.05          # Sampling time [s] → 1.0 s total horizon

_MPC_KWARGS = dict(
    Q_pos=5.0,   Q_vel=3.0,  Q_att=1.5,
    Q_omega=25.0, Q_omega_r=6.0,        # Q_omega_r: reduced yaw-rate weight
    P_scale=5.0,
    R_f=0.01,    R_tau=0.10, R_tau_z=0.20,
    tau_max=0.25, tau_z_max=0.06,       # τz tight: physical limit ≈ 0.19 Nm at hover
    f_min=0.40*MASS*G,
    f_max_scale=2.5,
    alpha_land=2.0,  W_land=500.0,      # landing cone: |vz| ≤ 2·z near ground
)
```

> **Note:** The default constructors in `mpc_solver.py` and `local_planner_mpc.py` use different values (e.g. `Q_vel=1.0`). The values above — from `_MPC_KWARGS` in the client — are the ones actually used at runtime.

### LocalPlannerMPC Obstacle Parameters

```python
_LOCAL_MPC_KWARGS = dict(
    n_obs_max = 5,       # Max simultaneous obstacles in OCP
    R_drone   = 0.35,    # Drone collision radius [m]  (keep-out Rd)
    W_obs     = 10000.0, # Soft constraint penalty weight
)
```

### Perception Levels

| Level | Source | Description |
| --- | --- | --- |
| 1 | Gazebo ground truth | Reads obstacle poses via gz transport / CLI |
| 2 | 2D Lidar | Clusters `/lidar/scan` into obstacles (DBSCAN, ε=0.25 m, n_min=3) |
| 3 | Static | Manually registered obstacle positions (default) |

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

# In-line slalom (obstacles ON the start→goal line; forces real avoidance):
bash simulation_obstacles.sh worlds/quad_obstacles_line.world
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

With `use_perception=True`, obstacles are detected via 2D lidar (LaserScan → DBSCAN → nearest-neighbor association) instead of hardcoded positions. Requires `perception_level=2` in `setup()`.

### Obstacle Avoidance — In-MPC Constraint vs APF (comparison)

Three avoidance architectures differing only in **where** avoidance happens. Run on the in-line slalom world (obstacles on the start→goal line, so the drone must actively deviate):

```bash
# Terminal 1
bash simulation_obstacles.sh worlds/quad_obstacles_line.world
```

```python
# Terminal 2
>>> setup(perception_level=2)       # lidar + DBSCAN

# (1) pure-MPC — goal-only reference; MPC keep-out constraint does ALL avoidance.
#     Reliable up to ~1.5 m/s.
>>> slalom_mpc_avoid(max_vel=1.5)

# (2) homotopy — reference gives a pass-side hint; MPC keep-out enforces clearance.
>>> slalom_mpc_homotopy(max_vel=2.0, tangent_gain=0.4)

# (3) APF + MPC tracker — APF reference avoids, MPC just tracks.
#     Tighten APF margin to match in-MPC clearance for a fair comparison:
>>> slalom_reactive(use_perception=True, max_vel=2.2, apf_d0=0.3, apf_R_drone=0.20)
```

| Function | Reference generator | Avoidance done by |
|---|---|---|
| `slalom_mpc_avoid` | `_goal_horizon` (goal only) | MPC keep-out constraint |
| `slalom_mpc_homotopy` | `_homotopy_horizon` (goal + side hint) | MPC keep-out constraint |
| `slalom_reactive` | `_apf_horizon` (full APF) | APF planner (MPC tracks) |

Key parameters:
- `max_vel` — commanded cruise speed (sweep to find reliable ceiling)
- `safety_margin` (avoid / homotopy) — extra keep-out radius added per obstacle
- `tangent_gain` (homotopy) — strength of the pass-side hint
- `apf_d0`, `apf_R_drone` (reactive) — APF influence distance and drone radius; lower values → tighter clearance (used to match in-MPC margin)

### Obstacle Avoidance — Offline APF

```python
# Use simulation_obstacles.sh world
>>> setup()
>>> slalom()
```

Pre-plans full path with APF, fits quintic polynomials, MPC tracks. Faster peak speed but less reliable on sharp turns.

### Backflip

```python
# Use simulation.sh (no obstacles; needs altitude clearance)
>>> setup()
>>> backflip()                # SO(3) PD recovery
>>> backflip_ilc()            # tuned params + PD position-feedback recovery
>>> backflip_mpc_recovery()   # feedforward flip + genuine NMPC recovery (paper version)
```

Lupashin 6-phase bang-coast-bang backflip (see [Architecture — Backflip](#backflip----6-phase-sequence) for full description).

**Recovery variants:**
- `backflip()` — SO(3) quaternion-based PD recovery with velocity damping.
- `backflip_ilc()` — tuned flip params (`f_accel=8.77 N`, `f_decel=8.81 N`) + position feedback (`K_pos=0.05`) on the SO(3) PD recovery.
- `backflip_mpc_recovery()` — open-loop feedforward flip; recovery done by NMPC with boosted attitude/rate weights (`Q_omega` 80, `f_min=0 N`). This is the variant used for the paper figures.

**Typical results — `backflip_mpc_recovery()` (good flip exit, qw ≈ −1):**

| Metric | Value |
| --- | --- |
| Flip duration | ~0.73 s |
| Peak pitch rate | ~9.5 rad/s |
| Peak lateral drift | ~1.7 m (at flip exit) |
| Altitude loss | none (fast flip; min z ≥ hover) |
| Landing error from hover | ~0.1 m |
| Recovery via | NMPC (Phase 4b) |

> **Note:** `backflip_mpc_recovery()` with `tau_flip=1.1 Nm` is the most consistent low-drift variant. Lowering `f_bang` below `0.70·mg` reduces drift but makes the flip unreliable (insufficient vertical support → altitude loss / tumble).

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

### APF Force Model

Attractive force pulls toward goal; repulsive force pushes away from obstacles with a rotational tangent component:

```
F_att = k_att × (goal − pos) / ‖goal − pos‖     (capped at k_att)
F_rep = k_rep × (1/ρ − 1/d0) × (1/ρ²) × (r̂ + 0.5 × τ̂)
```

where `ρ = dist − R_obs − R_drone` is the effective clearance margin. Tangent direction per obstacle: `sign(cross(line_dir, obs_offset))` — obstacle left of start→goal line → pass right, and vice versa. Creates a natural alternating slalom pattern.

### World Layouts

**Original (`quad_obstacles.world`) — 3 obstacles:**

```
                       obs2(6, 1.5)
Start(0,0)  →  obs1(3,0)  ──────────────────  obs3(9,−1)  →  Goal(12,0)
                 🔴 r=0.4      🟠 r=0.4          🔵 r=0.4
```

**Dense (`quad_obstacles_dense.world`) — 5 obstacles, alternating slalom:**

```
     y
 1.5 ┊       ○2(6)          ○4(12)
     ┊
   0 ─S┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄G(18,0)→ x
     ┊
-1.5 ┊    ○1(3)       ○3(9)        ○5(15)
     ┊
     0  3  6  9  12  15  18
```

All obstacles: radius = 0.4 m, height = 3.0 m. Obstacles alternate y = ±1.5 m to force zigzag weaving. Per-side margin = 1.5 − 0.4 − 0.5 = 0.60 m.

**In-line (`quad_obstacles_line.world`) — 5 obstacles ON the start→goal line:**

Used for controlled avoidance architecture comparison (paper). Obstacles at x = 3.5, 6.5, 9.5, 12.5, 15.5 m with y = ±0.3 m alternating offset. Any avoidance must be produced by the active mechanism.

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

### Avoidance Comparison Results (in-line slalom, matched ≈0.9 m clearance)

| Mode | Avoidance by | Min. clearance | Reliable speed |
| --- | --- | --- | --- |
| `slalom_mpc_avoid` (pure-MPC) | MPC keep-out constraint | ≈0.9 m | 1.5 m/s |
| `slalom_reactive` (APF tight) | APF planner | ≈0.9 m | 2.2 m/s |
| `slalom_reactive` (APF wide) | APF planner | ≈2.5 m | 2.81 m/s |

The APF speed advantage at matched clearance comes from wider look-ahead (earlier, smoother turns), not from the architecture itself. Both modes share the same failure mode above their speed ceiling: attitude runaway (qw drops below 0.7, body thrust axis tips sideways, speed spikes 2→9 m/s). The average MPC solve time is 8.6 ms, well within the 50 ms budget.

### Reactive Slalom Results — Original World

| Metric | Value |
| --- | --- |
| Duration | 21.0 s |
| MPC solve time | 8.7 ms avg |
| Obstacles avoided | 3 |
| Min distance to obs1 | 1.28 m |
| Min distance to obs2 | 2.47 m |
| Min distance to obs3 | 1.42 m |
| Max speed | 2.83 m/s |
| Torque saturation | 49% |
| qw min | 0.987 (stable) |

### Reactive Slalom Results — Dense World (Lidar Perception)

| Metric | Value |
| --- | --- |
| Duration | ~25 s |
| Obstacles | 5 (detected via 2D lidar) |
| Avg speed | 1.28 m/s |
| Obstacle margins | 0.9–1.9 m |
| Perception | DBSCAN clustering + nearest-neighbor association |

### Perception Pipeline (Lidar Mode)

```
LaserScan (360° 2D, 10 Hz) → Polar-to-Cartesian → DBSCAN Clustering (ε=0.25m, n_min=3)
                                                          ↓
                                         Nearest-Neighbor Data Association (1.5 m threshold)
                                                          ↓
                                              (p_obs, R_obs) list  →  LocalPlannerMPC
```

Filtering applied: clusters with radius > 1.0 m rejected; obstacles behind drone (x < drone_x − 1.5 m) ignored; |y| > 5.0 m rejected. Drone altitude < 1.5 m suppresses lidar processing entirely.

---

## Log Analysis

Logs saved to `logs/mpc/<log_tag>/mpc_log.npz`.

```python
import numpy as np

data = np.load('logs/mpc/slalom/mpc_log.npz')
t         = data['t']          # (N,)     Time [s]
x         = data['x']          # (N, 13)  State vector
u         = data['u']          # (M, 4)   Control input [f, τx, τy, τz]
xref      = data['xref']       # (N, 13)  Reference trajectory
mpc_times = data['mpc_times']  # (M,)     MPC solver times [ms]
n_obs     = data['n_obs']      # (M,)     Number of active obstacles
slack     = data['slack']      # (M,)     Max obstacle slack per step
```

### Plot Generation

```bash
# Inside Docker:
python3 plot_mpc_log.py logs/mpc/slalom/mpc_log.npz

# Save to plots/:
python3 plot_mpc_log.py --save logs/mpc/slalom/mpc_log.npz
```

### Avoidance-Comparison Tools

```bash
# Paper-ready comparison figures (trajectory / speed ceiling / runaway):
python3 plot_compare_paper.py

# Pure-MPC slalom analysis + controls (5-panel), styled like the paper:
python3 plot_slalom_paper.py --world line

# APF force field and potential surface:
python3 plot_apf_field.py                                          # dense world
python3 plot_apf_field.py --world original                         # 3-obstacle world
python3 plot_apf_field.py logs/mpc/slalom/mpc_log.npz --world original
```

---

## References

1. O. Khatib, "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots," *The International Journal of Robotics Research*, vol. 5, no. 1, pp. 90–98, 1986. [DOI: 10.1177/027836498600500106](https://doi.org/10.1177/027836498600500106)

2. S. S. Ge and Y. J. Cui, "New Potential Functions for Mobile Robot Path Planning," *IEEE Transactions on Robotics and Automation*, vol. 16, no. 5, pp. 615–620, 2000. [DOI: 10.1109/70.880813](https://doi.org/10.1109/70.880813)

3. S. Lupashin, A. Schöllig, M. Sherback, and R. D'Andrea, "A Simple Learning Strategy for High-Speed Quadrocopter Multi-Flips," in *Proc. IEEE International Conference on Robotics and Automation (ICRA)*, pp. 1642–1648, 2010. [DOI: 10.1109/ROBOT.2010.5509452](https://doi.org/10.1109/ROBOT.2010.5509452)

4. R. Verschueren, G. Frison, D. Kouzoupis, et al., "acados — A Modular Open-Source Framework for Fast Embedded Optimal Control," *Mathematical Programming Computation*, vol. 14, no. 1, pp. 147–183, 2022. [DOI: 10.1007/s12532-021-00208-8](https://doi.org/10.1007/s12532-021-00208-8)

5. G. Frison and M. Diehl, "HPIPM: A High-Performance Interior-Point Method for Quadratic Programming and Model Predictive Control," *IFAC-PapersOnLine*, vol. 53, no. 2, pp. 6563–6569, 2020.

---

## Contributing

Politecnico di Milano — Aerial Robotics 2025-26.

For questions or issues, please open a GitHub issue.
