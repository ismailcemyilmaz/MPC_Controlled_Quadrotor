# Quadrotor NMPC — Obstacle-Aware Local Planner

An **acados SQP-RTI** based Nonlinear MPC controller for a quadrotor.

Includes a 5th-order polynomial global trajectory generator and Gazebo simulation integration.

**Politecnico di Milano — Aerial Robotics**

---

## Table of Contents

* [Overview](#overview)
* [System Requirements](#system-requirements)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Configuration](#configuration)
* [Testing and Usage](#testing-and-usage)
* [Log Analysis](#log-analysis)
* [Known Issues](#known-issues)

---

## Overview

```
State  x ∈ R^13  :  [px, py, pz,  vx, vy, vz,  qw, qx, qy, qz,  p, q, r]
Input  u ∈ R^4   :  [f_total, τx, τy, τz]

```

| Component | Description |
| --- | --- |
| `local_planner_mpc.py` | Obstacle-aware NMPC (acados SQP-RTI, N=20, Ts=50ms) |
| `mpc_solver.py` | Core QuadrotorMPC (includes landing cone constraints) |
| `quadrotor_model.py` | CasADi/acados dynamic model (quaternion-based) |
| `global_planner.py` | Quintic polynomial WaypointTrajectory |
| `perception.py` | PerceptionManager (Gazebo GT / 2D Lidar / Static) |
| `quadrotor_mpc_client_v3.py` | Main control loop and public API |

**Features:**

* Quaternion-based NMPC (no Euler angle singularities)
* Online obstacle parameters (N_obs_max=5 slots, padded with dummy obstacles)
* Soft obstacle constraints: $W_{obs} \cdot \text{slack}^2$ penalty
* Physically consistent quaternion reference via differential flatness
* Shared log session: takeoff + hover + landing stored within a single `.npz` file
* Standardized test loop via `hover()`

---

## System Requirements

| Requirement | Version |
| --- | --- |
| Ubuntu | 22.04 / 24.04 |
| Docker Engine | ≥ 24 |
| Python | ≥ 3.10 |
| acados | ≥ 0.3 |
| CasADi | ≥ 3.6 |
| NumPy | ≥ 1.24 |
| Gazebo | Ionic (tk3lab Docker image) |
| GenoM3 | pocolibs middleware |

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url> MPC_Lidar
cd MPC_Lidar

```

### 2. Start the Docker environment (tk3lab)

```bash
# Inside the tk3lab workspace
./simulation.sh

```

This command initializes: `h2 init`, `genomixd`, `rotorcraft`, `pom`, `optitrack`, and `gz sim`.

### 3. Python Dependencies

Inside the Docker container:

```bash
pip install casadi numpy

```

### 4. acados Installation

```bash
# Recommended: /opt/acados or ~/acados
git clone https://github.com/acados/acados.git ~/acados
cd ~/acados && git submodule update --recursive --init
mkdir build && cd build
cmake .. -DACADOS_WITH_QPOASES=ON
make -j4 && sudo make install

# Python interface
pip install -e ~/acados/interfaces/acados_template

```

Set the `ACADOS_SOURCE_DIR` environment variable or adjust the paths at the top of `local_planner_mpc.py`:

```python
_ACADOS_CANDIDATES = ['/opt/acados', os.path.expanduser('~/acados')]

```

### 5. Deploy the Drone Model

```bash
# Copy the model.sdf and model.config files to the Gazebo models directory
cp model/mrsim-quadrotor-lidar/  ~/.gazebo/models/  -r

```

---

## Project Structure

```
MPC_Lidar/
├── quadrotor_mpc_client_v3.py   # Main API (hover, landing, set_position)
├── local_planner_mpc.py         # Obstacle-aware LocalPlannerMPC
├── mpc_solver.py                # Core QuadrotorMPC
├── quadrotor_model.py           # acados dynamic model
├── global_planner.py            # WaypointTrajectory (quintic)
├── perception.py                # PerceptionManager
├── simulation.sh                # Launches the simulation stack
├── model/
│   ├── mrsim-quadrotor-lidar/
│   │   ├── model.sdf            # Quadrotor with attached Lidar
│   │   └── model.config
│   └── mrsim-rotor/
│       ├── model.sdf
│       └── model.config
├── acados_generated/            # Auto-generated (on first run)
└── logs/mpc/                    # Test logs (.npz)

```

---

## Configuration

Parameters located within `quadrotor_mpc_client_v3.py`:

```python
# Physical constants
MASS    = 1.280        # kg
I_DIAG  = (22.916e-3, 22.916e-3, 22.132e-3)  # kg·m²
ARM_LEN = 0.23         # m
KF      = 6.5e-4       # N/(rad/s)²
KM      = 1e-5         # Nm/(rad/s)²

# MPC
MPC_N  = 20
MPC_TS = 0.05          # s — at 0.025s the horizon is too short (oscillation period > horizon)

_MPC_KWARGS = dict(
    Q_pos=5.0,   Q_vel=2.0,
    Q_att=6.0,                   # Attitude stability is critical — do not decrease
    Q_omega=1.5, Q_omega_r=3.0,
    P_scale=5.0,
    tau_max=0.80,                # Physical max ~1.44 Nm — safe margin
    tau_z_max=0.12,
    f_min=0.05*MASS*G,           # 0.63 N — Best result achieved in Run 4
    f_max_scale=2.5,
)

_LOCAL_MPC_KWARGS = dict(
    n_obs_max=5,
    R_drone=0.30,                # m
    W_obs=10000.0,
)

```

---

## Testing and Usage

### Terminal 1 — Start the Simulation Stack

```bash
cd /shared-workspace/src/MPC_Lidar
bash simulation.sh

```

### Terminal 2 — Control Python REPL

```bash
cd /shared-workspace/src/MPC_Lidar
python3 -i quadrotor_mpc_client_v3.py

```

### Standard Hover Test

```python
>>> setup()
>>> hover(0, 0, 4, T_hover=10, log_tag='run_v1')

```

Pipeline flow:

1. `start()` — Arms the motors
2. 3-second spin-up delay
3. Initial state report (z, roll, p)
4. `set_position(0,0,4, T_hold=10)` — Climb, hold, and start logging
5. `landing()` — Vertical descent, disarm motors, and save log

Expected terminal output:

```
[start] Motors armed — call set_position(x, y, z) to fly
[hover] Waiting for spin-up (3s)...
[hover] Initial: z=0.030m  roll=0.0°  p=0.001 rad/s
[set_position] target=(0.00,0.00,4.00)  obstacles=0
[log] Session opened — tag='run_v1'
[set_position] T_hold completed — drone at (0.00,0.00,4.00).
[landing] No obstacles detected — landing on ground (z = 0)
[landing] Motors stopped — final z = 0.042 m
[log] Saved → .../logs/mpc/run_v1/mpc_log.npz
      duration=19.6s  MPC=11.2ms  avg_obs=0.0  	τ_sat=92.1%

```

### Landing on top of an Obstacle Test

```python
# Add an obstacle in Gazebo, then run:
>>> setup(perception_level=1)
>>> add_obstacle(0.0, 0.0, 0.8, 0.3)   # (x, y, z_center, radius)
>>> hover(0, 0, 4, T_hover=10, log_tag='obstacle_test')
# landing() will automatically target the top of the obstacle: z_target = 0.8 + 0.3 + 0.05 = 1.15m

```

### Manual Control

```python
>>> setup()
>>> start()
>>> set_position(0, 0, 2, T_hold=5)    # Waypoint
>>> set_position(0, 0, 4, T_hold=10)   # Final target (logging stays active)
>>> landing()                           # Saves the log session

```

### Emergency Stop

```python
>>> stop()   # Shuts down immediately without saving logs

```

---

## Log Analysis

Log files are stored in `logs/mpc/<log_tag>/mpc_log.npz`.

```python
import numpy as np

data = np.load('logs/mpc/run_v1/mpc_log.npz')
t        = data['t']          # (N,)     Time [s]
x        = data['x']          # (N, 13)  State vector
u        = data['u']          # (M, 4)   Control input [f, τx, τy, τz]
xref     = data['xref']       # (N, 13)  Reference trajectory
mpc_times= data['mpc_times']  # (M,)     MPC solver step times [ms]
n_obs    = data['n_obs']       # (M,)     Number of active obstacles

# Euler angles
qw, qx, qy, qz = x[:,6], x[:,7], x[:,8], x[:,9]
roll  = np.degrees(np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2)))
pitch = np.degrees(np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1)))

# Basic metrics
print(f"z_max:    {x[:,2].max():.3f} m")
print(f"MPC mean: {mpc_times.mean():.1f} ms")
tau_max = 0.80
sat = np.mean(np.any(np.abs(u[:,1:3]) >= tau_max*0.99, axis=1))
print(f"τ_sat:    {sat*100:.1f}%")

```

---

## Known Issues

### Model-Plant Mismatch (Gazebo)

**Symptom:** Only 1 out of 7 test runs (Run 4) successfully reached the z=4m target.

**Cause:** The ground-contact dynamics present in the Gazebo simulation are completely missing from the MPC model. While the drone is still on the ground during motor spin-up, angular momentum builds up. This induces a structural oscillation (≈2.3 Hz) observed across all runs.

**Status:** Under active investigation — awaiting verification of the Gazebo SDF model's `<inertial><pose>` values.

### Systematic tau_y Bias

**Symptom:** The mean value of `tau_y` is persistently negative across all runs (-0.10 to -0.43 Nm).

**Likely Cause:** Center of Mass (CoM) displacement or rotor asymmetry within the Gazebo drone model.

**Workaround:** Added a feedforward term inside `wrench_to_rotorcraft`: `tau_y += TAU_Y_FF`.

### MPC_TS = 0.025s Fails to Work

**Cause:** Python is single-threaded; the MPC solve time (~11ms) combined with loop overhead yields an effective period of 36.9ms. Because the solver integrates assuming a 25ms step but updates at 37ms, a model drift of 12ms occurs at every single step. Additionally, when N=20 and Ts=0.025, the prediction horizon is 0.5s—which is shorter than the oscillation period (~0.44s).

**Solution:** Stick to `MPC_TS=0.05`. To extend the horizon safely, try `N=30, Ts=0.05` (horizon=1.5s).

---

## Contributing

This project is developed as part of the Aerial Robotics course at Politecnico di Milano.

For questions or issues, please open a GitHub issue.
