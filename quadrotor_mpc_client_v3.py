"""
quadrotor_mpc_client.py  (perception-integrated version)
=========================================================
Changes from base:
  - QuadrotorMPC -> LocalPlannerMPC
  - perception.PerceptionManager integrated
  - _run_loop: obstacles parameter
  - setup(): perception initialization
  - New public API: set_obstacle_level(), add_obstacle()
  - Shared log session: set_position() opens, landing() closes
  - landing(): vertical descent, auto-position above detected obstacles
"""

import time
import os
import numpy as np

from local_planner_mpc import LocalPlannerMPC
from global_planner import WaypointTrajectory, BackflipTrajectory, APFTrajectory
from perception import PerceptionManager

try:
    import genomix
    GENOMIX_AVAILABLE = True
except ImportError:
    GENOMIX_AVAILABLE = False
    print("[client] genomix not found — simulation only")

# ── Physical Constants ────────────────────────────────────────────────────────
G       = 9.81
MASS    = 1.280
I_DIAG  = (22.916e-3, 22.916e-3, 22.132e-3)
ARM_LEN = 0.23
KF      = 6.5e-4
KM      = 1e-5

# ── MPC Configuration ───────────────────────────────────────────────────────
MPC_N  = 20
MPC_TS = 0.05

TAU_Y_FF = 0.0 #0.20

_MPC_KWARGS = dict(
    Q_pos=5.0,     Q_vel=3.0,
    Q_att=1.5,
    Q_omega=25.0,  Q_omega_r=6.0,
    P_scale=5.0,
    R_f=0.01,      R_tau=0.10,       R_tau_z=0.20,
    tau_max=0.25,  tau_z_max=0.06,
    f_min=0.40*MASS*G,
    f_max_scale=2.5,
    alpha_land=2.0, W_land=500.0,
)

_LOCAL_MPC_KWARGS = dict(
    n_obs_max  = 5,
    R_drone    = 0.35,
    W_obs      = 10000.0,
)

# ── Perception configuration ────────────────────────────────────────────────
OBSTACLE_MODELS = [
    ('obstacle_1', 0.4),
    ('obstacle_2', 0.4),
    ('obstacle_3', 0.4),
]

# ── Flight parameters ───────────────────────────────────────────────────────
GROUND_Z        = 0.10
MAX_VEL         = 1.5
MIN_SAFE_Z      = 1.5
DESCENT_VEL     = 0.30   # landing descent speed [m/s]
LAND_XY_RADIUS  = 0.50   # obstacle search radius [m]
LAND_MARGIN     = 0.05   # safety clearance above obstacle [m]

_HERE   = os.path.dirname(os.path.abspath(__file__))
_WS     = os.path.normpath(os.path.join(_HERE, '..', '..'))
LOG_DIR = os.path.join(_WS, 'logs', 'mpc')

# ── Module state ────────────────────────────────────────────────────────────
_g          = None
_pom        = None
_rotorcraft = None
_optitrack  = None
_mpc        = None
_armed      = False
_perception = None
_flip_mode  = False

# ── Shared log session ──────────────────────────────────────────────────────
_session: dict = {}
_session_t0: float = 0.0


def _session_open(tag: str) -> None:
    """Open a new log session. Overwrites previous if still active."""
    global _session, _session_t0
    _session = dict(tag=tag, t=[], x=[], u=[], xref=[], mpc_ms=[], n_obs=[],
                    slack=[])
    _session_t0 = time.time()
    print(f"[log] Session opened — tag='{tag}'")


def _session_close() -> str | None:
    """Write current session to disk, return path. None if no session."""
    global _session
    if not _session or not _session['t']:
        print("[log] No session to save.")
        return None

    s    = _session
    tag  = s['tag']
    dest = os.path.join(LOG_DIR, tag)
    os.makedirs(dest, exist_ok=True)
    fname = os.path.join(dest, 'mpc_log.npz')

    np.savez(fname,
             t        = np.array(s['t']),
             x        = np.array(s['x']),
             u        = np.array(s['u'])        if s['u']      else np.empty((0, 4)),
             xref     = np.array(s['xref']),
             mpc_times= np.array(s['mpc_ms'])   if s['mpc_ms'] else np.empty(0),
             n_obs    = np.array(s['n_obs'])     if s['n_obs']  else np.empty(0),
             slack    = np.array(s['slack'])     if s.get('slack') else np.empty(0))

    total_s  = s['t'][-1] - s['t'][0] if len(s['t']) > 1 else 0
    avg_ms   = float(np.mean(s['mpc_ms']))  if s['mpc_ms']  else 0.0
    avg_obs  = float(np.mean(s['n_obs']))   if s['n_obs']   else 0.0

    tau_max  = _MPC_KWARGS['tau_max']
    sat      = 0.0
    if s['u']:
        u_arr = np.array(s['u'])
        sat   = float(np.mean(np.any(np.abs(u_arr[:, 1:3]) >= tau_max * 0.99, axis=1))) * 100

    print(f"[log] Saved -> {fname}")
    print(f"      duration={total_s:.1f}s  MPC={avg_ms:.1f}ms  "
          f"avg_obs={avg_obs:.1f}  tau_sat={sat:.1f}%")

    _session = {}
    return fname


# ── State conversion ────────────────────────────────────────────────────────
def pom_to_state(frame) -> np.ndarray:
    px = frame['pos']['x'];  py = frame['pos']['y'];  pz = frame['pos']['z']
    vx = frame['vel']['vx']; vy = frame['vel']['vy']; vz = frame['vel']['vz']
    qw = frame['att']['qw']; qx = frame['att']['qx']
    qy = frame['att']['qy']; qz = frame['att']['qz']
    norm = (qw**2 + qx**2 + qy**2 + qz**2)**0.5 + 1e-12
    qw /= norm; qx /= norm; qy /= norm; qz /= norm
    if not _flip_mode and qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    p = frame['avel']['wx']; q = frame['avel']['wy']; r = frame['avel']['wz']
    return np.array([px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r])


def _current_state() -> np.ndarray:
    return pom_to_state(_pom.frame('robot')['frame'])


# ── Motor mixer ──────────────────────────────────────────────────────────────
_ALLOC_INV = np.linalg.pinv(np.array([
    [ KF,            KF,           KF,           KF          ],
    [ 0,             ARM_LEN*KF,   0,           -ARM_LEN*KF  ],
    [-ARM_LEN*KF,    0,            ARM_LEN*KF,   0            ],
    [ KM,           -KM,           KM,          -KM           ],
]))
_mixer_warn_t = 0.0

def wrench_to_rotorcraft(f, tau_x, tau_y, tau_z):
    global _mixer_warn_t
    tau_y_comp = tau_y + TAU_Y_FF
    omega_sq = _ALLOC_INV @ np.array([f, tau_x, tau_y_comp, tau_z])
    if np.any(omega_sq < 0):
        now = time.time()
        if now - _mixer_warn_t > 2.0:
            print(f"[mixer] WARNING: negative omega^2 clipped (f={f:.1f} tau=[{tau_x:.3f},{tau_y_comp:.3f},{tau_z:.3f}])")
            _mixer_warn_t = now
    omega_sq = np.clip(omega_sq, 0.0, None)
    omega    = np.sqrt(omega_sq)
    omega    = np.clip(omega, 0.0, 1200.0)
    _rotorcraft.set_velocity({'desired': [
        float(omega[0]), float(omega[1]),
        float(omega[2]), float(omega[3]),
        0.0, 0.0, 0.0, 0.0,
    ]})


# ── Trajectory builder ──────────────────────────────────────────────────────
def _build_goto_traj(curr_pos, target_pos, max_vel=MAX_VEL):
    """Build goto trajectory for set_position(). Not used by landing()."""
    cx, cy, cz = curr_pos
    tx, ty, tz = target_pos

    dist     = np.linalg.norm(target_pos - curr_pos)
    T_travel = max(2.5 * dist / max_vel, 4.0)
    waypoints = [
        {'pos': curr_pos,    'vel': [0, 0, 0]},
        {'pos': target_pos,  'vel': [0, 0, 0]},
    ]
    traj = WaypointTrajectory(waypoints, seg_times=[T_travel])
    return traj, traj.total_duration()


def _build_land_traj(curr_pos, land_z, descent_vel=DESCENT_VEL):
    """Build vertical descent trajectory for landing(). x,y fixed, z -> land_z."""
    cx, cy, cz = curr_pos
    descent    = max(cz - land_z, 0.0)
    T_descend  = max(descent / descent_vel, 2.0)   # min 2s

    waypoints = [
        {'pos': np.array([cx, cy, cz]),    'vel': [0, 0, 0]},
        {'pos': np.array([cx, cy, land_z]),'vel': [0, 0, 0]},
    ]
    traj = WaypointTrajectory(waypoints, seg_times=[T_descend])
    return traj, traj.total_duration()


# ── Control loop ─────────────────────────────────────────────────────────────
def _run_loop(traj: WaypointTrajectory,
              T_total: float,
              ground_cutoff: bool = False,
              f_min_clamp: float = None) -> None:
    """
    MPC control loop.

    When trajectory ends (t > traj.total_duration()), reference freezes at
    the final point — drone hovers at target. Returns after T_total elapsed.

    If ground_cutoff=True, exits early when z < GROUND_Z and descending.

    f_min_clamp: lower bound applied to the MPC thrust output each step.
    Defaults to 0.60*mg (safety floor for normal flight). Pass 0.0 for
    backflip MPC recovery, which deliberately allows near-zero thrust while
    righting from an inverted attitude.

    Log data written to module-level _session; does not save to disk.
    """
    t_start    = time.time()
    t_next_mpc = t_start
    f_min_safe = 0.60 * MASS * G if f_min_clamp is None else f_min_clamp

    T_traj = traj.total_duration()

    while True:
        t_now = time.time() - t_start
        if t_now > T_total:
            break

        t_ref = min(t_now, T_traj)

        x_now     = _current_state()

        if ground_cutoff and x_now[2] < GROUND_Z and x_now[5] <= 0:
            print(f"[landing] ground contact — z={x_now[2]:.3f}m")
            break

        x_ref_now = traj.state_at(t_ref)

        t_global = time.time() - _session_t0

        if _session:
            _session['t'].append(t_global)
            _session['x'].append(x_now.copy())
            _session['xref'].append(x_ref_now.copy())

        if time.time() >= t_next_mpc:
            if _perception is not None:
                _perception.update_drone_pos(x_now[:3])
                obstacles = _perception.get_obstacles()
            else:
                obstacles = []

            xref_h      = traj.get_horizon(t_ref, MPC_N, MPC_TS)
            u_opt, info = _mpc.solve(x_now, xref_h, obstacles=obstacles,
                                     sign_correct=not _flip_mode)

            if u_opt[0] < f_min_safe:
                u_opt[0] = f_min_safe

            if _session:
                _session['u'].append(u_opt.copy())
                _session['mpc_ms'].append(info['solve_time_ms'])
                _session['n_obs'].append(len(obstacles))

            if info.get('slack_max', 0) > 1e-2:
                print(f"[run] t={t_now:.1f}s  "
                      f"obs_slack={info['slack_max']:.3f}  "
                      f"n_obs={len(obstacles)}")

            wrench_to_rotorcraft(*u_opt)
            now = time.time()
            while t_next_mpc <= now:
                t_next_mpc += MPC_TS

        time.sleep(0.002)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def setup(perception_level: int = 3):
    global _g, _pom, _rotorcraft, _optitrack, _mpc, _perception

    assert GENOMIX_AVAILABLE, "genomix not available"

    _g = genomix.connect()
    _g.rpath(os.environ['HOME'] + '/openrobots/lib/genom/pocolibs/plugins')

    _pom        = _g.load('pom')
    _rotorcraft = _g.load('rotorcraft')
    _optitrack  = _g.load('optitrack')

    _optitrack.connect({
        'host': 'localhost', 'host_port': '1509',
        'mcast': '', 'mcast_port': '0',
    })
    _rotorcraft.connect({'serial': '/tmp/pty-qr4', 'baud': 0})
    _rotorcraft.set_sensor_rate(
        {'rate': {'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1}})
    _rotorcraft.set_imu_filter(
        {'gfc': [20,20,20], 'afc': [5,5,5], 'mfc': [20,20,20]})
    _pom.set_prediction_model('::pom::constant_acceleration')
    _pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})
    _pom.set_history_length({'history_length': 0.25})
    _pom.set_mag_field(
        {'magdir': {'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06}})
    _pom.connect_port({'local': 'measure/imu',   'remote': 'rotorcraft/imu'})
    _pom.add_measurement('imu')
    _pom.connect_port({'local': 'measure/mag',   'remote': 'rotorcraft/mag'})
    _pom.add_measurement('mag')
    _pom.connect_port(
        {'local': 'measure/mocap', 'remote': 'optitrack/bodies/QR_4'})
    _pom.add_measurement('mocap')

    os.makedirs(LOG_DIR, exist_ok=True)

    _mpc = LocalPlannerMPC(
        N=MPC_N, Ts=MPC_TS, mass=MASS, I_diag=I_DIAG,
        rk4_steps=1,
        **_LOCAL_MPC_KWARGS,
        **_MPC_KWARGS,
    )

    if perception_level == 1 and not OBSTACLE_MODELS:
        print("[WARN] perception_level=1 but OBSTACLE_MODELS empty — falling back to level 3")
        perception_level = 3

    if perception_level == 1:
        _perception = PerceptionManager(
            level=1, obstacle_models=OBSTACLE_MODELS, update_rate_hz=10.0)
    elif perception_level == 2:
        _perception = PerceptionManager(level=2)
    else:
        _perception = PerceptionManager(level=3)

    _perception.start()
    print("[setup] Ready — call start() to arm motors")


def start():
    global _armed
    assert _pom is not None and _mpc is not None, "Call setup() first"
    _rotorcraft.start()
    _rotorcraft.servo(ack=True)
    time.sleep(1.0)
    _mpc.reset(_current_state())
    _armed = True
    print("[start] Motors armed — call set_position(x, y, z) to fly")


def set_position(x: float, y: float, z: float,
                 T_hold: float = 10.0,
                 max_vel: float = MAX_VEL,
                 log_tag: str = 'goto') -> None:
    """
    Fly drone to (x,y,z), hold for T_hold seconds, then return.

    Log session opens with this call. Next landing() call closes it.
    Typical tuning flow:

        set_position(0, 0, 4, T_hold=60)   # blocks ~66s
        landing()                           # saves log
    """
    assert _armed, "Call start() first"

    target   = np.array([float(x), float(y), float(z)])
    curr_pos = _current_state()[:3]

    obs = _perception.get_obstacles() if _perception else []
    print(f"\n[set_position] target=({x:.2f},{y:.2f},{z:.2f})  "
          f"obstacles={len(obs)}")

    traj, T_travel = _build_goto_traj(curr_pos, target, max_vel)
    T_total        = T_travel + T_hold

    _session_open(log_tag)
    _mpc.reset(_current_state())

    _run_loop(traj, T_total)

    print(f"[set_position] Hold finished — drone at ({x:.2f},{y:.2f},{z:.2f}).")
    print("  Call landing() to descend or set_position() for a new target.")


def landing(descent_vel: float = DESCENT_VEL,
            xy_radius:   float = LAND_XY_RADIUS,
            margin:      float = LAND_MARGIN) -> None:
    """
    Land the drone while maintaining current x,y position.

    Checks perceived obstacles directly below (horizontal distance < xy_radius).
    Landing z is set to the highest obstacle top surface + margin.
    If no obstacles below, lands at ground level (z=0).

    Parameters
    ----------
    descent_vel : descent speed [m/s]
    xy_radius   : obstacle search radius below drone [m]
    margin      : safety clearance above obstacle [m]

    Motors stop and log is saved when this function returns.
    """
    assert _armed, "Call start() first"

    x_now    = _current_state()
    curr_pos = x_now[:3]
    curr_xy  = curr_pos[:2]

    # ── Find obstacles below ─────────────────────────────────────────────────
    obstacles  = _perception.get_obstacles() if _perception else []
    land_z     = 0.0
    n_below    = 0

    for p_obs, R_obs in obstacles:
        horiz = np.linalg.norm(np.asarray(p_obs[:2], float) - curr_xy)
        if horiz < xy_radius:
            z_top  = float(p_obs[2]) + float(R_obs) + margin
            land_z = max(land_z, z_top)
            n_below += 1

    if n_below:
        print(f"\n[landing] {n_below} obstacle(s) below — "
              f"landing z = {land_z:.3f} m  (above obstacle)")
    else:
        print(f"\n[landing] No obstacles below — landing at ground (z=0)")

    # ── Descent trajectory ───────────────────────────────────────────────────
    traj, T_descend = _build_land_traj(curr_pos, land_z, descent_vel)
    T_total         = T_descend + 2.0

    print(f"[landing] current z={curr_pos[2]:.2f}m  "
          f"target z={land_z:.2f}m  duration~{T_descend:.1f}s")

    if not _session:
        _session_open('landing_only')

    _run_loop(traj, T_total, ground_cutoff=True)

    # ── Stop motors and save log ─────────────────────────────────────────────
    _rotorcraft.stop()
    print(f"[landing] Motors stopped — "
          f"final z = {_current_state()[2]:.3f} m")

    _session_close()


def stop():
    """Emergency stop — does not save log."""
    if _rotorcraft is not None:
        _rotorcraft.stop()
        print("[stop] Motors stopped (log not saved)")

def hover(x: float, y: float, z: float,
          T_hover: float = 10.0,
          log_tag: str = 'hover',
          spinup_wait: float = 3.0) -> None:
    """
    Standard test function — full cycle from motor start to log save.

    1. Arm motors and wait for spin-up  -> start()
    2. Fly to (x,y,z)                   -> set_position()
    3. Hold at altitude for T_hover seconds
    4. Vertical descent, stop motors    -> landing()

        >>> setup()
        >>> hover(0, 0, 4, T_hover=15)
    """
    start()

    print(f"[hover] waiting for spin-up ({spinup_wait:.0f}s)...")
    for i in range(int(spinup_wait), 0, -1):
        print(f"[hover]   {i}s", end='\r')
        time.sleep(1.0)
    print()

    x0 = _current_state()
    print(f"[hover] Start: z={x0[2]:.3f}m  "
          f"roll={np.degrees(np.arctan2(2*(x0[6]*x0[7]+x0[8]*x0[9]), 1-2*(x0[7]**2+x0[8]**2))):.1f}deg  "
          f"p={x0[10]:.3f} rad/s")

    set_position(x, y, z, T_hold=T_hover, log_tag=log_tag)
    landing()


def follow_waypoints(waypoints: list,
                     T_hold: float = 3.0,
                     max_vel: float = MAX_VEL,
                     log_tag: str = 'waypoints',
                     spinup_wait: float = 3.0) -> None:
    """
    Fly through a list of waypoints, hold at each, then land.

    Parameters
    ----------
    waypoints : list of [x, y, z] or (x, y, z)
    T_hold    : seconds to hover at final waypoint before landing
    max_vel   : max velocity along trajectory [m/s]
    log_tag   : log session name

    Example:
        >>> setup()
        >>> follow_waypoints([[0,0,3], [4,1,3], [8,-1,3], [12,0,3]])
    """
    assert len(waypoints) >= 2, "need >= 2 waypoints"

    start()
    print(f"[waypoints] waiting for spin-up ({spinup_wait:.0f}s)...")
    for i in range(int(spinup_wait), 0, -1):
        print(f"[waypoints]   {i}s", end='\r')
        time.sleep(1.0)
    print()

    curr_pos = _current_state()[:3]
    wp_list = [{'pos': curr_pos, 'vel': [0, 0, 0]}]
    for wp in waypoints:
        wp_list.append({'pos': np.array(wp, dtype=float), 'vel': [0, 0, 0]})

    traj = WaypointTrajectory(wp_list, max_vel=max_vel)
    T_total = traj.total_duration() + T_hold

    _session_open(log_tag)
    _mpc.reset(_current_state())
    _mpc.warm_start_trajectory(traj)

    print(f"[waypoints] {len(waypoints)} targets, T_traj={traj.total_duration():.1f}s + hold={T_hold:.0f}s")
    _run_loop(traj, T_total)

    print(f"[waypoints] Trajectory complete.")
    if _perception is not None:
        _perception.clear_obstacles()
    landing()


# ── Perception management API ───────────────────────────────────────────────

def add_obstacle(x: float, y: float, z: float, radius: float):
    """Add a static obstacle at runtime (level=3 perception)."""
    assert _perception is not None, "Call setup() first"
    _perception.add_static(x, y, z, radius)


def set_obstacle_level(level: int, **kwargs):
    global _perception
    if _perception:
        _perception.stop()
    _perception = PerceptionManager(level=level, **kwargs)
    _perception.start()
    print(f"[perception] Switched to level {level}")


# ── Obstacle avoidance demos ───────────────────────────────────────────────

_SLALOM_OBSTACLES = [
    (3.0,  0.0, 1.5, 0.4),
    (6.0,  1.5, 1.5, 0.4),
    (9.0, -1.0, 1.5, 0.4),
]

_SLALOM_WAYPOINTS = [
    [0.0,  0.0, 2.0],
    [12.0, 0.0, 2.0],
]


def slalom(alt: float = 2.0,
           max_vel: float = 1.0,
           T_hold: float = 3.0,
           spinup_wait: float = 3.0) -> None:
    """
    Obstacle avoidance demo using APF (Artificial Potential Field) path planning.

    APF generates an obstacle-free reference path from start to goal.
    MPC tracks this path with obstacle soft constraints as safety backup.

    Use with simulation_obstacles.sh world.

        >>> setup()
        >>> slalom()
    """
    assert _perception is not None, "Call setup() first"

    obstacles = []
    for ox, oy, oz, r in _SLALOM_OBSTACLES:
        obstacles.append((np.array([ox, oy, oz]), r))
    print(f"[slalom] {len(obstacles)} obstacles for APF planning (MPC tracks path only)")

    _session_open('slalom')

    start()
    print(f"[slalom] Takeoff to {alt}m...")
    curr_pos = _current_state()[:3]
    hover_target = np.array([curr_pos[0], curr_pos[1], alt])
    traj_up = WaypointTrajectory(
        [{'pos': curr_pos, 'vel': [0,0,0]},
         {'pos': hover_target, 'vel': [0,0,0]}],
        seg_times=[3.0])
    _mpc.reset(_current_state())
    _run_loop(traj_up, 5.0)

    print(f"[slalom] Hover stable, starting APF trajectory...")
    curr_pos = _current_state()[:3]
    goal_pos = np.array([12.0, 0.0, alt])

    apf_margin = _LOCAL_MPC_KWARGS['R_drone'] + 0.3
    traj = APFTrajectory(
        start=curr_pos, goal=goal_pos, obstacles=obstacles,
        max_vel=max_vel, R_drone=apf_margin)

    _mpc.reset(_current_state())
    _mpc.warm_start_trajectory(traj)

    print(f"[slalom] APF trajectory: T={traj.total_duration():.1f}s + hold={T_hold:.0f}s")
    _run_loop(traj, traj.total_duration() + T_hold)

    print(f"[slalom] Trajectory complete.")
    landing()


def _apf_force_2d(pos_2d, goal_2d, obstacles, k_att=1.0, k_rep=0.8, d0=2.5,
                  R_drone=0.65, obs_signs=None):
    """Compute 2D APF force at a given position."""
    diff_goal = goal_2d - pos_2d
    f_att = k_att * diff_goal
    f_att_norm = np.linalg.norm(f_att)
    if f_att_norm > k_att:
        f_att = f_att / f_att_norm * k_att

    f_rep = np.zeros(2)
    for idx, (obs_pos, obs_r) in enumerate(obstacles):
        diff = pos_2d - obs_pos[:2]
        dist = np.linalg.norm(diff)
        margin = dist - obs_r - R_drone
        if margin < d0 and margin > 0.01:
            strength = k_rep * (1.0/margin - 1.0/d0) * (1.0/margin**2)
            radial = diff / dist
            sign = obs_signs[idx] if obs_signs is not None else 1
            tangent = sign * np.array([-diff[1], diff[0]]) / dist
            f_rep += strength * (radial + 0.5 * tangent)
        elif margin <= 0.01:
            f_rep += k_rep * 100.0 * (diff / max(dist, 0.01))

    f_total = f_att + f_rep
    if np.linalg.norm(f_total) < 1e-6:
        f_total = f_att + np.array([0.0, 0.1])
    return f_total


def _apf_horizon(pos_3d, goal_3d, obstacles, obs_signs, N, Ts, max_vel,
                 apf_d0=2.5, apf_R_drone=0.50):
    """Build N+1 reference states by simulating APF forward from current pos.

    apf_d0 / apf_R_drone expose the APF influence distance and drone radius so
    the margin can be tightened for a fair comparison vs the MPC keep-out.
    Defaults reproduce the original behaviour.
    """
    xref = np.zeros((N + 1, 13))
    pos = pos_3d.copy()
    goal_2d = goal_3d[:2]

    for k in range(N + 1):
        dist_to_goal = np.linalg.norm(pos[:2] - goal_2d)

        if dist_to_goal < 0.3:
            vel = np.zeros(3)
            pos[:2] = goal_2d
        else:
            f = _apf_force_2d(pos[:2], goal_2d, obstacles,
                              d0=apf_d0, R_drone=apf_R_drone, obs_signs=obs_signs)
            f_norm = np.linalg.norm(f)
            direction = f / f_norm if f_norm > 1e-6 else np.array([1.0, 0.0])
            if direction[0] < 0.1:
                direction[0] = 0.1
                direction = direction / np.linalg.norm(direction)
            # Gentle braking near the goal (ramp speed with remaining distance)
            # so high-speed runs do not overshoot / fail to stop at the goal.
            speed = min(max_vel, dist_to_goal / 1.5)
            vel = np.array([direction[0] * speed, direction[1] * speed, 0.0])

        xref[k, 0:3] = pos
        xref[k, 3:6] = vel
        xref[k, 6] = 1.0  # qw

        if k < N:
            pos = pos.copy()
            pos[:2] += vel[:2] * Ts
            pos[2] = goal_3d[2]

    return xref


def _goal_horizon(pos_3d, goal_3d, N, Ts, max_vel):
    """Build N+1 goal-directed reference states — NO obstacle term.

    Pure attractive (go-to-goal) reference: a straight pull toward the goal
    at capped speed, with no repulsion. Obstacle avoidance is delegated
    entirely to the NMPC keep-out constraint. Used by slalom_mpc_avoid().
    Output layout matches _apf_horizon: [p(3), v(3), q(4), w(3)], qw=1.
    """
    xref = np.zeros((N + 1, 13))
    pos = pos_3d.copy()
    goal_2d = goal_3d[:2]

    for k in range(N + 1):
        to_goal = goal_2d - pos[:2]
        dist_to_goal = np.linalg.norm(to_goal)

        if dist_to_goal < 0.3:
            vel = np.zeros(3)
            pos[:2] = goal_2d
        else:
            direction = to_goal / dist_to_goal
            # Gentle braking near the goal: ramp speed with remaining distance
            # so a high-speed run does not have to decelerate abruptly at the
            # goal (abrupt stop -> hard pitch-back -> tumble).
            speed = min(max_vel, dist_to_goal / 1.5)
            vel = np.array([direction[0] * speed, direction[1] * speed, 0.0])

        xref[k, 0:3] = pos
        xref[k, 3:6] = vel
        xref[k, 6] = 1.0  # qw

        if k < N:
            pos = pos.copy()
            pos[:2] += vel[:2] * Ts
            pos[2] = goal_3d[2]

    return xref


def _homotopy_horizon(pos_3d, goal_3d, obstacles, obs_signs, N, Ts, max_vel,
                      tangent_gain=0.8, d_influence=2.5):
    """Goal-directed reference with ONLY a tangential side-bias per obstacle.

    Unlike _apf_horizon (full radial + tangential repulsion, which routes a
    collision-free path), this adds NO radial push — only a small tangential
    ("which side to pass") nudge. The reference therefore still approaches
    close to obstacles, leaving the actual clearance to the MPC keep-out
    constraint. The reference picks the homotopy class; the MPC enforces
    safety. Used by slalom_mpc_homotopy().
    """
    xref = np.zeros((N + 1, 13))
    pos = pos_3d.copy()
    goal_2d = goal_3d[:2]

    for k in range(N + 1):
        to_goal = goal_2d - pos[:2]
        dist_to_goal = np.linalg.norm(to_goal)

        if dist_to_goal < 0.3:
            vel = np.zeros(3)
            pos[:2] = goal_2d
        else:
            direction = to_goal / dist_to_goal          # attractive, no radial
            tang = np.zeros(2)
            for idx, (obs_pos, obs_r) in enumerate(obstacles):
                diff = pos[:2] - obs_pos[:2]
                od = np.linalg.norm(diff)
                margin = od - obs_r
                if 0.01 < margin < d_influence and od > 1e-6:
                    sign = obs_signs[idx] if obs_signs is not None else 1
                    t = sign * np.array([-diff[1], diff[0]]) / od
                    w = tangent_gain * (1.0 - margin / d_influence)
                    tang += w * t
            d2 = direction + tang
            n2 = np.linalg.norm(d2)
            d2 = d2 / n2 if n2 > 1e-6 else direction
            # Force forward (+x) progress ONLY while still en route to the goal,
            # so a tangent term can't stall the drone. Past the goal x, let the
            # reference point freely (back toward goal) so the drone can stop.
            if d2[0] < 0.1 and pos[0] < goal_2d[0] - 0.5:
                d2[0] = 0.1
                d2 = d2 / np.linalg.norm(d2)
            # Gentle braking near the goal (see _goal_horizon).
            speed = min(max_vel, dist_to_goal / 1.5)
            vel = np.array([d2[0] * speed, d2[1] * speed, 0.0])

        xref[k, 0:3] = pos
        xref[k, 3:6] = vel
        xref[k, 6] = 1.0  # qw

        if k < N:
            pos = pos.copy()
            pos[:2] += vel[:2] * Ts
            pos[2] = goal_3d[2]

    return xref


def slalom_reactive(alt: float = 2.0,
                    max_vel: float = 2.5,
                    T_timeout: float = 40.0,
                    goal_tol: float = 0.5,
                    use_perception: bool = False,
                    apf_d0: float = 2.5,
                    apf_R_drone: float = 0.50) -> None:
    """
    Real-time APF obstacle avoidance — no pre-planned waypoints.

    At each MPC step, APF computes velocity reference from current position.
    MPC tracks this reactive reference.

    use_perception=False: hardcoded obstacles (default, tested)
    use_perception=True:  read obstacles from PerceptionManager (lidar/GT)

        >>> setup()
        >>> slalom_reactive()                        # hardcoded
        >>> slalom_reactive(use_perception=True)     # perception
    """
    assert _perception is not None, "Call setup() first"

    goal_pos = np.array([18.0, 0.0, alt])
    start_pos = np.array([0.0, 0.0, alt])
    line_dir = goal_pos[:2] - start_pos[:2]

    if use_perception:
        print(f"[slalom_reactive] PERCEPTION mode — obstacles from PerceptionManager")
        obstacles = []
        obs_signs = []
        _known_obs = []
    else:
        obstacles = []
        for ox, oy, oz, r in _SLALOM_OBSTACLES:
            obstacles.append((np.array([ox, oy, oz]), r))
        obs_signs = []
        for obs_pos, obs_r in obstacles:
            obs_off = obs_pos[:2] - start_pos[:2]
            cross = line_dir[0] * obs_off[1] - line_dir[1] * obs_off[0]
            obs_signs.append(+1 if cross >= 0 else -1)
            side = "left" if cross > 0 else ("right" if cross < 0 else "center")
            print(f"  obs({obs_pos[0]:.0f},{obs_pos[1]:.0f}): {side} -> pass {'right' if cross >= 0 else 'left'}")

    print(f"[slalom_reactive] goal=({goal_pos[0]},{goal_pos[1]},{goal_pos[2]}), max_vel={max_vel}")

    tag = 'slalom_reactive_lidar' if use_perception else 'slalom_reactive'
    _session_open(tag)
    start()

    # Takeoff
    print(f"[slalom_reactive] Takeoff to {alt}m...")
    curr_pos = _current_state()[:3]
    hover_target = np.array([curr_pos[0], curr_pos[1], alt])
    traj_up = WaypointTrajectory(
        [{'pos': curr_pos, 'vel': [0, 0, 0]},
         {'pos': hover_target, 'vel': [0, 0, 0]}],
        seg_times=[3.0])
    _mpc.reset(_current_state())
    _run_loop(traj_up, 5.0)

    # Reactive APF loop
    print(f"[slalom_reactive] Hover stable, starting reactive APF...")
    _mpc.reset(_current_state())

    t_start = time.time()
    t_next_mpc = t_start
    reached_goal = False

    while True:
        t_now = time.time() - t_start
        if t_now > T_timeout:
            print(f"[slalom_reactive] Timeout ({T_timeout:.0f}s)")
            break

        x_now = _current_state()
        dist_to_goal = np.linalg.norm(x_now[:2] - goal_pos[:2])

        if dist_to_goal < goal_tol and not reached_goal:
            reached_goal = True
            t_goal = t_now
            print(f"[slalom_reactive] Reached goal at t={t_now:.1f}s, holding 1.5s...")

        if reached_goal and t_now - t_goal > 1.5:
            print(f"[slalom_reactive] Hold complete.")
            break

        # Update obstacles from perception
        if use_perception:
            _perception.update_drone_pos(x_now[:3])
            perceived = _perception.get_obstacles()
            obstacles = []
            obs_signs = []
            for obs_pos, obs_r in perceived:
                if obs_r > 1.0:
                    continue
                if obs_pos[0] < x_now[0] - 1.5:
                    continue
                if abs(obs_pos[1]) > 5.0:
                    continue

                obstacles.append((obs_pos, obs_r))

                matched_sign = None
                for kpos, ksign in _known_obs:
                    if np.linalg.norm(obs_pos[:2] - kpos) < 1.5:
                        matched_sign = ksign
                        break

                if matched_sign is not None:
                    obs_signs.append(matched_sign)
                else:
                    obs_off = obs_pos[:2] - start_pos[:2]
                    cross = line_dir[0] * obs_off[1] - line_dir[1] * obs_off[0]
                    sign = +1 if cross >= 0 else -1
                    _known_obs.append((obs_pos[:2].copy(), sign))
                    obs_signs.append(sign)
                    side = "left" if cross > 0 else "right"
                    print(f"  [perception] new obs({obs_pos[0]:.1f},{obs_pos[1]:.1f}) r={obs_r:.2f}: {side} -> pass {'right' if cross >= 0 else 'left'}")

        t_global = time.time() - _session_t0

        xref_h = _apf_horizon(x_now[:3], goal_pos, obstacles, obs_signs,
                              MPC_N, MPC_TS, max_vel,
                              apf_d0=apf_d0, apf_R_drone=apf_R_drone)

        if _session:
            _session['t'].append(t_global)
            _session['x'].append(x_now.copy())
            _session['xref'].append(xref_h[0].copy())

        if time.time() >= t_next_mpc:
            u_opt, info = _mpc.solve(x_now, xref_h, obstacles=[],
                                     sign_correct=True)

            if _session:
                _session['u'].append(u_opt.copy())
                _session['mpc_ms'].append(info['solve_time_ms'])
                _session['n_obs'].append(len(obstacles))

            wrench_to_rotorcraft(*u_opt)
            now = time.time()
            while t_next_mpc <= now:
                t_next_mpc += MPC_TS

        time.sleep(0.002)

    # Obstacle distances
    if _session and obstacles:
        xs = np.array(_session['x'])
        for i, (obs_pos, obs_r) in enumerate(obstacles):
            dists = np.sqrt((xs[:, 0] - obs_pos[0])**2 + (xs[:, 1] - obs_pos[1])**2)
            print(f"  obs{i+1}: min_dist={np.min(dists):.2f}m (safe={obs_r + 0.35:.2f}m)")

    # Stop lidar before landing to avoid phantom obstacles during descent
    if use_perception and hasattr(_perception, '_backend'):
        _perception.stop()

    landing()


def slalom_mpc_avoid(alt: float = 2.0,
                     max_vel: float = 1.5,
                     safety_margin: float = 0.30,
                     T_timeout: float = 40.0,
                     goal_tol: float = 0.5) -> None:
    """
    Pure-MPC obstacle avoidance: the reference only says "go to goal", the
    NMPC keep-out constraint does ALL the avoiding.

    Real-time lidar perception only — NO hardcoded obstacle positions.
    At each control step:
      - lidar + DBSCAN detect obstacles online (PerceptionManager level 2),
      - _goal_horizon() builds a straight goal-directed reference with NO
        repulsion (APF avoidance is OFF here),
      - the detected obstacles are passed to the NMPC, whose keep-out
        constraint  ||p_drone - p_obs||^2 >= (R_obs + R_drone)^2  actively
        enforces clearance inside the optimal control problem.

    Tuning (added after the 2.5 m/s run collided):
      - max_vel       (A): slower approach => more reaction distance for the
                           keep-out to bend the drone around in time.
      - safety_margin (B): each detected obstacle radius is inflated by this
                           amount before being handed to the MPC, so the
                           keep-out activates EARLIER. This is local to this
                           function — it does NOT change R_drone in the shared
                           MPC build, so slalom_reactive() is unaffected.

    Difference vs slalom_reactive(): there the MPC is a pure tracker
    (solve(..., obstacles=[])) and avoidance is done by APF alone. Here the
    reference carries no obstacle information and the MPC enforces all safety.

        >>> setup(perception_level=2)   # lidar + DBSCAN
        >>> slalom_mpc_avoid()                       # 1.5 m/s, margin 0.30
        >>> slalom_mpc_avoid(max_vel=2.0, safety_margin=0.4)  # push faster
    """
    assert _perception is not None, "Call setup(perception_level=2) first"

    goal_pos   = np.array([18.0, 0.0, alt])
    obstacles  = []
    _known_obs = []   # detected obstacle XY positions (for logging only)

    print(f"[slalom_mpc_avoid] REALTIME lidar perception — MPC keep-out active")
    print(f"[slalom_mpc_avoid] goal=({goal_pos[0]},{goal_pos[1]},{goal_pos[2]}), "
          f"max_vel={max_vel}, safety_margin={safety_margin}")

    _session_open('slalom_mpc_avoid')
    start()

    # Takeoff
    print(f"[slalom_mpc_avoid] Takeoff to {alt}m...")
    curr_pos = _current_state()[:3]
    hover_target = np.array([curr_pos[0], curr_pos[1], alt])
    traj_up = WaypointTrajectory(
        [{'pos': curr_pos, 'vel': [0, 0, 0]},
         {'pos': hover_target, 'vel': [0, 0, 0]}],
        seg_times=[3.0])
    _mpc.reset(_current_state())
    _run_loop(traj_up, 5.0)

    print(f"[slalom_mpc_avoid] Hover stable, starting goal-only ref + MPC avoidance...")
    _mpc.reset(_current_state())

    t_start = time.time()
    t_next_mpc = t_start
    reached_goal = False

    while True:
        t_now = time.time() - t_start
        if t_now > T_timeout:
            print(f"[slalom_mpc_avoid] Timeout ({T_timeout:.0f}s)")
            break

        x_now = _current_state()
        dist_to_goal = np.linalg.norm(x_now[:2] - goal_pos[:2])

        if dist_to_goal < goal_tol and not reached_goal:
            reached_goal = True
            t_goal = t_now
            print(f"[slalom_mpc_avoid] Reached goal at t={t_now:.1f}s, holding 1.5s...")

        if reached_goal and t_now - t_goal > 1.5:
            print(f"[slalom_mpc_avoid] Hold complete.")
            break

        # ── Real-time obstacle update from lidar (no hardcoded positions) ────
        # Detected obstacles are handed to the MPC ONLY. The reference
        # generator never sees them, so every avoidance decision is made by
        # the NMPC keep-out constraint — APF does no avoidance here.
        _perception.update_drone_pos(x_now[:3])
        perceived = _perception.get_obstacles()
        obstacles = []
        for obs_pos, obs_r in perceived:
            if obs_r > 1.0:
                continue
            if obs_pos[0] < x_now[0] - 1.5:
                continue
            if abs(obs_pos[1]) > 5.0:
                continue

            # B: inflate radius by safety_margin so the MPC keep-out activates
            # earlier (more reaction distance). Local to this function — the
            # shared MPC build / R_drone is untouched.
            obstacles.append((obs_pos, obs_r + safety_margin))

            is_new = all(np.linalg.norm(obs_pos[:2] - kpos) >= 1.5
                         for kpos in _known_obs)
            if is_new:
                _known_obs.append(obs_pos[:2].copy())
                print(f"  [perception] new obs({obs_pos[0]:.1f},{obs_pos[1]:.1f}) "
                      f"r={obs_r:.2f} (+{safety_margin:.2f} margin) -> MPC keep-out")

        t_global = time.time() - _session_t0

        # Goal-ONLY reference (no repulsion). Avoidance is the MPC's job.
        xref_h = _goal_horizon(x_now[:3], goal_pos, MPC_N, MPC_TS, max_vel)

        if _session:
            _session['t'].append(t_global)
            _session['x'].append(x_now.copy())
            _session['xref'].append(xref_h[0].copy())

        if time.time() >= t_next_mpc:
            # KEY: detected obstacles are passed to the NMPC, so the keep-out
            # constraint enforces clearance (vs slalom_reactive's obstacles=[]).
            u_opt, info = _mpc.solve(x_now, xref_h, obstacles=obstacles,
                                     sign_correct=True)

            if _session:
                _session['u'].append(u_opt.copy())
                _session['mpc_ms'].append(info['solve_time_ms'])
                _session['n_obs'].append(info.get('n_obs_active', len(obstacles)))

            wrench_to_rotorcraft(*u_opt)
            now = time.time()
            while t_next_mpc <= now:
                t_next_mpc += MPC_TS

        time.sleep(0.002)

    # Min clearance per discovered obstacle (over the whole flight)
    if _session and _known_obs:
        xs = np.array(_session['x'])
        for i, kpos in enumerate(_known_obs):
            dists = np.sqrt((xs[:, 0] - kpos[0])**2 + (xs[:, 1] - kpos[1])**2)
            print(f"  obs{i+1}@({kpos[0]:.1f},{kpos[1]:.1f}): "
                  f"min_dist={np.min(dists):.2f}m")

    # Stop lidar before landing to avoid phantom obstacles during descent
    if hasattr(_perception, '_backend'):
        _perception.stop()

    landing()


def slalom_mpc_homotopy(alt: float = 2.0,
                        max_vel: float = 2.5,
                        safety_margin: float = 0.20,
                        tangent_gain: float = 0.8,
                        T_timeout: float = 40.0,
                        goal_tol: float = 1.0) -> None:
    """
    Architecture C: the reference gives only a HOMOTOPY HINT (which side to
    pass); the NMPC keep-out constraint produces the actual clearance.

    Real-time lidar perception only — NO hardcoded obstacle positions.
    Per control step:
      - lidar + DBSCAN detect obstacles online,
      - a pass-side is chosen per obstacle from sign(cross(start->goal, offset)),
      - _homotopy_horizon() builds a goal-directed reference with ONLY a small
        tangential side-bias (NO radial repulsion) — the drone still comes
        close to obstacles,
      - the detected obstacles are passed to the NMPC keep-out constraint,
        which enforces clearance. The per-step constraint slack is logged;
        slack>0 proves the MPC (not the reference) is doing the avoidance.

    Sits between the two baselines kept for comparison:
      - slalom_reactive   : APF routes a collision-free path, MPC just tracks
                            (MPC does no avoidance).
      - slalom_mpc_avoid  : goal-only reference, MPC does 100% reactively
                            (fragile/unreliable above ~1.4 m/s).
      - slalom_mpc_homotopy: reference picks the side, MPC enforces safety
                            (standard planner-homotopy + MPC-clearance design).

        >>> setup(perception_level=2)
        >>> slalom_mpc_homotopy()
    """
    assert _perception is not None, "Call setup(perception_level=2) first"

    goal_pos   = np.array([18.0, 0.0, alt])
    start_pos  = np.array([0.0, 0.0, alt])
    line_dir   = goal_pos[:2] - start_pos[:2]

    obstacles  = []
    obs_signs  = []
    _known_obs = []   # (xy, sign) per discovered obstacle

    print(f"[slalom_mpc_homotopy] REALTIME lidar — homotopy hint + MPC keep-out")
    print(f"[slalom_mpc_homotopy] goal=({goal_pos[0]},{goal_pos[1]},{goal_pos[2]}), "
          f"max_vel={max_vel}, margin={safety_margin}, tangent_gain={tangent_gain}")

    _session_open('slalom_mpc_homotopy')
    start()

    # Takeoff
    print(f"[slalom_mpc_homotopy] Takeoff to {alt}m...")
    curr_pos = _current_state()[:3]
    hover_target = np.array([curr_pos[0], curr_pos[1], alt])
    traj_up = WaypointTrajectory(
        [{'pos': curr_pos, 'vel': [0, 0, 0]},
         {'pos': hover_target, 'vel': [0, 0, 0]}],
        seg_times=[3.0])
    _mpc.reset(_current_state())
    _run_loop(traj_up, 5.0)

    print(f"[slalom_mpc_homotopy] Hover stable, starting homotopy ref + MPC avoidance...")
    _mpc.reset(_current_state())

    t_start = time.time()
    t_next_mpc = t_start
    reached_goal = False

    while True:
        t_now = time.time() - t_start
        if t_now > T_timeout:
            print(f"[slalom_mpc_homotopy] Timeout ({T_timeout:.0f}s)")
            break

        x_now = _current_state()
        dist_to_goal = np.linalg.norm(x_now[:2] - goal_pos[:2])

        if dist_to_goal < goal_tol and not reached_goal:
            reached_goal = True
            t_goal = t_now
            print(f"[slalom_mpc_homotopy] Reached goal at t={t_now:.1f}s, holding 1.5s...")

        if reached_goal and t_now - t_goal > 1.5:
            print(f"[slalom_mpc_homotopy] Hold complete.")
            break

        # ── Real-time detection + per-obstacle pass-side decision ────────────
        _perception.update_drone_pos(x_now[:3])
        perceived = _perception.get_obstacles()
        obstacles = []
        obs_signs = []
        for obs_pos, obs_r in perceived:
            if obs_r > 1.0:
                continue
            if obs_pos[0] < x_now[0] - 1.5:
                continue
            if abs(obs_pos[1]) > 5.0:
                continue

            # inflate radius slightly so the keep-out has reaction room
            obstacles.append((obs_pos, obs_r + safety_margin))

            matched_sign = None
            for kpos, ksign in _known_obs:
                if np.linalg.norm(obs_pos[:2] - kpos) < 1.5:
                    matched_sign = ksign
                    break
            if matched_sign is not None:
                obs_signs.append(matched_sign)
            else:
                obs_off = obs_pos[:2] - start_pos[:2]
                cross = line_dir[0] * obs_off[1] - line_dir[1] * obs_off[0]
                sign = +1 if cross >= 0 else -1
                _known_obs.append((obs_pos[:2].copy(), sign))
                obs_signs.append(sign)
                print(f"  [perception] new obs({obs_pos[0]:.1f},{obs_pos[1]:.1f}) "
                      f"r={obs_r:.2f} -> pass {'right' if cross >= 0 else 'left'}")

        t_global = time.time() - _session_t0

        # Homotopy-hint reference: goal pull + tangential side-bias, NO repulsion
        xref_h = _homotopy_horizon(x_now[:3], goal_pos, obstacles, obs_signs,
                                   MPC_N, MPC_TS, max_vel, tangent_gain)

        if _session:
            _session['t'].append(t_global)
            _session['x'].append(x_now.copy())
            _session['xref'].append(xref_h[0].copy())

        if time.time() >= t_next_mpc:
            # Obstacles fed to the MPC -> keep-out constraint enforces clearance.
            u_opt, info = _mpc.solve(x_now, xref_h, obstacles=obstacles,
                                     sign_correct=True)

            if _session:
                _session['u'].append(u_opt.copy())
                _session['mpc_ms'].append(info['solve_time_ms'])
                _session['n_obs'].append(info.get('n_obs_active', len(obstacles)))
                _session['slack'].append(info.get('slack_max', 0.0))

            wrench_to_rotorcraft(*u_opt)
            now = time.time()
            while t_next_mpc <= now:
                t_next_mpc += MPC_TS

        time.sleep(0.002)

    # Min clearance per discovered obstacle
    if _session and _known_obs:
        xs = np.array(_session['x'])
        for i, (kpos, ksign) in enumerate(_known_obs):
            dists = np.sqrt((xs[:, 0] - kpos[0])**2 + (xs[:, 1] - kpos[1])**2)
            print(f"  obs{i+1}@({kpos[0]:.1f},{kpos[1]:.1f}): "
                  f"min_dist={np.min(dists):.2f}m")

    # MPC-activity proof: fraction of steps where the keep-out was binding
    if _session and _session['slack']:
        sl = np.array(_session['slack'])
        active = float(np.mean(sl > 1e-6)) * 100
        print(f"  [MPC keep-out] slack>0 in {active:.0f}% of steps, "
              f"max slack={sl.max():.3f}  (slack>0 => MPC actively avoiding)")

    # Stop lidar before landing to avoid phantom obstacles during descent
    if hasattr(_perception, '_backend'):
        _perception.stop()

    landing()


def _quat_pitch(x):
    """Euler pitch from quaternion state vector."""
    qw, qx, qy, qz = x[6], x[7], x[8], x[9]
    return np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qx**2 + qy**2))


def backflip(alt: float = 10.0,
             tau_flip: float = 0.9,
             spinup_wait: float = 3.0,
             log_tag: str = 'backflip') -> None:
    """
    Backflip — Lupashin 5-phase bang-coast-bang approach.

    Phases:
      1. MPC climb to altitude and hover
      2. Open-loop pop-up impulse (gain upward velocity)
      3. Open-loop flip: accel(+tau) → coast(freefall) → decel(-tau)
      4. MPC recovery
      5. MPC landing

    Angle tracking uses quaternion-based Euler pitch (unwrapped),
    not body-rate integration, to handle gyroscopic coupling.

        >>> setup()
        >>> backflip()
    """
    global _flip_mode
    assert _mpc is not None, "Call setup() first"

    I_yy = I_DIAG[1]
    f_bang = 0.70 * MASS * G
    f_coast = 0.15 * MASS * G
    alpha_nom = tau_flip / I_yy

    theta_accel_end = np.radians(45)
    theta_target = 2 * np.pi
    theta_done_min = np.radians(340)
    alpha_eff_decel = alpha_nom * 0.60

    print(f"[backflip] Lupashin 5-phase (bang-coast-bang):")
    print(f"  tau={tau_flip:.2f} Nm  alpha_nom={alpha_nom:.1f} rad/s²")
    print(f"  f_bang={f_bang:.1f}N  f_coast={f_coast:.1f}N (rotor idle)")
    print(f"  Quat-pitch tracking: accel→coast at {np.degrees(theta_accel_end):.0f}°, "
          f"dynamic decel start")

    if alt < 5.0:
        print(f"[backflip] WARNING: need at least 5m altitude")
        return

    start()
    print(f"[backflip] waiting for spin-up ({spinup_wait:.0f}s)...")
    for i in range(int(spinup_wait), 0, -1):
        print(f"[backflip]   {i}s", end='\r')
        time.sleep(1.0)
    print()

    # Phase 1: climb to altitude
    curr_pos = _current_state()[:3]
    target = np.array([curr_pos[0], curr_pos[1], alt])
    traj_climb, T_climb = _build_goto_traj(curr_pos, target)

    _session_open(log_tag)
    _mpc.reset(_current_state())

    print(f"[backflip] Phase 1: climbing to z={alt:.1f}m")
    _run_loop(traj_climb, T_climb + 2.0)

    # Phase 2: pop-up impulse
    f_popup = 2.0 * MASS * G
    T_popup = 0.40
    print(f"[backflip] Phase 2: pop-up (f={f_popup:.1f}N, {T_popup:.2f}s)")

    t_popup_start = time.time()
    t_global_base = time.time() - _session_t0
    while time.time() - t_popup_start < T_popup:
        dt = time.time() - t_popup_start
        x_now = _current_state()
        if _session:
            _session['t'].append(t_global_base + dt)
            _session['x'].append(x_now.copy())
            _session['xref'].append(x_now.copy())
            _session['u'].append(np.array([f_popup, 0.0, 0.0, 0.0]))
            _session['mpc_ms'].append(0.0)
            _session['n_obs'].append(0)
        wrench_to_rotorcraft(f_popup, 0.0, 0.0, 0.0)
        time.sleep(0.002)

    x_popup = _current_state()
    print(f"[backflip] Post pop-up: z={x_popup[2]:.2f}m vz={x_popup[5]:.2f} m/s")

    # Phase 3: flip with body-rate integration
    print(f"[backflip] Phase 3: FLIP!")
    _flip_mode = True

    t_flip_start = time.time()
    t_global_base = time.time() - _session_t0
    prev_time = t_flip_start
    cumulative_pitch = 0.0
    phase = 'accel'

    while True:
        now = time.time()
        x_now = _current_state()

        dt = now - prev_time
        prev_time = now
        q_rate = x_now[11]
        cumulative_pitch += q_rate * dt

        if phase == 'accel' and cumulative_pitch >= theta_accel_end:
            phase = 'coast'
            print(f"[backflip]   -> coast at {np.degrees(cumulative_pitch):.0f}° "
                  f"q={q_rate:.1f}  t={now-t_flip_start:.3f}s")

        elif phase == 'coast' and q_rate > 0.1:
            decel_angle = q_rate**2 / (2 * alpha_eff_decel)
            remaining = theta_target - cumulative_pitch
            if remaining <= decel_angle:
                phase = 'decel'
                print(f"[backflip]   -> decel at {np.degrees(cumulative_pitch):.0f}° "
                      f"q={q_rate:.1f}  stop_dist={np.degrees(decel_angle):.0f}°  "
                      f"t={now-t_flip_start:.3f}s")

        if phase == 'decel':
            if q_rate < 1.0:
                print(f"[backflip]   -> done at {np.degrees(cumulative_pitch):.0f}° "
                      f"q={q_rate:.1f}  t={now-t_flip_start:.3f}s")
                break
            if cumulative_pitch >= np.radians(450):
                print(f"[backflip]   -> overshoot at {np.degrees(cumulative_pitch):.0f}° "
                      f"q={q_rate:.1f}  t={now-t_flip_start:.3f}s")
                break

        if now - t_flip_start > 3.0:
            print(f"[backflip]   -> TIMEOUT at {np.degrees(cumulative_pitch):.0f}° "
                  f"q={q_rate:.1f}")
            break

        if _session:
            _session['t'].append(t_global_base + (now - t_flip_start))
            _session['x'].append(x_now.copy())
            _session['xref'].append(x_now.copy())

        if phase == 'accel':
            f_cmd, tau_cmd = f_bang, tau_flip
        elif phase == 'coast':
            f_cmd, tau_cmd = f_coast, 0.0
        else:
            f_cmd, tau_cmd = f_bang, -tau_flip

        wrench_to_rotorcraft(f_cmd, 0.0, tau_cmd, 0.0)

        if _session:
            _session['u'].append(np.array([f_cmd, 0.0, tau_cmd, 0.0]))
            _session['mpc_ms'].append(0.0)
            _session['n_obs'].append(0)

        time.sleep(0.002)

    x_after = _current_state()
    print(f"[backflip] Post-flip: z={x_after[2]:.2f}m  vz={x_after[5]:.2f}  "
          f"qw={x_after[6]:.3f}  pitch_rate={x_after[11]:.1f} rad/s")

    # Phase 4: SO(3) recovery with lateral velocity damping
    # MPC doesn't converge after flip — SO(3) handles full recovery.
    # Tilts against velocity to decelerate, not just level attitude.
    _flip_mode = False
    print(f"[backflip] Phase 4: SO3 recovery (attitude + velocity damping)")

    Kp_att = 1.5
    Kd_att = 1.2
    K_vz   = 3.0
    K_vxy  = 0.10
    tau_budget = tau_flip

    t_rec_start = time.time()
    phase4_settled = False
    while time.time() - t_rec_start < 6.0:
        x_now = _current_state()
        omega = x_now[10:13]
        vx, vy, vz = x_now[3], x_now[4], x_now[5]

        qw, qx, qy, qz = x_now[6], x_now[7], x_now[8], x_now[9]
        if qw < 0:
            qw, qx, qy, qz = -qw, -qx, -qy, -qz

        qx_des = np.clip(K_vxy * vy, -0.20, 0.20)
        qy_des = np.clip(-K_vxy * vx, -0.20, 0.20)

        q_err_vec = np.array([qx - qx_des, qy - qy_des, qz])
        q_err_norm = np.linalg.norm(q_err_vec)
        if q_err_norm > 1e-6:
            theta_err = 2.0 * np.arctan2(q_err_norm, qw)
            axis = q_err_vec / q_err_norm
        else:
            theta_err = 0.0
            axis = np.array([0.0, 0.0, 1.0])

        tau_vec = -Kp_att * theta_err * axis - Kd_att * omega
        tau_vec[2] = np.clip(tau_vec[2], -0.1, 0.1)
        tau_norm = np.linalg.norm(tau_vec)
        if tau_norm > tau_budget:
            tau_vec = tau_vec / tau_norm * tau_budget

        f_up = MASS * G + np.clip(-K_vz * vz, -MASS * G, MASS * G)
        f_cmd = np.clip(f_up, 0.5 * MASS * G, 2.0 * MASS * G)

        wrench_to_rotorcraft(f_cmd, tau_vec[0], tau_vec[1], tau_vec[2])

        if _session:
            t_global = time.time() - _session_t0
            _session['t'].append(t_global)
            _session['x'].append(x_now.copy())
            _session['xref'].append(x_now.copy())
            _session['u'].append(np.array([f_cmd, tau_vec[0], tau_vec[1], tau_vec[2]]))
            _session['mpc_ms'].append(0.0)
            _session['n_obs'].append(0)

        dt = time.time() - t_rec_start
        v_lateral = np.sqrt(vx**2 + vy**2)
        if not phase4_settled and dt > 0.5 and theta_err < 0.2 and np.linalg.norm(omega) < 0.5:
            print(f"[backflip]   upright at t={dt:.2f}s, v_lat={v_lateral:.2f} — decelerating")
            phase4_settled = True

        if phase4_settled and v_lateral < 0.5 and abs(vz) < 0.5:
            print(f"[backflip]   hovering: v_lat={v_lateral:.2f} vz={vz:.2f} "
                  f"z={x_now[2]:.2f}  t={dt:.2f}s")
            break

        time.sleep(0.002)

    x_rec = _current_state()
    v_lat = np.sqrt(x_rec[3]**2 + x_rec[4]**2)
    print(f"[backflip] Post-recovery: z={x_rec[2]:.2f}m  vz={x_rec[5]:.2f}  "
          f"v_lat={v_lat:.2f}  xy=({x_rec[0]:.1f},{x_rec[1]:.1f})")

    # Phase 5: MPC return to origin + landing
    # Drone is now hovering — MPC should work from clean state
    print(f"[backflip] Phase 5: MPC return to origin")
    _mpc.reset_weights()
    _mpc.reset_bounds()
    _mpc.reset(_current_state())
    curr = _current_state()[:3]
    origin = np.array([0.0, 0.0, alt])
    dist = np.linalg.norm(curr - origin)
    T_return = max(dist / 2.0, 3.0)
    traj_return = WaypointTrajectory(
        [{'pos': curr, 'vel': [0,0,0]},
         {'pos': origin, 'vel': [0,0,0]}],
        seg_times=[T_return])
    _run_loop(traj_return, T_return + 3.0)

    print(f"[backflip] Phase 6: landing")
    final = _current_state()[:3]
    print(f"  Position: ({final[0]:.2f}, {final[1]:.2f}, {final[2]:.2f})")
    landing()


def backflip_ilc(alt: float = 10.0, spinup_wait: float = 3.0,
                 log_tag: str = '') -> None:
    """Backflip with tuned params and PD position-feedback recovery."""
    global _flip_mode
    if not log_tag:
        log_tag = f"backflip_ilc_{time.strftime('%H%M%S')}"
    assert _mpc is not None, "Call setup() first"

    f_accel    = 8.77
    f_decel    = 8.81
    f_popup    = 2.0 * MASS * G
    T_popup    = 0.40
    f_coast    = 0.15 * MASS * G
    tau_flip   = 0.9
    I_yy       = I_DIAG[1]
    alpha_nom  = tau_flip / I_yy
    theta_accel_end  = np.radians(45)
    theta_target     = 2 * np.pi
    alpha_eff_decel  = alpha_nom * 0.60
    vx_bias    = 0.0
    vy_bias    = 0.0
    T_burst    = 0.30

    print(f"[flip] f_accel={f_accel:.2f}N  f_decel={f_decel:.2f}N  "
          f"bias=({vx_bias:+.3f},{vy_bias:+.3f})")

    if alt < 5.0:
        print("[flip] WARNING: need at least 5m altitude")
        return

    start()
    print(f"[flip] spin-up ({spinup_wait:.0f}s)...")
    for i in range(int(spinup_wait), 0, -1):
        print(f"  {i}s", end='\r')
        time.sleep(1.0)
    print()

    # Phase 1: MPC climb
    curr_pos = _current_state()[:3]
    target = np.array([curr_pos[0], curr_pos[1], alt])
    traj_climb, T_climb = _build_goto_traj(curr_pos, target)
    _session_open(log_tag)
    _mpc.reset(_current_state())
    print(f"[flip] Phase 1: climb to z={alt:.1f}m")
    _run_loop(traj_climb, T_climb + 2.0)

    # Stabilize hover before flip
    hover_state = _current_state()
    hover_pos = hover_state[:3].copy()
    print(f"[flip] Hover: ({hover_pos[0]:.2f}, {hover_pos[1]:.2f}, {hover_pos[2]:.2f})")

    # Phase 2: pop-up
    print(f"[flip] Phase 2: pop-up (f={f_popup:.1f}N, {T_popup:.3f}s)")
    t_popup_start = time.time()
    t_global_base = time.time() - _session_t0
    while time.time() - t_popup_start < T_popup:
        dt_p = time.time() - t_popup_start
        x_now = _current_state()
        if _session:
            _session['t'].append(t_global_base + dt_p)
            _session['x'].append(x_now.copy())
            _session['xref'].append(x_now.copy())
            _session['u'].append(np.array([f_popup, 0.0, 0.0, 0.0]))
            _session['mpc_ms'].append(0.0)
            _session['n_obs'].append(0)
        wrench_to_rotorcraft(f_popup, 0.0, 0.0, 0.0)
        time.sleep(0.002)

    x_popup = _current_state()
    print(f"[flip] Post pop-up: z={x_popup[2]:.2f}m vz={x_popup[5]:.2f} m/s")

    # Phase 3: flip
    print(f"[flip] Phase 3: FLIP!")
    _flip_mode = True
    t_flip_start = time.time()
    t_global_base = time.time() - _session_t0
    prev_time = t_flip_start
    cumulative_pitch = 0.0
    phase = 'accel'

    while True:
        now = time.time()
        x_now = _current_state()
        dt_f = now - prev_time
        prev_time = now
        q_rate = x_now[11]
        cumulative_pitch += q_rate * dt_f

        if phase == 'accel' and cumulative_pitch >= theta_accel_end:
            phase = 'coast'
            print(f"[flip]   -> coast at {np.degrees(cumulative_pitch):.0f}° "
                  f"q={q_rate:.1f}  t={now-t_flip_start:.3f}s")
        elif phase == 'coast' and q_rate > 0.1:
            decel_angle = q_rate**2 / (2 * alpha_eff_decel)
            remaining = theta_target - cumulative_pitch
            if remaining <= decel_angle:
                phase = 'decel'
                print(f"[flip]   -> decel at {np.degrees(cumulative_pitch):.0f}° "
                      f"q={q_rate:.1f}  stop={np.degrees(decel_angle):.0f}°  "
                      f"t={now-t_flip_start:.3f}s")

        if phase == 'decel':
            if q_rate < 1.0:
                print(f"[flip]   -> done at {np.degrees(cumulative_pitch):.0f}° "
                      f"q={q_rate:.1f}  t={now-t_flip_start:.3f}s")
                break
            if cumulative_pitch >= np.radians(450):
                print(f"[flip]   -> overshoot at {np.degrees(cumulative_pitch):.0f}°")
                break

        if now - t_flip_start > 3.0:
            print(f"[flip]   -> TIMEOUT at {np.degrees(cumulative_pitch):.0f}° q={q_rate:.1f}")
            break

        if _session:
            _session['t'].append(t_global_base + (now - t_flip_start))
            _session['x'].append(x_now.copy())
            _session['xref'].append(x_now.copy())

        if phase == 'accel':
            f_cmd, tau_cmd = f_accel, tau_flip
        elif phase == 'coast':
            f_cmd, tau_cmd = f_coast, 0.0
        else:
            f_cmd, tau_cmd = f_decel, -tau_flip

        wrench_to_rotorcraft(f_cmd, 0.0, tau_cmd, 0.0)

        if _session:
            _session['u'].append(np.array([f_cmd, 0.0, tau_cmd, 0.0]))
            _session['mpc_ms'].append(0.0)
            _session['n_obs'].append(0)
        time.sleep(0.002)

    x_post = _current_state()
    print(f"[flip] Post-flip: qw={x_post[6]:.3f}  "
          f"vel=({x_post[3]:+.2f},{x_post[4]:+.2f},{x_post[5]:+.2f})")

    # Phase 4: SO(3) recovery — PD position + velocity + bias
    _flip_mode = False
    Kp_att = 1.5;  Kd_att = 1.2;  K_vz = 3.0
    K_vxy = 0.10;  K_pos = 0.05
    tau_budget = tau_flip

    t_rec_start = time.time()
    phase4_settled = False
    while time.time() - t_rec_start < 6.0:
        x_now = _current_state()
        omega = x_now[10:13]
        vx, vy, vz = x_now[3], x_now[4], x_now[5]
        qw, qx, qy, qz = x_now[6], x_now[7], x_now[8], x_now[9]
        if qw < 0:
            qw, qx, qy, qz = -qw, -qx, -qy, -qz

        dt_r = time.time() - t_rec_start
        dx = x_now[0] - hover_pos[0]
        dy = x_now[1] - hover_pos[1]
        bx = vx_bias if dt_r < T_burst else 0.0
        by = vy_bias if dt_r < T_burst else 0.0
        qx_des = np.clip(K_vxy * (vy + by) + K_pos * dy, -0.20, 0.20)
        qy_des = np.clip(-K_vxy * (vx + bx) - K_pos * dx, -0.20, 0.20)

        q_err_vec = np.array([qx - qx_des, qy - qy_des, qz])
        q_err_norm = np.linalg.norm(q_err_vec)
        if q_err_norm > 1e-6:
            theta_err = 2.0 * np.arctan2(q_err_norm, qw)
            axis = q_err_vec / q_err_norm
        else:
            theta_err = 0.0
            axis = np.array([0.0, 0.0, 1.0])

        tau_vec = -Kp_att * theta_err * axis - Kd_att * omega
        tau_vec[2] = np.clip(tau_vec[2], -0.1, 0.1)
        tau_norm = np.linalg.norm(tau_vec)
        if tau_norm > tau_budget:
            tau_vec = tau_vec / tau_norm * tau_budget

        f_up = MASS * G + np.clip(-K_vz * vz, -MASS * G, MASS * G)
        f_cmd = np.clip(f_up, 0.5 * MASS * G, 2.0 * MASS * G)
        wrench_to_rotorcraft(f_cmd, tau_vec[0], tau_vec[1], tau_vec[2])

        if _session:
            t_global = time.time() - _session_t0
            _session['t'].append(t_global)
            _session['x'].append(x_now.copy())
            _session['xref'].append(x_now.copy())
            _session['u'].append(np.array([f_cmd, tau_vec[0], tau_vec[1], tau_vec[2]]))
            _session['mpc_ms'].append(0.0)
            _session['n_obs'].append(0)

        v_lateral = np.sqrt(vx**2 + vy**2)
        if not phase4_settled and dt_r > 0.5 and theta_err < 0.2 and np.linalg.norm(omega) < 0.5:
            print(f"[flip]   upright at t={dt_r:.2f}s, v_lat={v_lateral:.2f}")
            phase4_settled = True
        if phase4_settled and v_lateral < 0.5 and abs(vz) < 0.5:
            print(f"[flip]   hovering: v_lat={v_lateral:.2f} vz={vz:.2f} z={x_now[2]:.2f}")
            break
        time.sleep(0.002)

    x_rec = _current_state()
    drift_x = x_rec[0] - hover_pos[0]
    drift_y = x_rec[1] - hover_pos[1]
    v_lat = np.sqrt(x_rec[3]**2 + x_rec[4]**2)
    print(f"[flip] Recovery: drift=({drift_x:+.1f},{drift_y:+.1f})m  "
          f"v_lat={v_lat:.2f}  z={x_rec[2]:.2f}")

    # Phase 5: MPC return + landing
    _mpc.reset_weights()
    _mpc.reset_bounds()
    _mpc.reset(_current_state())
    curr = _current_state()[:3]
    origin = np.array([0.0, 0.0, alt])
    dist = np.linalg.norm(curr - origin)
    T_return = max(dist / 2.0, 3.0)
    traj_return = WaypointTrajectory(
        [{'pos': curr, 'vel': [0,0,0]},
         {'pos': origin, 'vel': [0,0,0]}],
        seg_times=[T_return])
    print(f"[flip] Phase 5: return to origin ({dist:.1f}m)")
    _run_loop(traj_return, T_return + 3.0)

    final = _current_state()[:3]
    print(f"[flip] Landing from ({final[0]:.2f}, {final[1]:.2f}, {final[2]:.2f})")
    landing()


def backflip_mpc_recovery(alt: float = 10.0,
             tau_flip: float = 1.1,   # drift-tune: 0.9->1.1, faster flip = less time tilted = less drift
             spinup_wait: float = 3.0,
             log_tag: str = 'backflip_mpc_rec') -> None:
    """
    Backflip (first-pushed version) — MPC-based recovery.
    Lupashin 5-phase bang-coast-bang; Phase 4a brief open-loop rate-kill,
    Phase 4b genuine MPC recovery with boosted damping weights.

    Phases:
      1. MPC climb to altitude and hover
      2. Open-loop pop-up impulse (gain upward velocity)
      3. Open-loop flip: accel(+tau) → coast(freefall) → decel(-tau)
      4. MPC recovery
      5. MPC landing

    Angle tracking uses quaternion-based Euler pitch (unwrapped),
    not body-rate integration, to handle gyroscopic coupling.

        >>> setup()
        >>> backflip()
    """
    global _flip_mode
    assert _mpc is not None, "Call setup() first"

    I_yy = I_DIAG[1]
    f_bang = 0.70 * MASS * G
    f_coast = 0.15 * MASS * G
    alpha_nom = tau_flip / I_yy

    theta_accel_end = np.radians(45)
    theta_decel_start = np.radians(270)
    theta_done = np.radians(300)

    print(f"[bf_mpcrec] Lupashin 5-phase (bang-coast-bang):")
    print(f"  tau={tau_flip:.2f} Nm  alpha_nom={alpha_nom:.1f} rad/s²")
    print(f"  f_bang={f_bang:.1f}N  f_coast={f_coast:.1f}N (rotor idle)")
    print(f"  Quat-pitch tracking: accel→coast at {np.degrees(theta_accel_end):.0f}°, "
          f"coast→decel at {np.degrees(theta_decel_start):.0f}°")

    if alt < 5.0:
        print(f"[bf_mpcrec] WARNING: need at least 5m altitude")
        return

    start()
    print(f"[bf_mpcrec] waiting for spin-up ({spinup_wait:.0f}s)...")
    for i in range(int(spinup_wait), 0, -1):
        print(f"[bf_mpcrec]   {i}s", end='\r')
        time.sleep(1.0)
    print()

    # Phase 1: climb to altitude
    curr_pos = _current_state()[:3]
    target = np.array([curr_pos[0], curr_pos[1], alt])
    traj_climb, T_climb = _build_goto_traj(curr_pos, target)

    _session_open(log_tag)
    _mpc.reset(_current_state())

    print(f"[bf_mpcrec] Phase 1: climbing to z={alt:.1f}m")
    _run_loop(traj_climb, T_climb + 2.0)

    # Hover point captured before the flip — return target after recovery.
    hover_pos = _current_state()[:3].copy()
    print(f"[bf_mpcrec] Hover before flip: "
          f"({hover_pos[0]:.2f},{hover_pos[1]:.2f},{hover_pos[2]:.2f})")

    # Phase 2: pop-up impulse
    f_popup = 2.0 * MASS * G
    T_popup = 0.40
    print(f"[bf_mpcrec] Phase 2: pop-up (f={f_popup:.1f}N, {T_popup:.2f}s)")

    t_popup_start = time.time()
    t_global_base = time.time() - _session_t0
    while time.time() - t_popup_start < T_popup:
        dt = time.time() - t_popup_start
        x_now = _current_state()
        if _session:
            _session['t'].append(t_global_base + dt)
            _session['x'].append(x_now.copy())
            _session['xref'].append(x_now.copy())
            _session['u'].append(np.array([f_popup, 0.0, 0.0, 0.0]))
            _session['mpc_ms'].append(0.0)
            _session['n_obs'].append(0)
        wrench_to_rotorcraft(f_popup, 0.0, 0.0, 0.0)
        time.sleep(0.002)

    x_popup = _current_state()
    print(f"[bf_mpcrec] Post pop-up: z={x_popup[2]:.2f}m vz={x_popup[5]:.2f} m/s")

    # Phase 3: flip with quaternion-based pitch tracking
    print(f"[bf_mpcrec] Phase 3: FLIP!")
    _flip_mode = True

    t_flip_start = time.time()
    t_global_base = time.time() - _session_t0
    prev_pitch = _quat_pitch(_current_state())
    cumulative_pitch = 0.0
    phase = 'accel'

    while True:
        now = time.time()
        x_now = _current_state()

        curr_pitch = _quat_pitch(x_now)
        d_pitch = curr_pitch - prev_pitch
        if d_pitch > np.pi:
            d_pitch -= 2 * np.pi
        elif d_pitch < -np.pi:
            d_pitch += 2 * np.pi
        cumulative_pitch += d_pitch
        prev_pitch = curr_pitch

        q_rate = x_now[11]

        if phase == 'accel' and cumulative_pitch >= theta_accel_end:
            phase = 'coast'
            print(f"[bf_mpcrec]   -> coast at {np.degrees(cumulative_pitch):.0f}° "
                  f"q={q_rate:.1f}  t={now-t_flip_start:.3f}s")

        elif phase == 'coast' and cumulative_pitch >= theta_decel_start:
            phase = 'decel'
            print(f"[bf_mpcrec]   -> decel at {np.degrees(cumulative_pitch):.0f}° "
                  f"q={q_rate:.1f}  t={now-t_flip_start:.3f}s")

        if phase == 'decel':
            if cumulative_pitch >= theta_done and abs(q_rate) < 1.5:
                print(f"[bf_mpcrec]   -> done at {np.degrees(cumulative_pitch):.0f}° "
                      f"q={q_rate:.1f}  t={now-t_flip_start:.3f}s")
                break
            if cumulative_pitch >= np.radians(300) and q_rate < -1.0:
                print(f"[bf_mpcrec]   -> over-braked at {np.degrees(cumulative_pitch):.0f}° "
                      f"q={q_rate:.1f}  t={now-t_flip_start:.3f}s")
                break
            if cumulative_pitch >= np.radians(420):
                print(f"[bf_mpcrec]   -> overshoot at {np.degrees(cumulative_pitch):.0f}° "
                      f"q={q_rate:.1f}  t={now-t_flip_start:.3f}s")
                break

        if now - t_flip_start > 3.0:
            print(f"[bf_mpcrec]   -> TIMEOUT at {np.degrees(cumulative_pitch):.0f}° "
                  f"q={q_rate:.1f}")
            break

        if _session:
            _session['t'].append(t_global_base + (now - t_flip_start))
            _session['x'].append(x_now.copy())
            _session['xref'].append(x_now.copy())

        if phase == 'accel':
            f_cmd, tau_cmd = f_bang, tau_flip
        elif phase == 'coast':
            f_cmd, tau_cmd = f_coast, 0.0
        else:
            f_cmd, tau_cmd = f_bang, -tau_flip

        wrench_to_rotorcraft(f_cmd, 0.0, tau_cmd, 0.0)

        if _session:
            _session['u'].append(np.array([f_cmd, 0.0, tau_cmd, 0.0]))
            _session['mpc_ms'].append(0.0)
            _session['n_obs'].append(0)

        time.sleep(0.002)

    x_after = _current_state()
    print(f"[bf_mpcrec] Post-flip: z={x_after[2]:.2f}m  vz={x_after[5]:.2f}  "
          f"qw={x_after[6]:.3f}  pitch_rate={x_after[11]:.1f} rad/s")

    # Phase 4a: open-loop rate kill (P-controller on pitch rate)
    _flip_mode = False
    print(f"[bf_mpcrec] Phase 4a: rate kill")
    K_rate = 0.04
    t_rate_start = time.time()
    while time.time() - t_rate_start < 0.8:
        x_now = _current_state()
        q_rate = x_now[11]
        p_rate = x_now[10]
        r_rate = x_now[12]

        tau_y = np.clip(-K_rate * q_rate, -tau_flip, tau_flip)
        tau_x = np.clip(-K_rate * p_rate, -0.3, 0.3)
        tau_z = np.clip(-0.02 * r_rate, -0.06, 0.06)

        pitch_err = _quat_pitch(x_now)
        f_hold = MASS * G * max(np.cos(pitch_err), 0.3)

        wrench_to_rotorcraft(f_hold, tau_x, tau_y, tau_z)

        if _session:
            t_global = time.time() - _session_t0
            _session['t'].append(t_global)
            _session['x'].append(x_now.copy())
            _session['xref'].append(x_now.copy())
            _session['u'].append(np.array([f_hold, tau_x, tau_y, tau_z]))
            _session['mpc_ms'].append(0.0)
            _session['n_obs'].append(0)

        if abs(q_rate) < 0.5 and abs(p_rate) < 0.5:
            print(f"[bf_mpcrec]   rates killed: q={q_rate:.2f} p={p_rate:.2f} "
                  f"t={time.time()-t_rate_start:.3f}s")
            break
        time.sleep(0.002)

    x_post_kill = _current_state()
    print(f"[bf_mpcrec] Post-kill: z={x_post_kill[2]:.2f}m  "
          f"q={x_post_kill[11]:.2f}  p={x_post_kill[10]:.2f}  "
          f"qw={x_post_kill[6]:.3f}")

    # Phase 4b: MPC recovery with boosted damping
    print(f"[bf_mpcrec] Phase 4b: MPC recovery")
    _mpc.reset_bounds()
    _mpc.set_bounds(f_min=0.0, tau_max=0.40)

    W_recover = _mpc._W.copy()
    W_recover[10, 10] = 80.0   # Q_omega p
    W_recover[11, 11] = 80.0   # Q_omega q
    W_recover[12, 12] = 20.0   # Q_omega_r
    W_recover[6, 6]   = 5.0    # Q_att qw
    W_recover[7, 7]   = 5.0    # Q_att qx
    W_recover[8, 8]   = 5.0    # Q_att qy
    W_recover[9, 9]   = 5.0    # Q_att qz
    WN_recover = W_recover[:13, :13].copy()
    WN_recover[10, 10] = 40.0
    WN_recover[11, 11] = 40.0
    WN_recover[12, 12] = 10.0
    _mpc.set_weights(W=W_recover, WN=WN_recover)
    _mpc.reset(_current_state())

    recover_pos = _current_state()[:3]
    recover_target = np.array([recover_pos[0], recover_pos[1],
                               max(recover_pos[2], 3.0)])
    traj_recover = WaypointTrajectory(
        [{'pos': recover_pos, 'vel': [0,0,0]},
         {'pos': recover_target, 'vel': [0,0,0]}],
        seg_times=[4.0])

    # f_min_clamp=0.0: recovery set_bounds(f_min=0.0) must not be overridden
    # by the _run_loop 0.60*mg safety floor, else thrust can't drop while
    # righting from inversion (this floor was added after the first push and
    # is what broke the original MPC recovery).
    _run_loop(traj_recover, 6.0, f_min_clamp=0.0)
    _mpc.reset_weights()
    _mpc.reset_bounds()
    _mpc.reset(_current_state())

    # Phase 5: MPC return to the pre-flip hover point, hold to settle
    rec = _current_state()[:3]
    print(f"[bf_mpcrec] Phase 5: return to hover point "
          f"(from ({rec[0]:.2f},{rec[1]:.2f},{rec[2]:.2f}))")
    return_vel = 3.0                          # faster than cruise to cut return time
    return_target = np.array([hover_pos[0], hover_pos[1], alt])
    dist = np.linalg.norm(rec - return_target)
    T_return = max(dist / return_vel, 2.5)
    traj_return = WaypointTrajectory(
        [{'pos': rec,           'vel': [0, 0, 0]},
         {'pos': return_target, 'vel': [0, 0, 0]}],
        seg_times=[T_return])
    _run_loop(traj_return, T_return + 2.5)    # +2.5s hold to converge above start

    print(f"[bf_mpcrec] Phase 6: landing at hover point")
    final = _current_state()[:3]
    print(f"  Position: ({final[0]:.2f}, {final[1]:.2f}, {final[2]:.2f})  "
          f"target=({hover_pos[0]:.2f},{hover_pos[1]:.2f})")
    landing()


if __name__ == '__main__':
    pass
