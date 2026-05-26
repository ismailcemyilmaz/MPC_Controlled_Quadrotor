"""
quadrotor_mpc_client.py  (perception entegrasyonlu versiyon)
============================================================
Değişiklikler:
  - QuadrotorMPC → LocalPlannerMPC
  - perception.PerceptionManager entegre edildi
  - _run_loop: obstacles parametresi
  - setup(): perception başlatılıyor
  - Yeni public API: set_obstacle_level(), add_obstacle()
"""

import time
import os
import numpy as np

from MPC_Lidar.quadrotor_mpc_client_v3 import landing
from local_planner_mpc import LocalPlannerMPC
from global_planner import WaypointTrajectory
from perception import PerceptionManager

try:
    import genomix
    GENOMIX_AVAILABLE = True
except ImportError:
    GENOMIX_AVAILABLE = False
    print("[client] genomix not found — simulation only")

# ── Fiziksel sabitler ────────────────────────────────────────────────────────
G       = 9.81
MASS    = 1.280
I_DIAG  = (22.916e-3, 22.916e-3, 22.132e-3)
ARM_LEN = 0.23
KF      = 6.5e-4
KM      = 1e-5

# ── MPC konfigürasyonu ───────────────────────────────────────────────────────
MPC_N  = 20
MPC_TS = 0.025

_MPC_KWARGS = dict(
    Q_pos=5.0,     Q_vel=2.0,
    Q_att=6.0,
    Q_omega=1.5,   Q_omega_r=3.0,    # Run 4: limit cycle düzeltmesi
    P_scale=5.0,
    R_f=0.02,      R_tau=0.10,       R_tau_z=0.15,
    tau_max=0.80,  tau_z_max=0.12, #tau_z_max=0.35 -> 0.45
    f_min=0.25*MASS*G,
    f_max_scale=2.5,
    alpha_land=2.0, W_land=500.0,
)

_LOCAL_MPC_KWARGS = dict(
    n_obs_max  = 5,
    R_drone    = 0.30,
    W_obs      = 10000.0,
)

# ── Perception konfigürasyonu ────────────────────────────────────────────────
# Gazebo world'deki engel model isimleri ve yarıçapları
# World dosyasına eklediğin her model için buraya bir satır ekle:
OBSTACLE_MODELS = [
    # ('model_ismi_world_dosyasinda', yarıçap_metre)
    # Örnek:
    # ('obstacle_cylinder_1', 0.35),
    # ('obstacle_box_1',      0.40),
]

# ── Uçuş parametreleri ───────────────────────────────────────────────────────
GROUND_Z   = 0.10
MAX_VEL    = 1.5
MIN_SAFE_Z = 1.5

_HERE   = os.path.dirname(os.path.abspath(__file__))
_WS     = os.path.normpath(os.path.join(_HERE, '..', '..'))
LOG_DIR = os.path.join(_WS, 'logs', 'mpc')

# ── Modül state ──────────────────────────────────────────────────────────────
_g          = None
_pom        = None
_rotorcraft = None
_optitrack  = None
_mpc        = None
_armed      = False
_perception = None     # PerceptionManager instance


# ── State dönüşümü ───────────────────────────────────────────────────────────
def pom_to_state(frame) -> np.ndarray:
    px = frame['pos']['x'];  py = frame['pos']['y'];  pz = frame['pos']['z']
    vx = frame['vel']['vx']; vy = frame['vel']['vy']; vz = frame['vel']['vz']
    qw = frame['att']['qw']; qx = frame['att']['qx']
    qy = frame['att']['qy']; qz = frame['att']['qz']
    norm = (qw**2 + qx**2 + qy**2 + qz**2)**0.5 + 1e-12
    qw /= norm; qx /= norm; qy /= norm; qz /= norm
    p = frame['avel']['wx']; q = frame['avel']['wy']; r = frame['avel']['wz']
    return np.array([px, py, pz, vx, vy, vz, qw, qx, qy, qz, p, q, r])

def _current_state() -> np.ndarray:
    return pom_to_state(_pom.frame('robot')['frame'])


# ── Motor mixer ──────────────────────────────────────────────────────────────
def wrench_to_rotorcraft(f, tau_x, tau_y, tau_z):
    M = np.array([
        [ KF,            KF,           KF,           KF      ],
        [ 0,             ARM_LEN*KF,   0,           -ARM_LEN*KF],
        [-ARM_LEN*KF,    0,            ARM_LEN*KF,   0       ],
        [ KM,           -KM,           KM,          -KM      ],
    ])
    omega_sq = np.linalg.pinv(M) @ np.array([f, tau_x, tau_y, tau_z])
    omega_sq = np.clip(omega_sq, 0.0, None)
    omega    = np.sqrt(omega_sq)
    omega    = np.clip(omega, 0.0, 1200.0)
    _rotorcraft.set_velocity({'desired': [
        float(omega[0]), float(omega[1]),
        float(omega[2]), float(omega[3]),
        0.0, 0.0, 0.0, 0.0,
    ]})


# ── Trakjekteri builder ──────────────────────────────────────────────────────
def _build_goto_traj(curr_pos, target_pos, max_vel=MAX_VEL):
    cx, cy, cz = curr_pos
    tx, ty, tz = target_pos
    is_landing = (tz <= GROUND_Z)

    if is_landing:
        horiz_dist = np.linalg.norm(np.array([tx,ty]) - np.array([cx,cy]))
        safe_z     = max(cz, MIN_SAFE_Z)
        T_descend  = max(safe_z / 0.50, 4.0)

        if horiz_dist < 0.3:
            waypoints = [
                {'pos': np.array([cx,cy,cz]), 'vel': [0,0,0]},
                {'pos': np.array([tx,ty,0.0]),'vel': [0,0,0]},
            ]
            seg_times = [T_descend]
        else:
            T_horiz = max(horiz_dist / max_vel, 1.5)
            waypoints = [
                {'pos': np.array([cx,cy,cz]),         'vel': [0,0,0]},
                {'pos': np.array([tx,ty,safe_z]),      'vel': [0,0,0]},
                {'pos': np.array([tx,ty,0.0]),         'vel': [0,0,0]},
            ]
            seg_times = [T_horiz, T_descend]
    else:
        dist     = np.linalg.norm(target_pos - curr_pos)
        T_travel = max(2.5 * dist / max_vel, 4.0)
        waypoints = [
            {'pos': curr_pos,   'vel': [0,0,0]},
            {'pos': target_pos, 'vel': [0,0,0]},
        ]
        seg_times = [T_travel]

    traj = WaypointTrajectory(waypoints, seg_times=seg_times)
    return traj, traj.total_duration()


# ── Kontrol döngüsü ──────────────────────────────────────────────────────────
def _run_loop(traj: WaypointTrajectory,
              T_total: float,
              log_tag: str = 'flight') -> None:
    """MPC kontrol döngüsü — perception entegreli."""
    _mpc.reset(_current_state())
    os.makedirs(os.path.join(LOG_DIR, log_tag), exist_ok=True)

    log_t, log_x, log_u, log_xref, log_mpc_ms = [], [], [], [], []
    log_n_obs = []

    t_start    = time.time()
    t_next_mpc = t_start

    while True:
        t_now = time.time() - t_start
        if t_now > T_total:
            break

        x_now     = _current_state()
        x_ref_now = traj.state_at(t_now)

        log_t.append(t_now)
        log_x.append(x_now.copy())
        log_xref.append(x_ref_now.copy())

        if time.time() >= t_next_mpc:
            # ── Perception: anlık engel listesi ─────────────────────────────
            if _perception is not None:
                _perception.update_drone_pos(x_now[:3])
                obstacles = _perception.get_obstacles()
            else:
                obstacles = []

            log_n_obs.append(len(obstacles))

            # ── MPC çöz ─────────────────────────────────────────────────────
            xref_h      = traj.get_horizon(t_now, MPC_N, MPC_TS)
            u_opt, info = _mpc.solve(x_now, xref_h, obstacles=obstacles)

            log_u.append(u_opt.copy())
            log_mpc_ms.append(info['solve_time_ms'])

            if info.get('slack_max', 0) > 1e-2:
                print(f"[run] t={t_now:.1f}s  "
                      f"obs_slack={info['slack_max']:.3f}  "
                      f"n_obs={len(obstacles)}")

            wrench_to_rotorcraft(*u_opt)
            t_next_mpc += MPC_TS

        time.sleep(0.002)

    # ── Log kaydet ───────────────────────────────────────────────────────────
    fname = os.path.join(LOG_DIR, log_tag, 'mpc_log.npz')
    if log_t:
        np.savez(fname,
                 t=np.array(log_t),         x=np.array(log_x),
                 u=np.array(log_u),         xref=np.array(log_xref),
                 mpc_times=np.array(log_mpc_ms),
                 n_obs=np.array(log_n_obs))
        avg_ms = np.mean(log_mpc_ms) if log_mpc_ms else 0
        avg_obs = np.mean(log_n_obs) if log_n_obs else 0
        print(f"[run] Log → {fname}  "
              f"(MPC mean={avg_ms:.1f}ms  avg_obs={avg_obs:.1f})")

        # Satürasyon raporu
        _tau_max = _MPC_KWARGS['tau_max']
        u_arr = np.array(log_u)
        if len(u_arr):
            st = np.mean(np.any(np.abs(u_arr[:,1:3]) >= _tau_max*0.99, axis=1))
            print(f"[sat] τ={st*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def setup(perception_level: int = 1):
    """
    Sensörleri bağla, MPC ve Perception'ı başlat.

    perception_level:
        1 = Gazebo ground truth  (gz CLI/transport)
        2 = 2D Lidar
        3 = Statik engeller (default, sensor gerektirmez)
    """
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
    _rotorcraft.set_sensor_rate({
        'rate': {'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1}
    })
    _rotorcraft.set_imu_filter({
        'gfc': [20,20,20], 'afc': [5,5,5], 'mfc': [20,20,20]
    })
    _pom.set_prediction_model('::pom::constant_acceleration')
    _pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})
    _pom.set_history_length({'history_length': 0.25})
    _pom.set_mag_field({
        'magdir': {'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06}
    })
    _pom.connect_port({'local': 'measure/imu',   'remote': 'rotorcraft/imu'})
    _pom.add_measurement('imu')
    _pom.connect_port({'local': 'measure/mag',   'remote': 'rotorcraft/mag'})
    _pom.add_measurement('mag')
    _pom.connect_port({'local': 'measure/mocap', 'remote': 'optitrack/bodies/QR_4'})
    _pom.add_measurement('mocap')

    os.makedirs(LOG_DIR, exist_ok=True)

    # ── MPC ──────────────────────────────────────────────────────────────────
    _mpc = LocalPlannerMPC(
        N=MPC_N, Ts=MPC_TS, mass=MASS, I_diag=I_DIAG,
        rk4_steps=1,
        **_LOCAL_MPC_KWARGS,
        **_MPC_KWARGS,
    )

    # ── Perception ───────────────────────────────────────────────────────────
    if perception_level == 1:
        _perception = PerceptionManager(
            level=1,
            obstacle_models=OBSTACLE_MODELS,
            update_rate_hz=10.0,
        )
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
                 max_vel: float = MAX_VEL) -> None:
    assert _armed, "Call start() first"
    target   = np.array([float(x), float(y), float(z)])
    curr_pos = _current_state()[:3]
    is_landing = (z <= GROUND_Z)

    obs = _perception.get_obstacles() if _perception else []
    print(f"\n[set_position] target=({x:.2f},{y:.2f},{z:.2f})  "
          f"mode={'LAND' if is_landing else 'GOTO'}  "
          f"obstacles={len(obs)}")

    traj, T_travel = _build_goto_traj(curr_pos, target, max_vel)
    T_total = T_travel + (0.5 if is_landing else T_hold)

    _run_loop(traj, T_total, log_tag='land' if is_landing else 'goto')

    if is_landing:
        _rotorcraft.stop()
        print("[set_position] Landed — motors stopped")
    else:
        print(f"[set_position] Holding at ({x:.2f},{y:.2f},{z:.2f})")


def stop():
    if _rotorcraft is not None:
        _rotorcraft.stop()
        print("[stop] Motors stopped")

    
def hover(x: float, y: float, z: float,
          T_hover: float = 10.0,
          log_tag: str = 'hover',
          spinup_wait: float = 3.0) -> None:
    """
    Standart test fonksiyonu — motordan loga kadar tam döngü.

    1. Motorları çalıştır ve spin-up bekle  → start()
    2. (x,y,z)'ye git                       → set_position()
    3. z'de T_hover saniye bekle
    4. Dikey iniş, motorları durdur         → landing()

        >>> setup()
        >>> hover(0, 0, 4, T_hover=15)
    """
    start()

    # Motor spin-up ve titreşim sönümlemesi için bekle.
    # Bu sürede IMU/MoCap okumaları da stabilize olur.
    print(f"[hover] Spin-up bekleniyor ({spinup_wait:.0f}s)...")
    for i in range(int(spinup_wait), 0, -1):
        print(f"[hover]   {i}s", end='\r')
        time.sleep(1.0)
    print()

    # Kalkıştan önce son durum kontrolü
    x0 = _current_state()
    print(f"[hover] Başlangıç: z={x0[2]:.3f}m  "
          f"roll={np.degrees(np.arctan2(2*(x0[6]*x0[7]+x0[8]*x0[9]), 1-2*(x0[7]**2+x0[8]**2))):.1f}°  "
          f"p={x0[10]:.3f} rad/s")

    set_position(x, y, z, T_hold=T_hover, log_tag=log_tag)
    landing()


# ── Perception yönetim API'si ────────────────────────────────────────────────

def add_obstacle(x: float, y: float, z: float, radius: float):
    """Çalışma zamanında statik engel ekle (level=3 perception için)."""
    assert _perception is not None, "Call setup() first"
    _perception.add_static(x, y, z, radius)


def set_obstacle_level(level: int, **kwargs):
    """
    Perception seviyesini çalışma zamanında değiştir.
    Örnek: set_obstacle_level(1, obstacle_models=[('cyl',0.3)])
    """
    global _perception
    if _perception:
        _perception.stop()
    _perception = PerceptionManager(level=level, **kwargs)
    _perception.start()
    print(f"[perception] Seviye {level}'e geçildi")


if __name__ == '__main__':
    pass
