"""
quadrotor_mpc_client.py  (perception entegrasyonlu versiyon)
============================================================
Değişiklikler:
  - QuadrotorMPC → LocalPlannerMPC
  - perception.PerceptionManager entegre edildi
  - _run_loop: obstacles parametresi
  - setup(): perception başlatılıyor
  - Yeni public API: set_obstacle_level(), add_obstacle()
  - Paylaşımlı log session: set_position() açar, landing() kapatır
  - landing(): dikey iniş, algılanan engel üstüne otomatik konum
"""

import time
import os
import numpy as np

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
MPC_TS = 0.05

TAU_Y_FF = 0.20

_MPC_KWARGS = dict(
    Q_pos=5.0,     Q_vel=2.0,
    Q_att=6.0,
    Q_omega=1.5,   Q_omega_r=3.0,
    P_scale=5.0,
    R_f=0.02,      R_tau=0.10,       R_tau_z=0.15,
    tau_max=0.80,  tau_z_max=0.12,
    f_min=0.05*MASS*G,
    f_max_scale=2.5,
    alpha_land=2.0, W_land=500.0,
)

_LOCAL_MPC_KWARGS = dict(
    n_obs_max  = 5,
    R_drone    = 0.30,
    W_obs      = 10000.0,
)

# ── Perception konfigürasyonu ────────────────────────────────────────────────
OBSTACLE_MODELS = []

# ── Uçuş parametreleri ───────────────────────────────────────────────────────
GROUND_Z        = 0.10
MAX_VEL         = 1.5
MIN_SAFE_Z      = 1.5
DESCENT_VEL     = 0.40   # landing() iniş hızı [m/s]
LAND_XY_RADIUS  = 0.50   # engel arama yarıçapı [m]
LAND_MARGIN     = 0.05   # engel üstü güvenlik boşluğu [m]

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
_perception = None

# ── Paylaşımlı log session ───────────────────────────────────────────────────
# set_position() açar, landing() kapatır. İkisi arasında tek npz dosyası.
_session: dict = {}     # boş → aktif session yok
_session_t0: float = 0.0


def _session_open(tag: str) -> None:
    """Yeni bir log session başlat. Önceki bitmeden çağrılırsa üzerine yazar."""
    global _session, _session_t0
    _session = dict(tag=tag, t=[], x=[], u=[], xref=[], mpc_ms=[], n_obs=[])
    _session_t0 = time.time()
    print(f"[log] Session açıldı — tag='{tag}'")


def _session_close() -> str | None:
    """Mevcut session'ı diske yaz, yolu döndür. Session yoksa None."""
    global _session
    if not _session or not _session['t']:
        print("[log] Kaydedilecek session yok.")
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
             n_obs    = np.array(s['n_obs'])     if s['n_obs']  else np.empty(0))

    total_s  = s['t'][-1] - s['t'][0] if len(s['t']) > 1 else 0
    avg_ms   = float(np.mean(s['mpc_ms']))  if s['mpc_ms']  else 0.0
    avg_obs  = float(np.mean(s['n_obs']))   if s['n_obs']   else 0.0

    tau_max  = _MPC_KWARGS['tau_max']
    sat      = 0.0
    if s['u']:
        u_arr = np.array(s['u'])
        sat   = float(np.mean(np.any(np.abs(u_arr[:, 1:3]) >= tau_max * 0.99, axis=1))) * 100

    print(f"[log] Kaydedildi → {fname}")
    print(f"      süre={total_s:.1f}s  MPC={avg_ms:.1f}ms  "
          f"avg_obs={avg_obs:.1f}  τ_sat={sat:.1f}%")

    _session = {}
    return fname


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
    tau_y_comp = tau_y + TAU_Y_FF 
    M = np.array([
        [ KF,            KF,           KF,           KF          ],
        [ 0,             ARM_LEN*KF,   0,           -ARM_LEN*KF  ],
        [-ARM_LEN*KF,    0,            ARM_LEN*KF,   0            ],
        [ KM,           -KM,           KM,          -KM           ],
    ])
    omega_sq = np.linalg.pinv(M) @ np.array([f, tau_x, tau_y_comp, tau_z])
    omega_sq = np.clip(omega_sq, 0.0, None)
    omega    = np.sqrt(omega_sq)
    omega    = np.clip(omega, 0.0, 1200.0)
    _rotorcraft.set_velocity({'desired': [
        float(omega[0]), float(omega[1]),
        float(omega[2]), float(omega[3]),
        0.0, 0.0, 0.0, 0.0,
    ]})


# ── Trayektori builder ───────────────────────────────────────────────────────
def _build_goto_traj(curr_pos, target_pos, max_vel=MAX_VEL):
    """set_position() için trayektori. landing() bunu kullanmaz."""
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
    """
    landing() için dikey iniş trayektorisi.
    x,y sabit, z → land_z.
    """
    cx, cy, cz = curr_pos
    descent    = max(cz - land_z, 0.0)
    T_descend  = max(descent / descent_vel, 2.0)   # min 2s

    waypoints = [
        {'pos': np.array([cx, cy, cz]),    'vel': [0, 0, 0]},
        {'pos': np.array([cx, cy, land_z]),'vel': [0, 0, 0]},
    ]
    traj = WaypointTrajectory(waypoints, seg_times=[T_descend])
    return traj, traj.total_duration()


# ── Kontrol döngüsü ──────────────────────────────────────────────────────────
def _run_loop(traj: WaypointTrajectory,
              T_total: float) -> None:
    """
    MPC kontrol döngüsü.

    Trayektori bitince (t > traj.total_duration()) referans son noktada
    dondurulur — drone hedefte asılı kalır.  T_total süresi dolunca döner.

    Log verisi module-level _session'a yazılır; kaydetmez.
    """
    t_start    = time.time()
    t_next_mpc = t_start

    T_traj = traj.total_duration()

    while True:
        t_now = time.time() - t_start
        if t_now > T_total:
            break

        # Trayektori bittiyse son noktada dondur
        t_ref = min(t_now, T_traj)

        x_now     = _current_state()
        x_ref_now = traj.state_at(t_ref)

        # Global zaman damgası (session başından)
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
            u_opt, info = _mpc.solve(x_now, xref_h, obstacles=obstacles)

            if _session:
                _session['u'].append(u_opt.copy())
                _session['mpc_ms'].append(info['solve_time_ms'])
                _session['n_obs'].append(len(obstacles))

            if info.get('slack_max', 0) > 1e-2:
                print(f"[run] t={t_now:.1f}s  "
                      f"obs_slack={info['slack_max']:.3f}  "
                      f"n_obs={len(obstacles)}")

            wrench_to_rotorcraft(*u_opt)
            t_next_mpc += MPC_TS

        time.sleep(0.002)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def setup(perception_level: int = 1):
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
    Drone'u (x,y,z)'ye götür, T_hold saniye orada tut, döner.

    Log session bu çağrıyla açılır.  Sonraki landing() çağrısı kapatır.
    T_hold içinde landing() çağrılırsa bu fonksiyon bloke etmeye devam eder —
    tuning döngüsü için tipik akış:

        set_position(0, 0, 4, T_hold=60)   # 66s bloke
        landing()                          # log kaydedilir
    """
    assert _armed, "Call start() first"

    target   = np.array([float(x), float(y), float(z)])
    curr_pos = _current_state()[:3]

    obs = _perception.get_obstacles() if _perception else []
    print(f"\n[set_position] target=({x:.2f},{y:.2f},{z:.2f})  "
          f"obstacles={len(obs)}")

    traj, T_travel = _build_goto_traj(curr_pos, target, max_vel)
    T_total        = T_travel + T_hold

    # Yeni session aç (öncekini siler)
    _session_open(log_tag)
    _mpc.reset(_current_state())

    _run_loop(traj, T_total)

    print(f"[set_position] T_hold bitti — drone ({x:.2f},{y:.2f},{z:.2f})'de.")
    print("  landing() ile indir veya set_position() ile yeni hedef ver.")


def landing(descent_vel: float = DESCENT_VEL,
            xy_radius:   float = LAND_XY_RADIUS,
            margin:      float = LAND_MARGIN) -> None:
    """
    Drone'u mevcut x,y pozisyonunu koruyarak indirir.

    Perception'dan gelen engeller içinde drone'un tam altında
    (yatay uzaklık < xy_radius) olanların en yüksek üst yüzeyi
    hedef iniş z'si olur.  Engel yoksa zemine (z=0) iner.

    Parametre
    ---------
    descent_vel : iniş hızı [m/s]
    xy_radius   : "altımda engel var mı?" arama yarıçapı [m]
    margin      : engel üstü boşluk [m]

    Bu fonksiyon döndüğünde motorlar durur ve log kaydedilir.
    """
    assert _armed, "Call start() first"

    x_now    = _current_state()
    curr_pos = x_now[:3]
    curr_xy  = curr_pos[:2]

    # ── Altındaki engelleri bul ──────────────────────────────────────────────
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
        print(f"\n[landing] {n_below} engel altımda — "
              f"iniş z = {land_z:.3f} m  (engel üstü)")
    else:
        print(f"\n[landing] Engel yok — zemine iniş  (z = 0)")

    # ── İniş trayektorisi ────────────────────────────────────────────────────
    traj, T_descend = _build_land_traj(curr_pos, land_z, descent_vel)
    T_total         = T_descend + 1.0   # inmeden sonra 1s kısa tutma

    print(f"[landing] mevcut z={curr_pos[2]:.2f}m  "
          f"hedef z={land_z:.2f}m  süre≈{T_descend:.1f}s")

    # Session açık değilse otomatik aç (bağımsız kullanım)
    if not _session:
        _session_open('landing_only')

    _run_loop(traj, T_total)

    # ── Motorları durdur ve logu kaydet ──────────────────────────────────────
    _rotorcraft.stop()
    print(f"[landing] Motorlar durduruldu — "
          f"son z = {_current_state()[2]:.3f} m")

    _session_close()


def stop():
    """Acil durdurma — logu kaydetmez."""
    if _rotorcraft is not None:
        _rotorcraft.stop()
        print("[stop] Motorlar durduruldu (log kaydedilmedi)")

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
    print(f"[hover] waiting for spin-up ({spinup_wait:.0f}s)...")
    for i in range(int(spinup_wait), 0, -1):
        print(f"[hover]   {i}s", end='\r')
        time.sleep(1.0)
    print()

    # Kalkıştan önce son durum kontrolü
    x0 = _current_state()
    print(f"[hover] Start: z={x0[2]:.3f}m  "
          f"roll={np.degrees(np.arctan2(2*(x0[6]*x0[7]+x0[8]*x0[9]), 1-2*(x0[7]**2+x0[8]**2))):.1f}°  "
          f"p={x0[10]:.3f} rad/s")

    set_position(x, y, z, T_hold=T_hover, log_tag=log_tag)
    landing()


# ── Perception yönetim API'si ────────────────────────────────────────────────

def add_obstacle(x: float, y: float, z: float, radius: float):
    """Çalışma zamanında statik engel ekle (level=3 perception)."""
    assert _perception is not None, "Call setup() first"
    _perception.add_static(x, y, z, radius)


def set_obstacle_level(level: int, **kwargs):
    global _perception
    if _perception:
        _perception.stop()
    _perception = PerceptionManager(level=level, **kwargs)
    _perception.start()
    print(f"[perception] Seviye {level}'e geçildi")


if __name__ == '__main__':
    pass
