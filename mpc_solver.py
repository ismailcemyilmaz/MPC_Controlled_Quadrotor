"""
Quadrotor NMPC — acados solver
================================
State  x ∈ R^13 : [px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r]
Input  u ∈ R^4  : [f_total, τx, τy, τz]

Cost (NONLINEAR_LS, h = [x; u], 17 outputs):
  stage:    ||[x; u] - [x_ref; u_hover]||_W^2
  terminal: ||x_N - x_ref_N||_WN^2

Landing cone soft constraint (every stage):
  h_land(x) = vz + alpha_land * z  ≥  0
  → near ground vz is bounded: |vz| ≤ alpha_land * z
  → enforced softly with quadratic penalty W_land
"""

import os
import numpy as np
import time

_CMAKE = '/usr/local/MATLAB/R2025b/bin/glnxa64/cmake/bin'
if os.path.isdir(_CMAKE) and _CMAKE not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _CMAKE + ':' + os.environ.get('PATH', '')

_ACADOS_CANDIDATES = ['/opt/acados', os.path.expanduser('~/acados')]
ACADOS_SOURCE_DIR = os.environ.get(
    'ACADOS_SOURCE_DIR',
    next((p for p in _ACADOS_CANDIDATES if os.path.isfile(os.path.join(p, 'lib', 'libacados.so'))), _ACADOS_CANDIDATES[-1])
)
os.environ.setdefault('ACADOS_SOURCE_DIR', ACADOS_SOURCE_DIR)

_ACADOS_LIB = os.path.join(ACADOS_SOURCE_DIR, 'lib')
if _ACADOS_LIB not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = _ACADOS_LIB + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import ctypes
for _lib in ['libblasfeo.so', 'libhpipm.so', 'libqpOASES_e.so', 'libacados.so']:
    _path = os.path.join(_ACADOS_LIB, _lib)
    if os.path.isfile(_path):
        ctypes.CDLL(_path, mode=ctypes.RTLD_GLOBAL)

try:
    import casadi as cs
    from acados_template import AcadosOcp, AcadosOcpSolver
    from quadrotor_model import export_quadrotor_model
    ACADOS_AVAILABLE = True
except ImportError as _e:
    ACADOS_AVAILABLE = False
    print(f"[mpc_solver] acados import failed: {_e}")

G = 9.81

_CODEGEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'acados_generated')
os.makedirs(_CODEGEN_DIR, exist_ok=True)


def _json_path(tag: str) -> str:
    return os.path.join(_CODEGEN_DIR, f'{tag}.json')


class QuadrotorMPC:
    """
    Nonlinear MPC for a quadrotor using acados SQP-RTI.

    Parameters
    ----------
    N            : prediction horizon (number of steps)
    Ts           : sampling time [s]
    mass         : vehicle mass [kg]
    I_diag       : (Ixx, Iyy, Izz) [kg·m²]
    Q_pos/vel/att/omega : tracking weights
    P_scale      : terminal cost multiplier
    R_f          : thrust cost
    R_tau        : torque cost (τx, τy)
    R_tau_z      : yaw torque cost — defaults to R_tau if None
    f_min        : minimum thrust [N]
    f_max_scale  : f_max = f_max_scale * mass * g
    tau_max      : roll/pitch torque limit [N·m]
    tau_z_max    : yaw torque limit [N·m] — defaults to tau_max if None
                   Physical limit at hover: f_hover * km/cf ≈ 0.19 Nm
    alpha_land   : landing cone slope [1/s]: |vz| ≤ alpha_land * z near ground
                   alpha=2.0 → at z=0.3m max descent = 0.6 m/s
                   Set to 0.0 to disable the constraint entirely.
    W_land       : quadratic penalty for landing cone violation [cost/m²]
    max_iter     : (unused — API compat)
    rk4_steps    : ERK integration steps per MPC interval
    """

    def __init__(self, N=20, Ts=0.05, mass=1.28,
             I_diag=(22.916e-3, 22.916e-3, 22.132e-3),
             Q_pos=5.0, Q_vel=1.0, Q_att=2.0, Q_omega=1.0, Q_omega_r=None,
             P_scale=3.0,
                 R_f=0.001, R_tau=0.02, R_tau_z=None,
                 f_min=0.0, f_max_scale=2.5,
                 tau_max=0.50, tau_z_max=None,
                 alpha_land=2.0, W_land=500.0,
                 max_iter=50,
                 rk4_steps=1):
        assert ACADOS_AVAILABLE, "acados / acados_template not importable"

        self.N        = N
        self.Ts       = Ts
        self.m        = mass
        self.nx       = 13
        self.nu       = 4
        self.ny       = 17
        self.nyN      = 13
        self.f_hover  = mass * G
        self.alpha_land = alpha_land

        _tau_z_max = tau_z_max if tau_z_max is not None else tau_max
        _R_tau_z   = R_tau_z   if R_tau_z   is not None else R_tau

        _Q_omega_r = Q_omega_r if Q_omega_r is not None else Q_omega
        self._W  = self._make_W(Q_pos, Q_vel, Q_att, Q_omega, _Q_omega_r, R_f, R_tau, _R_tau_z)
        self._WN = self._make_WN(Q_pos, Q_vel, Q_att, P_scale)

        # Tag includes alpha_land so a change triggers recompile
        land_tag = f'_land{int(alpha_land*10)}' if alpha_land > 0 else '_noland'
        tag = f'quad_N{N}_Ts{int(Ts*1000)}ms_m{int(mass*1000)}g{land_tag}'
        json_file = _json_path(tag)

        ocp = self._build_ocp(N, Ts, mass, I_diag, f_min, f_max_scale,
                               tau_max, _tau_z_max, alpha_land, W_land, rk4_steps)

        self.solver = AcadosOcpSolver(ocp, json_file=json_file,
                                       build=True, generate=True,
                                       verbose=False)
        self._apply_weights()

        land_str = f'alpha={alpha_land}, W={W_land}' if alpha_land > 0 else 'disabled'
        print(f"[MPC] acados solver ready — N={N}, Ts={Ts}s, "
              f"ERK steps={rk4_steps}, landing_cone=[{land_str}]")

    # ── OCP construction ──────────────────────────────────────────────────────

    @staticmethod
    def _build_ocp(N, Ts, mass, I_diag, f_min, f_max_scale,
                   tau_max, tau_z_max, alpha_land, W_land, rk4_steps):
        ocp   = AcadosOcp()
        model = export_quadrotor_model(mass=mass, I_diag=I_diag)

        x = model.x   # [px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r]
        # z  = x[2],  vz = x[5]

        # ── Cost: NONLINEAR_LS  h(x,u) = [x; u] ─────────────────────────────
        model.cost_y_expr   = cs.vertcat(x, model.u)
        model.cost_y_expr_e = x
        ocp.model = model

        ocp.cost.cost_type   = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.W   = np.eye(17)
        ocp.cost.W_e = np.eye(13)

        yref   = np.zeros(17); yref[6]  = 1.0; yref[13] = mass * G
        yref_e = np.zeros(13); yref_e[6] = 1.0
        ocp.cost.yref   = yref
        ocp.cost.yref_e = yref_e

        # ── Landing cone soft constraint ─────────────────────────────────────
        # h_land(x) = vz + alpha*z  ≥  0
        # Meaning: vz ≥ -alpha*z  (descent speed bounded by altitude)
        # Active only when drone descends near ground; always satisfied at height.
        if alpha_land > 0:
            h_land = x[5] + alpha_land * x[2]   # vz + alpha*z
            model.con_h_expr   = cs.vertcat(h_land)
            model.con_h_expr_e = cs.vertcat(h_land)   # terminal stage too

            ocp.constraints.lh   = np.array([0.0])
            ocp.constraints.uh   = np.array([1e12])
            ocp.constraints.lh_e = np.array([0.0])
            ocp.constraints.uh_e = np.array([1e12])

            # Soft (lower-bound slack only — we only penalise violations vz < -alpha*z)
            ocp.constraints.Jsh   = np.eye(1)
            ocp.constraints.Jsh_e = np.eye(1)

            # Quadratic penalty on slack (linear term = 0)
            ocp.cost.zl   = np.zeros(1); ocp.cost.zu   = np.zeros(1)
            ocp.cost.Zl   = np.array([W_land]); ocp.cost.Zu = np.zeros(1)
            ocp.cost.zl_e = np.zeros(1); ocp.cost.zu_e = np.zeros(1)
            ocp.cost.Zl_e = np.array([W_land]); ocp.cost.Zu_e = np.zeros(1)

        # ── Input constraints  [f, τx, τy, τz] ──────────────────────────────
        f_max = f_max_scale * mass * G
        ocp.constraints.lbu   = np.array([f_min,    -tau_max, -tau_max, -tau_z_max])
        ocp.constraints.ubu   = np.array([f_max,     tau_max,  tau_max,  tau_z_max])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        hover_x0 = np.array([0,0,0, 0,0,0, 1,0,0,0, 0,0,0], dtype=float)
        ocp.constraints.x0 = hover_x0

        # ── Solver options ────────────────────────────────────────────────────
        ocp.solver_options.N_horizon              = N
        ocp.solver_options.qp_solver              = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.nlp_solver_type        = 'SQP_RTI'
        ocp.solver_options.integrator_type        = 'ERK'
        ocp.solver_options.sim_method_num_stages  = 4
        ocp.solver_options.sim_method_num_steps   = max(1, rk4_steps)
        ocp.solver_options.hessian_approx         = 'GAUSS_NEWTON'
        ocp.solver_options.tf                     = N * Ts
        ocp.solver_options.print_level            = 0

        return ocp

    # ── Weight helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _make_W(Q_pos, Q_vel, Q_att, Q_omega, Q_omega_r, R_f, R_tau, R_tau_z=None):
        if R_tau_z is None:
            R_tau_z = R_tau
        return np.diag([
            Q_pos,    Q_pos,    Q_pos,
            Q_vel,    Q_vel,    Q_vel,
            Q_att,    Q_att,    Q_att,    Q_att,
            Q_omega,  Q_omega,  Q_omega_r,   # p, q, r — separate weight for yaw rate
            R_f,      R_tau,    R_tau,    R_tau_z,
        ])

    @staticmethod
    def _make_WN(Q_pos, Q_vel, Q_att, P_scale):
        s = P_scale
        return np.diag([
            s*Q_pos, s*Q_pos, s*Q_pos,
            s*Q_vel, s*Q_vel, s*Q_vel,
            s*Q_att, s*Q_att, s*Q_att, s*Q_att,
            0.0, 0.0, 0.0,     # no omega penalty at terminal
        ])

    def _apply_weights(self):
        for k in range(self.N):
            self.solver.cost_set(k, 'W', self._W)
        self.solver.cost_set(self.N, 'W', self._WN)

    # ── Solver reset ─────────────────────────────────────────────────────────

    def reset(self, x0: np.ndarray = None):
        """Clear warm-start state. Call before starting a new experiment."""
        if x0 is None:
            x0 = np.array([0,0,0, 0,0,0, 1,0,0,0, 0,0,0], dtype=float)
        if x0[6] < 0:          # antipodal fix
            x0[6:10] *= -1
        u_hover = np.array([self.f_hover, 0.0, 0.0, 0.0])
        for k in range(self.N + 1):
            self.solver.set(k, 'x', x0)
        for k in range(self.N):
            self.solver.set(k, 'u', u_hover)

    # ── Public interface ──────────────────────────────────────────────────────

    def solve(self, x0: np.ndarray,
              x_ref_horizon: np.ndarray) -> tuple:
        """
        Solve one MPC step.

        Parameters
        ----------
        x0            : current state (13,)
        x_ref_horizon : reference trajectory (N+1, 13)

        Returns
        -------
        u_opt : optimal first control (4,)   [f, τx, τy, τz]
        info  : dict with solve_time_ms, cost, feasible, status, land_slack
        """
        assert x0.shape            == (self.nx,),          "x0 shape"
        assert x_ref_horizon.shape == (self.N+1, self.nx), "xref shape"

        N = self.N

        self.solver.set(0, 'lbx', x0)
        self.solver.set(0, 'ubx', x0)

        q0 = x0[6:10]
        for k in range(N):
            xref = x_ref_horizon[k].copy()
            if np.dot(xref[6:10], q0) < 0:
                xref[6:10] *= -1
            yref = np.concatenate([xref, [self.f_hover, 0.0, 0.0, 0.0]])
            self.solver.set(k, 'yref', yref)

        xrefN = x_ref_horizon[N].copy()
        if np.dot(xrefN[6:10], q0) < 0:
            xrefN[6:10] *= -1
        self.solver.set(N, 'yref', xrefN)

        t_start = time.time()
        status  = self.solver.solve()
        dt_ms   = (time.time() - t_start) * 1000

        u_opt = self.solver.get(0, 'u')
        cost  = float(self.solver.get_cost())

        # Landing cone slack diagnostics
        land_slack = 0.0
        if self.alpha_land > 0:
            try:
                sl = self.solver.get(0, 'sl')
                land_slack = float(sl[0])
            except Exception:
                pass

        feasible = (status == 0)
        if not feasible:
            print(f"[MPC] WARNING — status={status} ({dt_ms:.1f} ms)")
        if land_slack > 0.01:
            vz = x0[5]; z = x0[2]
            print(f"[MPC] Landing cone active: z={z:.2f}m  vz={vz:.2f}  "
                  f"slack={land_slack:.4f}  (limit vz≥{-self.alpha_land*z:.2f})")

        return u_opt, {
            'solve_time_ms': round(dt_ms, 2),
            'cost':          cost,
            'feasible':      feasible,
            'status':        status,
            'land_slack':    land_slack,
        }
