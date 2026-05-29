"""
Local Planner NMPC — obstacle-aware quadrotor controller (acados)
=================================================================
Extends QuadrotorMPC with soft obstacle-avoidance constraints.

State  x ∈ R^13        : [px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r]
Input  u ∈ R^4         : [f_total, τx, τy, τz]
Params p ∈ R^(4·n_obs) : [px_o,py_o,pz_o,R_o] per obstacle slot

Obstacle soft constraint at each stage:
    h_i(x,p) = ||p_drone − p_obs_i||² − (R_obs_i + R_drone)²  ≥ 0
    Soft penalty: W_obs · sl_i²  (acados lower-bound slack)

Unused obstacle slots → dummy obstacle at 1e6 m.

Usage
-----
    solver = LocalPlannerMPC(N=10, Ts=0.05, mass=1.0, n_obs_max=3)

    obstacles = [
        (np.array([2.0, 0.0, 1.5]), 0.5),   # (p_obs, R_obs)
    ]
    u_opt, info = solver.solve(x0, x_ref_horizon, obstacles=obstacles)
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
    print(f"[local_planner] acados import failed: {_e}")

G = 9.81

_DUMMY_P_OBS = np.array([500.0, 0.0, 0.0, 0.01], dtype=float)  # far-away, tiny radius

_CODEGEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'acados_generated')
os.makedirs(_CODEGEN_DIR, exist_ok=True)

def _clean_generated():
    """Remove old generated code to avoid permission/stale issues."""
    import shutil
    for d in [_CODEGEN_DIR,
              os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c_generated_code')]:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)


class LocalPlannerMPC:
    """
    Obstacle-aware NMPC for quadrotor (acados SQP-RTI).

    Drop-in replacement for the CasADi LocalPlannerMPC.
    solve() accepts the same optional `obstacles` argument.

    Parameters
    ----------
    N           : prediction horizon steps
    Ts          : sampling time [s]
    mass        : vehicle mass [kg]
    I_diag      : (Ixx, Iyy, Izz) [kg·m²]
    n_obs_max   : max simultaneous obstacles (fixed at build time)
    R_drone     : drone safety radius [m]
    W_obs       : obstacle soft-constraint penalty (quadratic)
    Q_pos/vel/att/omega : tracking weights
    P_scale     : terminal cost multiplier
    R_f/R_tau   : control cost weights
    f_max_scale : f_max = f_max_scale * mass * g
    tau_max     : torque limit [N·m]
    max_iter    : (unused — API compat)
    rk4_steps   : ERK integration steps per interval
    """

    def __init__(self, N=10, Ts=0.05, mass=1.28,
             I_diag=(22.916e-3, 22.916e-3, 22.132e-3),
             n_obs_max=5, R_drone=0.3, W_obs=10000.0,
             Q_pos=5.0, Q_vel=1.0, Q_att=2.0, Q_omega=1.0, Q_omega_r=None,
             P_scale=3.0,
             R_f=0.001, R_tau=0.02, R_tau_z=None,
             f_min=0.0, f_max_scale=2.5,
             tau_max=0.12, tau_z_max=None,
             alpha_land=0.0, W_land=500.0,
             max_iter=50, rk4_steps=1):
        assert ACADOS_AVAILABLE, "acados not importable"

        self.N          = N
        self.Ts         = Ts
        self.m          = mass
        self.nx         = 13
        self.nu         = 4
        self.ny         = 17
        self.nyN        = 13
        self.n_obs_max  = n_obs_max
        self.R_drone    = R_drone
        self.f_hover    = mass * G
        self.alpha_land = alpha_land

        _tau_z_max = tau_z_max if tau_z_max is not None else tau_max
        _R_tau_z   = R_tau_z   if R_tau_z   is not None else R_tau

        _Q_omega_r = Q_omega_r if Q_omega_r is not None else Q_omega
        self._W  = self._make_W(Q_pos, Q_vel, Q_att, Q_omega, _Q_omega_r, R_f, R_tau, _R_tau_z)
        self._WN = self._make_WN(Q_pos, Q_vel, Q_att, P_scale)

        land_tag = f'_land{int(alpha_land*10)}' if alpha_land > 0 else ''
        tag = (f'local_N{N}_Ts{int(Ts*1000)}ms_m{int(mass*1000)}g'
               f'_obs{n_obs_max}{land_tag}')
        json_file = os.path.join(_CODEGEN_DIR, f'{tag}.json')

        _clean_generated()

        ocp = self._build_ocp(N, Ts, mass, I_diag, n_obs_max, R_drone,
                               W_obs, f_min, f_max_scale, tau_max, _tau_z_max,
                               alpha_land, W_land, rk4_steps)

        self.solver = AcadosOcpSolver(ocp, json_file=json_file,
                                       build=True, generate=True,
                                       verbose=False)

        self._apply_weights()

        f_max = f_max_scale * mass * G
        self._default_lbu = np.array([f_min, -tau_max, -tau_max, -_tau_z_max])
        self._default_ubu = np.array([f_max,  tau_max,  tau_max,  _tau_z_max])

        land_str = f'alpha={alpha_land}, W={W_land}' if alpha_land > 0 else 'disabled'
        print(f"[LocalMPC] acados solver ready — N={N}, Ts={Ts}s, "
              f"n_obs_max={n_obs_max}, R_drone={R_drone}m, W_obs={W_obs:.0f}, "
              f"landing_cone=[{land_str}]")

    # ── OCP construction ──────────────────────────────────────────────────────

    @staticmethod
    def _build_ocp(N, Ts, mass, I_diag, n_obs_max, R_drone,
                   W_obs, f_min, f_max_scale, tau_max, tau_z_max,
                   alpha_land, W_land, rk4_steps):
        ocp   = AcadosOcp()
        model = export_quadrotor_model(mass=mass, I_diag=I_diag,
                                        name='quadrotor_local')

        x = model.x
        u = model.u

        # ── Online parameters: [px,py,pz,R per obstacle] ─────────────────────
        p = cs.SX.sym('p', 4 * n_obs_max)
        model.p = p

        # ── Nonlinear constraints h(x,p) >= 0 ───────────────────────────────
        h_list = []

        # Obstacle avoidance constraints
        for i in range(n_obs_max):
            px_o = p[4*i];  py_o = p[4*i+1];  pz_o = p[4*i+2];  R_o = p[4*i+3]
            d_sq = (x[0]-px_o)**2 + (x[1]-py_o)**2 + (x[2]-pz_o)**2
            h_list.append(d_sq - (R_o + R_drone)**2)

        # Landing cone constraint: vz + alpha*z >= 0
        has_land = alpha_land > 0
        if has_land:
            h_land = x[5] + alpha_land * x[2]
            h_list.append(h_land)

        nsh = len(h_list)
        model.con_h_expr = cs.vertcat(*h_list)

        # Terminal stage also gets landing cone
        if has_land:
            model.con_h_expr_e = cs.vertcat(x[5] + alpha_land * x[2])
            ocp.constraints.lh_e = np.array([0.0])
            ocp.constraints.uh_e = np.array([1e12])
            ocp.constraints.Jsh_e = np.eye(1)
            ocp.cost.zl_e = np.zeros(1)
            ocp.cost.zu_e = np.zeros(1)
            ocp.cost.Zl_e = np.array([W_land])
            ocp.cost.Zu_e = np.zeros(1)

        # ── Tracking cost ─────────────────────────────────────────────────────
        model.cost_y_expr   = cs.vertcat(x, u)
        model.cost_y_expr_e = x

        ocp.model = model

        # Constraint bounds: all h >= 0
        ocp.constraints.lh = np.zeros(nsh)
        ocp.constraints.uh = 1e12 * np.ones(nsh)

        # Soft constraints: all h constraints softened (lower-bound slack)
        ocp.constraints.Jsh = np.eye(nsh)
        ocp.cost.zl  = np.zeros(nsh)
        ocp.cost.zu  = np.zeros(nsh)
        # Obstacle slots get W_obs penalty, landing cone gets W_land penalty
        Zl_vals = np.full(nsh, W_obs)
        if has_land:
            Zl_vals[-1] = W_land
        ocp.cost.Zl  = Zl_vals
        ocp.cost.Zu  = np.zeros(nsh)

        ocp.cost.cost_type   = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        ocp.cost.W   = np.eye(17)
        ocp.cost.W_e = np.eye(13)

        yref   = np.zeros(17); yref[6]  = 1.0; yref[13] = mass * G
        yref_e = np.zeros(13); yref_e[6] = 1.0
        ocp.cost.yref   = yref
        ocp.cost.yref_e = yref_e

        ocp.parameter_values = np.tile(_DUMMY_P_OBS, n_obs_max)

        # Input constraints  [f, τx, τy, τz]
        # τz has a tighter physical limit: |τz_max| = f·km/cf ≈ 0.19 Nm at hover
        f_max = f_max_scale * mass * G
        ocp.constraints.lbu   = np.array([f_min,    -tau_max, -tau_max, -tau_z_max])
        ocp.constraints.ubu   = np.array([f_max,     tau_max,  tau_max,  tau_z_max])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        hover_x0 = np.array([0,0,0, 0,0,0, 1,0,0,0, 0,0,0], dtype=float)
        ocp.constraints.x0 = hover_x0

        # ── Solver options ────────────────────────────────────────────────────
        ocp.solver_options.N_horizon             = N
        ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.nlp_solver_type       = 'SQP_RTI'
        ocp.solver_options.integrator_type       = 'ERK'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps  = max(1, rk4_steps)
        ocp.solver_options.hessian_approx        = 'GAUSS_NEWTON'
        ocp.solver_options.tf                    = N * Ts
        ocp.solver_options.print_level           = 0

        return ocp

    # ── Weight helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _make_W(Q_pos, Q_vel, Q_att, Q_omega, Q_omega_r, R_f, R_tau, R_tau_z=None):
        if R_tau_z is None:
            R_tau_z = R_tau
        return np.diag([
            Q_pos,  Q_pos,  Q_pos,
            Q_vel,  Q_vel,  Q_vel,
            Q_att,  Q_att,  Q_att,  Q_att,
            Q_omega, Q_omega, Q_omega_r,
            R_f,    R_tau,  R_tau,  R_tau_z,
        ])

    @staticmethod
    def _make_WN(Q_pos, Q_vel, Q_att, P_scale):
        s = P_scale
        return np.diag([
            s*Q_pos, s*Q_pos, s*Q_pos,
            s*Q_vel, s*Q_vel, s*Q_vel,
            s*Q_att, s*Q_att, s*Q_att, s*Q_att,
            0.0, 0.0, 0.0,
        ])

    def _apply_weights(self):
        for k in range(self.N):
            self.solver.cost_set(k, 'W', self._W)
        self.solver.cost_set(self.N, 'W', self._WN)

    
    def reset(self, x0: np.ndarray = None):
        """Clear warm-start state. Call before starting a new experiment."""
        if x0 is None:
            x0 = np.array([0,0,0, 0,0,0, 1,0,0,0, 0,0,0], dtype=float)
        if x0[6] < 0:
            x0[6:10] *= -1
        u_hover = np.array([self.f_hover, 0.0, 0.0, 0.0])
        for k in range(self.N + 1):
            self.solver.set(k, 'x', x0)
        for k in range(self.N):
            self.solver.set(k, 'u', u_hover)

    # ── Runtime bound / weight switching ─────────────────────────────────────

    def set_bounds(self, f_min=None, tau_max=None, tau_z_max=None):
        lbu = self._default_lbu.copy()
        ubu = self._default_ubu.copy()
        if f_min is not None:
            lbu[0] = f_min
        if tau_max is not None:
            lbu[1] = -tau_max;  lbu[2] = -tau_max
            ubu[1] =  tau_max;  ubu[2] =  tau_max
        if tau_z_max is not None:
            lbu[3] = -tau_z_max
            ubu[3] =  tau_z_max
        for k in range(self.N):
            self.solver.constraints_set(k, 'lbu', lbu)
            self.solver.constraints_set(k, 'ubu', ubu)

    def reset_bounds(self):
        for k in range(self.N):
            self.solver.constraints_set(k, 'lbu', self._default_lbu)
            self.solver.constraints_set(k, 'ubu', self._default_ubu)

    def set_weights(self, W=None, WN=None):
        if W is not None:
            for k in range(self.N):
                self.solver.cost_set(k, 'W', W)
        if WN is not None:
            self.solver.cost_set(self.N, 'W', WN)

    def reset_weights(self):
        self._apply_weights()

    def warm_start_trajectory(self, traj, t0=0.0):
        """Initialize solver states from a trajectory object for better convergence."""
        for k in range(self.N + 1):
            x_ref = traj.state_at(t0 + k * self.Ts)
            self.solver.set(k, 'x', x_ref)

    # ── Public interface ──────────────────────────────────────────────────────

    def solve(self, x0: np.ndarray,
              x_ref_horizon: np.ndarray,
              obstacles=None,
              sign_correct=True) -> tuple:
        """
        Solve one MPC step.

        Parameters
        ----------
        x0            : current state (13,)
        x_ref_horizon : reference trajectory (N+1, 13)
        obstacles     : list of (p_obs, R_obs) tuples, len <= n_obs_max
                        None or [] → no obstacles active (dummy used)

        Returns
        -------
        u_opt : optimal first control (4,)
        info  : dict — 'solve_time_ms', 'cost', 'feasible', 'status',
                       'n_obs_active', 'slack_max'
        """
        assert x0.shape            == (self.nx,),          "x0 shape"
        assert x_ref_horizon.shape == (self.N+1, self.nx), "xref shape"

        N = self.N

        # ── Pack obstacle parameters ──────────────────────────────────────────
        p_val    = np.tile(_DUMMY_P_OBS, self.n_obs_max).copy()
        n_active = 0

        if obstacles:
            n_active = min(len(obstacles), self.n_obs_max)
            for i in range(n_active):
                p_obs, R_obs = obstacles[i]
                p_val[4*i:4*i+4] = [p_obs[0], p_obs[1], p_obs[2], float(R_obs)]

        # ── Fix initial state & set parameters ───────────────────────────────
        self.solver.set(0, 'lbx', x0)
        self.solver.set(0, 'ubx', x0)

        for k in range(N):
            self.solver.set(k, 'p', p_val)

        # ── Set references ────────────────────────────────────────────────────
        q0 = x0[6:10]
        for k in range(N):
            xref = x_ref_horizon[k].copy()
            if sign_correct and np.dot(xref[6:10], q0) < 0:
                xref[6:10] *= -1
            yref = np.concatenate([xref, [self.f_hover, 0.0, 0.0, 0.0]])
            self.solver.set(k, 'yref', yref)

        xrefN = x_ref_horizon[N].copy()
        if sign_correct and np.dot(xrefN[6:10], q0) < 0:
            xrefN[6:10] *= -1
        self.solver.set(N, 'yref', xrefN)

        # ── Solve ─────────────────────────────────────────────────────────────
        t_start = time.time()
        status  = self.solver.solve()
        dt_ms   = (time.time() - t_start) * 1000

        u_opt = self.solver.get(0, 'u')
        cost  = float(self.solver.get_cost())

        feasible = (status == 0)

        # Slack diagnostics
        slack_max = 0.0
        land_slack = 0.0
        try:
            sl = self.solver.get(0, 'sl')
            if n_active > 0:
                slack_max = float(np.max(sl[:n_active]))
            if self.alpha_land > 0 and len(sl) > self.n_obs_max:
                land_slack = float(sl[self.n_obs_max])
        except Exception:
            pass

        info = {
            'solve_time_ms': round(dt_ms, 2),
            'cost':          cost,
            'feasible':      feasible,
            'status':        status,
            'n_obs_active':  n_active,
            'slack_max':     slack_max,
            'land_slack':    land_slack,
        }

        if not feasible:
            print(f"[LocalMPC] WARNING — status={status} ({dt_ms:.1f} ms)")
        if slack_max > 1e-3:
            print(f"[LocalMPC] Obstacle slack active: max_slack={slack_max:.4f}")
        if land_slack > 0.01:
            vz = x0[5]; z = x0[2]
            print(f"[LocalMPC] Landing cone active: z={z:.2f}m vz={vz:.2f} "
                  f"slack={land_slack:.4f} (limit vz>={-self.alpha_land*z:.2f})")

        return u_opt, info
