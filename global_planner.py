"""
Global Trajectory Planner — Quadrotor MPC Control
==================================================
Waypoint-based quintic polynomial (minimum-jerk) trajectory with yaw interpolation.

Implements the same interface as reference_trajectory.py trajectory classes:
  state_at(t)           -> np.array(13,)
  get_horizon(t, N, Ts) -> np.array(N+1, 13)
  initial_state()       -> np.array(13,)
  total_duration()      -> float

State convention (matches mpc_solver.py):
  x = [px, py, pz,  vx, vy, vz,  qw, qx, qy, qz,  p, q, r]

Quintic polynomial (5th order) guarantees continuity up to acceleration at
waypoint junctions — prevents the jerk discontinuities of cubic polynomials
that cause oscillations in quadrotor dynamics.

Ref: Aerial Robotics — Trajectory Generation, PoliMi, Sec. 3

Usage
-----
    from global_planner import WaypointTrajectory

    waypoints = [
        {'pos': np.array([0, 0, 1.5]), 'yaw': 0.0},
        {'pos': np.array([3, 0, 1.5]), 'yaw': 0.0},
        {'pos': np.array([3, 3, 2.0]), 'yaw': np.pi/2},
        {'pos': np.array([0, 3, 1.5]), 'yaw': np.pi},
    ]
    traj = WaypointTrajectory(waypoints, seg_times=[3.0, 3.0, 3.0])

    # Interactive use with quadrotor_mpc_client.py _execute():
    _execute(traj, 'navigate')
"""

import numpy as np

G = 9.81


# ── Rotation helpers ───────────────────────────────────────────────────────────

def _rot_to_quat(R):
    """Rotation matrix → quaternion [qw, qx, qy, qz]  (Shepperd method)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        qw = (R[2,1] - R[1,2]) / s
        qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s
        qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        qw = (R[0,2] - R[2,0]) / s
        qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s
        qz = (R[1,2] + R[2,1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / s
        qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    return q / (np.linalg.norm(q) + 1e-12)


def _flatness_attitude(acc_ref, yaw):
    """
    Differential flatness map: desired acceleration + yaw → reference quaternion.

    For an underactuated quadrotor the body z-axis must align with the
    net force direction  F = m*(a_des + g*e3).  Given the desired yaw ψ the
    full attitude is determined uniquely.

    acc_ref : np.array(3,)  desired acceleration in world frame [m/s²]
    yaw     : float         desired yaw [rad]
    Returns : np.array(4,)  quaternion [qw, qx, qy, qz]
    """
    thrust = acc_ref + np.array([0.0, 0.0, G])   # required thrust direction
    thrust_norm = np.linalg.norm(thrust)

    if thrust_norm < 1e-6:
        return _yaw_to_quat(yaw)     # near-zero thrust: fall back to yaw-only

    z_b = thrust / thrust_norm       # body z = thrust direction

    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_b = np.cross(z_b, x_c)
    y_b_norm = np.linalg.norm(y_b)

    if y_b_norm < 1e-6:              # degenerate: z_b ∥ x_c
        return _yaw_to_quat(yaw)

    y_b /= y_b_norm
    x_b  = np.cross(y_b, z_b)
    R    = np.column_stack([x_b, y_b, z_b])
    return _rot_to_quat(R)


# ── Quintic polynomial core ────────────────────────────────────────────────────

def _quintic_coeffs(q0, v0, a0, qf, vf, af, T):
    """
    Closed-form quintic coefficients for one axis.

    q(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5

    Boundary conditions:
        q(0)=q0,  dq(0)=v0,  ddq(0)=a0
        q(T)=qf,  dq(T)=vf,  ddq(T)=af
    """
    dq = qf - (q0 + v0*T + 0.5*a0*T**2)
    dv = vf - (v0 + a0*T)
    da = af - a0
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    c3 = 10*dq/T3 - 4*dv/T2 + 0.5*da/T
    c4 = -15*dq/T4 + 7*dv/T3 - da/T2
    c5 = 6*dq/T5 - 3*dv/T4 + 0.5*da/T3
    return np.array([q0, v0, 0.5*a0, c3, c4, c5])


def _eval_quintic(c, t):
    """Evaluate polynomial at t → (pos, vel, acc)."""
    pos  =    c[0] +    c[1]*t +    c[2]*t**2 +    c[3]*t**3 +    c[4]*t**4 +    c[5]*t**5
    vel  =    c[1] +  2*c[2]*t +  3*c[3]*t**2 +  4*c[4]*t**3 +  5*c[5]*t**4
    acc  =  2*c[2] +  6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3
    return pos, vel, acc


class _QuinticSegment3D:
    """One 3D quintic polynomial segment [0, T]."""

    def __init__(self, p0, v0, a0, p1, v1, a1, T):
        self.T = float(T)
        self.C = np.array([
            _quintic_coeffs(p0[i], v0[i], a0[i], p1[i], v1[i], a1[i], T)
            for i in range(3)
        ])  # shape (3, 6)

    def eval(self, t):
        """Returns pos (3,), vel (3,), acc (3,) at local time t."""
        t = np.clip(float(t), 0.0, self.T)
        data = [_eval_quintic(self.C[i], t) for i in range(3)]
        return (np.array([d[0] for d in data]),
                np.array([d[1] for d in data]),
                np.array([d[2] for d in data]))


# ── Attitude helpers ───────────────────────────────────────────────────────────

def _yaw_to_quat(yaw):
    """Pure yaw quaternion [qw, qx, qy, qz] from yaw angle [rad]."""
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── Main trajectory class ──────────────────────────────────────────────────────

class WaypointTrajectory:
    """
    Multi-segment quintic trajectory through a list of waypoints.

    Parameters
    ----------
    waypoints : list of dict, each with:
        'pos' : np.array(3,)   position [m]                (required)
        'yaw' : float          yaw angle [rad]              (default 0)
        'vel' : np.array(3,)   velocity at waypoint [m/s]   (default zeros)
        'acc' : np.array(3,)   acceleration [m/s²]          (default zeros)

        Interior waypoints (not first/last) with zero vel/acc produce C2-continuous
        trajectories. Set non-zero vel/acc for directional waypoints.

    seg_times : list of float, length = len(waypoints) - 1
                Segment durations [s]. If None, auto-computed from distance/max_vel.
    max_vel   : float  — used for auto seg_times [m/s]
    min_seg_T : float  — minimum segment duration for auto seg_times [s]
    """

    def __init__(self, waypoints, seg_times=None,
                 max_vel=2.0, min_seg_T=1.0):
        assert len(waypoints) >= 2, "need >= 2 waypoints"

        if seg_times is None:
            seg_times = []
            for i in range(len(waypoints) - 1):
                d = np.linalg.norm(
                    np.asarray(waypoints[i+1]['pos'], float) -
                    np.asarray(waypoints[i]['pos'],   float)
                )
                seg_times.append(max(d / max_vel, min_seg_T))

        assert len(seg_times) == len(waypoints) - 1, \
            "len(seg_times) must equal len(waypoints)-1"

        self._waypoints = waypoints
        self._seg_times = list(seg_times)
        self._segments  = []
        self._t_starts  = []

        t = 0.0
        for i, T in enumerate(seg_times):
            wp0, wp1 = waypoints[i], waypoints[i+1]
            seg = _QuinticSegment3D(
                p0=np.asarray(wp0['pos'],              float),
                v0=np.asarray(wp0.get('vel', [0,0,0]), float),
                a0=np.asarray(wp0.get('acc', [0,0,0]), float),
                p1=np.asarray(wp1['pos'],              float),
                v1=np.asarray(wp1.get('vel', [0,0,0]), float),
                a1=np.asarray(wp1.get('acc', [0,0,0]), float),
                T=T
            )
            self._segments.append(seg)
            self._t_starts.append(t)
            t += T

        self._T_total = t
        self._yaws    = [float(wp.get('yaw', 0.0)) for wp in waypoints]

        print(f"[WaypointTraj] {len(waypoints)} waypoints  "
              f"{len(seg_times)} segments  total={self._T_total:.1f}s")
        for i, T in enumerate(seg_times):
            p0 = np.asarray(waypoints[i]['pos']).round(2)
            p1 = np.asarray(waypoints[i+1]['pos']).round(2)
            print(f"  seg {i}: {p0} -> {p1}  T={T:.2f}s"
                  f"  yaw {np.degrees(self._yaws[i]):.1f}° -> "
                  f"{np.degrees(self._yaws[i+1]):.1f}°")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _locate(self, t_global):
        """Return (seg_idx, t_local) for given global time."""
        t_global = np.clip(float(t_global), 0.0, self._T_total)
        for i, ts in enumerate(self._t_starts):
            if t_global <= ts + self._segments[i].T + 1e-9:
                return i, max(0.0, t_global - ts)
        last = len(self._segments) - 1
        return last, self._segments[last].T

    def _yaw_at(self, t_global):
        """Linear yaw interpolation between waypoints (shortest path)."""
        t_global = np.clip(float(t_global), 0.0, self._T_total)
        for i, ts in enumerate(self._t_starts):
            T = self._segments[i].T
            if t_global <= ts + T + 1e-9:
                tau = np.clip((t_global - ts) / T, 0.0, 1.0)
                dy  = _wrap_angle(self._yaws[i+1] - self._yaws[i])
                return self._yaws[i] + tau * dy
        return self._yaws[-1]

    def _yaw_rate_at(self, t_global):
        """Constant yaw rate within each segment (from linear interpolation)."""
        t_global = np.clip(float(t_global), 0.0, self._T_total)
        for i, ts in enumerate(self._t_starts):
            T = self._segments[i].T
            if t_global <= ts + T + 1e-9:
                dy = _wrap_angle(self._yaws[i+1] - self._yaws[i])
                return dy / T
        return 0.0

    # ── Public interface ───────────────────────────────────────────────────────

    def state_at(self, t):
        """
        Reference state at global time t.

        Attitude is computed via differential flatness: the body z-axis aligns
        with the required thrust direction  F = m*(a_des + g*e3),  and the
        desired yaw determines the remaining degree of freedom.  This produces
        physically consistent pitch/roll references and prevents IPOPT from
        converging to the f=0 local minimum that arises when the attitude
        reference conflicts with the required tilt for acceleration.

        Returns
        -------
        x : np.array(13,) — [px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r]
        """
        seg_idx, t_local = self._locate(t)
        pos, vel, acc    = self._segments[seg_idx].eval(t_local)
        yaw              = self._yaw_at(t)

        # Reference attitude from differential flatness
        q = _flatness_attitude(acc, yaw)

        # Body angular velocity via finite-difference of reference quaternion
        _dt = 1e-2  #### 5e-3 causes some numerical issues with IPOPT convergence — maybe too small for accurate finite-diff?
        t2  = min(float(t) + _dt, self._T_total)
        si2, tl2 = self._locate(t2)
        _, _, acc2 = self._segments[si2].eval(tl2)
        q2  = _flatness_attitude(acc2, self._yaw_at(t2))

        q_dot = (q2 - q) / _dt
        # omega_body = 2 * q̄ ⊗ q_dot  (vector part)
        qw, qx, qy, qz     = q
        qdw, qdx, qdy, qdz = q_dot
        omega = 2.0 * np.array([
            qw*qdx - qx*qdw - qy*qdz + qz*qdy,
            qw*qdy + qx*qdz - qy*qdw - qz*qdx,
            qw*qdz - qx*qdy + qy*qdx - qz*qdw,
        ])

        return np.concatenate([pos, vel, q, omega])

    def get_horizon(self, t_now, N, Ts):
        """
        Reference horizon for NMPC.

        Returns
        -------
        xref : np.array(N+1, 13)
        """
        return np.array([self.state_at(t_now + k * Ts) for k in range(N + 1)])

    def initial_state(self):
        """State at t=0."""
        return self.state_at(0.0)

    def total_duration(self):
        """Total trajectory duration [s]."""
        return self._T_total

    def is_complete(self, t_now, tol=0.1):
        """True when t_now is within tol seconds of trajectory end."""
        return float(t_now) >= self._T_total - tol
