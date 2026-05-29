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


# ══════════════════════════════════════════════════════════════════════════════
#  BACKFLIP TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════

def _quintic_s(tau):
    """Quintic interpolant s(tau) for tau in [0,1]. s=0→0, s=1→1, s'=s''=0 at endpoints."""
    return 10*tau**3 - 15*tau**4 + 6*tau**5

def _quintic_ds(tau, T):
    """First derivative ds/dt = ds/dtau / T."""
    return (30*tau**2 - 60*tau**3 + 30*tau**4) / T

def _quintic_dds(tau, T):
    """Second derivative d²s/dt²."""
    return (60*tau - 180*tau**2 + 120*tau**3) / (T*T)


class BackflipTrajectory:
    """
    Full-state reference trajectory for a backflip (360° pitch rotation).

    Timeline:
        [0, T_pre)          : hold at start position (build up MPC warm-start)
        [T_pre, T_pre+T_up) : vertical pop-up to gain upward velocity
        [T_pre+T_up, T_pre+T_up+T_flip) : 360° pitch rotation, ballistic arc
        [T_pre+T_up+T_flip, T_total)     : recovery hover at end position

    The flip phase uses a quintic angle profile for smooth angular jerk.
    Position during flip follows a ballistic arc (free-fall with optional
    partial thrust from the rotating body frame).

    Same interface as WaypointTrajectory: state_at(), get_horizon(), total_duration().
    """

    def __init__(self, start_pos, T_flip=0.8, T_pre=1.0, T_up=0.5, T_post=3.0,
                 pop_up_vel=4.0):
        self._start = np.array(start_pos, dtype=float)
        self._T_up = T_up
        self._T_flip = T_flip
        self._T_total = T_pre + T_up + T_flip + T_post

        self._t1 = T_pre                       # pop-up start
        self._t2 = T_pre + T_up                # flip start
        self._t3 = T_pre + T_up + T_flip       # recovery start

        # Pop-up endpoint: quintic moves z from start to start + dz_up
        self._dz_up = pop_up_vel * T_up * 0.5
        self._vz_up = pop_up_vel

        # Flip endpoint: ballistic from (z_flip_start, vz_up)
        z_fs = self._start[2] + self._dz_up
        self._z_flip_start = z_fs
        self._z_flip_end = z_fs + self._vz_up * T_flip - 0.5 * G * T_flip**2
        self._vz_flip_end = self._vz_up - G * T_flip

        # Recovery target
        self._z_recover = max(self._start[2], self._z_flip_end)

    def total_duration(self):
        return self._T_total

    def state_at(self, t):
        t = np.clip(float(t), 0.0, self._T_total)
        x0, y0, z0 = self._start

        if t < self._t1:
            return np.array([x0, y0, z0, 0,0,0, 1,0,0,0, 0,0,0])

        elif t < self._t2:
            # Pop-up: quintic position profile, upright attitude
            tau = (t - self._t1) / self._T_up
            s = _quintic_s(tau)
            ds = _quintic_ds(tau, self._T_up)
            z = z0 + self._dz_up * s
            vz = self._dz_up * ds
            return np.array([x0, y0, z, 0, 0, vz, 1,0,0,0, 0,0,0])

        elif t < self._t3:
            # Flip: ballistic arc + 360° pitch rotation (quintic angle)
            dt = t - self._t2
            tau = dt / self._T_flip

            z = self._z_flip_start + self._vz_up * dt - 0.5 * G * dt**2
            vz = self._vz_up - G * dt

            theta = 2.0 * np.pi * _quintic_s(tau)
            dtheta = 2.0 * np.pi * _quintic_ds(tau, self._T_flip)

            qw = np.cos(theta / 2.0)
            qy = np.sin(theta / 2.0)

            return np.array([x0, y0, z, 0, 0, vz, qw, 0, qy, 0, 0, dtheta, 0])

        else:
            # Recovery: smooth convergence to hover at z_recover
            dt = t - self._t3
            T_conv = 1.5
            tau = min(dt / T_conv, 1.0)
            s = _quintic_s(tau)

            z = self._z_flip_end * (1.0 - s) + self._z_recover * s
            vz = self._vz_flip_end * (1.0 - s)

            return np.array([x0, y0, z, 0, 0, vz, 1,0,0,0, 0,0,0])

    def get_horizon(self, t_now, N, Ts):
        return np.array([self.state_at(t_now + k * Ts) for k in range(N + 1)])


# ══════════════════════════════════════════════════════════════════════════════
#  APF TRAJECTORY — Artificial Potential Field path planning
# ══════════════════════════════════════════════════════════════════════════════

class APFTrajectory:
    """
    Artificial Potential Field trajectory from start to goal through obstacles.

    Generates a smooth path offline by integrating the APF gradient, then
    fits quintic polynomials for velocity/acceleration continuity.

    Same interface as WaypointTrajectory: state_at(), get_horizon(), total_duration().

    Parameters
    ----------
    start     : [x, y, z]  start position
    goal      : [x, y, z]  goal position
    obstacles : [(pos, radius), ...]  list of (np.array(3,), float)
    k_att     : attractive gain
    k_rep     : repulsive gain
    d0        : repulsive influence distance [m]
    max_vel   : max velocity along path [m/s]
    R_drone   : drone collision radius [m]
    step_size : integration step [m]
    """

    def __init__(self, start, goal, obstacles,
                 k_att=1.0, k_rep=0.8, d0=2.5,
                 max_vel=1.5, R_drone=0.35, step_size=0.05):
        self._start = np.array(start, dtype=float)
        self._goal  = np.array(goal, dtype=float)

        path = self._plan_path(
            self._start, self._goal, obstacles,
            k_att, k_rep, d0, R_drone, step_size)

        waypoints = []
        for i, p in enumerate(path):
            if i == 0 or i == len(path) - 1:
                waypoints.append({'pos': p, 'vel': [0, 0, 0]})
            else:
                direction = path[i + 1] - path[i - 1]
                d = np.linalg.norm(direction)
                if d > 1e-6:
                    vel = direction / d * max_vel * 0.6
                else:
                    vel = np.zeros(3)
                waypoints.append({'pos': p, 'vel': vel})

        self._traj = WaypointTrajectory(waypoints, max_vel=max_vel)

        min_dists = []
        for obs_pos, obs_r in obstacles:
            dists = [np.linalg.norm(p[:2] - obs_pos[:2]) for p in path]
            min_dists.append((np.min(dists), obs_r))

        print(f"[APF] {len(path)} waypoints, {self._traj.total_duration():.1f}s")
        for i, (md, r) in enumerate(min_dists):
            print(f"  obs{i+1}: min_dist={md:.2f}m  (r={r:.1f}, safe={r+R_drone:.2f})")

    @staticmethod
    def _plan_path(start, goal, obstacles, k_att, k_rep, d0, R_drone, step_size):
        pos = start.copy()
        alt = goal[2]
        path = [pos.copy()]
        max_steps = 5000
        goal_tol = 0.3

        line_dir = goal[:2] - start[:2]
        obs_signs = []
        for obs_pos, obs_r in obstacles:
            obs_off = obs_pos[:2] - start[:2]
            cross = line_dir[0] * obs_off[1] - line_dir[1] * obs_off[0]
            obs_signs.append(+1 if cross >= 0 else -1)
            side = "left" if cross > 0 else ("right" if cross < 0 else "center")
            print(f"  obs({obs_pos[0]:.0f},{obs_pos[1]:.0f}): {side} of line -> pass {'right' if cross >= 0 else 'left'}")

        for _ in range(max_steps):
            if np.linalg.norm(pos[:2] - goal[:2]) < goal_tol:
                break

            diff_goal = goal[:2] - pos[:2]
            f_att = k_att * diff_goal
            f_att_norm = np.linalg.norm(f_att)
            if f_att_norm > k_att:
                f_att = f_att / f_att_norm * k_att

            f_rep = np.zeros(2)
            for idx, (obs_pos, obs_r) in enumerate(obstacles):
                diff = pos[:2] - obs_pos[:2]
                dist = np.linalg.norm(diff)
                margin = dist - obs_r - R_drone
                if margin < d0 and margin > 0.01:
                    strength = k_rep * (1.0/margin - 1.0/d0) * (1.0/margin**2)
                    radial = diff / dist
                    sign = obs_signs[idx]
                    tangent = sign * np.array([-diff[1], diff[0]]) / dist
                    f_rep += strength * (radial + 0.5 * tangent)
                elif margin <= 0.01:
                    f_rep += k_rep * 100.0 * (diff / max(dist, 0.01))

            f_total = f_att + f_rep
            f_norm = np.linalg.norm(f_total)
            if f_norm < 1e-6:
                f_total = f_att + np.array([0.0, 0.1])

            direction = f_total / np.linalg.norm(f_total)
            pos[:2] = pos[:2] + step_size * direction
            pos[2] = alt
            path.append(pos.copy())

        path.append(goal.copy())

        raw_len = len(path)
        ys = [p[1] for p in path]
        print(f"[APF] raw path: {raw_len} pts, y range [{min(ys):.2f}, {max(ys):.2f}]")

        path = APFTrajectory._downsample(path, min_dist=2.0)
        return path

    @staticmethod
    def _downsample(path, min_dist=0.5):
        result = [path[0]]
        for p in path[1:-1]:
            if np.linalg.norm(p - result[-1]) >= min_dist:
                result.append(p)
        result.append(path[-1])
        return result

    def state_at(self, t):
        return self._traj.state_at(t)

    def get_horizon(self, t_now, N, Ts):
        return self._traj.get_horizon(t_now, N, Ts)

    def total_duration(self):
        return self._traj.total_duration()

    def initial_state(self):
        return self._traj.initial_state()
