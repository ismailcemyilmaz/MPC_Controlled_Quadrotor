"""
Quadrotor continuous-time model for acados
==========================================
State  x ∈ R^13 : [px,py,pz, vx,vy,vz, qw,qx,qy,qz, p,q,r]
Input  u ∈ R^4  : [f_total, tau_x, tau_y, tau_z]

Used by both mpc_solver.py and local_planner_mpc.py.
"""

import casadi as cs
from acados_template import AcadosModel

G = 9.81


def export_quadrotor_model(mass: float = 1.0,
                            I_diag: tuple = (22.916e-3, 22.916e-3, 22.132e-3),
                            name: str = 'quadrotor') -> AcadosModel:
    """
    Build AcadosModel with explicit/implicit ODE expressions.

    Parameters
    ----------
    mass   : vehicle mass [kg]
    I_diag : (Ixx, Iyy, Izz) inertia diagonal [kg·m²]
    name   : model name (must be unique when multiple models used together)
    """
    model = AcadosModel()
    model.name = name

    x    = cs.SX.sym('x', 13)
    xdot = cs.SX.sym('xdot', 13)
    u    = cs.SX.sym('u', 4)

    px, py, pz         = x[0], x[1], x[2]
    vx, vy, vz         = x[3], x[4], x[5]
    qw, qx, qy, qz     = x[6], x[7], x[8], x[9]
    p,  q,  r          = x[10], x[11], x[12]

    f_total            = u[0]
    tau_x, tau_y, tau_z = u[1], u[2], u[3]

    m             = mass
    Ixx, Iyy, Izz = I_diag

    # Translational acceleration (world frame)
    ax = 2.0*(qw*qy + qx*qz) * f_total / m
    ay = 2.0*(qy*qz - qw*qx) * f_total / m
    az = (1.0 - 2.0*qx*qx - 2.0*qy*qy) * f_total / m - G

    # Quaternion kinematics
    qw_dot = 0.5*(-qx*p - qy*q - qz*r)
    qx_dot = 0.5*( qw*p + qy*r - qz*q)
    qy_dot = 0.5*( qw*q - qx*r + qz*p)
    qz_dot = 0.5*( qw*r + qx*q - qy*p)

    # Euler equations
    p_dot = (tau_x - (Izz - Iyy)*q*r) / Ixx
    q_dot = (tau_y - (Ixx - Izz)*p*r) / Iyy
    r_dot = (tau_z - (Iyy - Ixx)*p*q) / Izz

    f_expl = cs.vertcat(
        vx, vy, vz,
        ax, ay, az,
        qw_dot, qx_dot, qy_dot, qz_dot,
        p_dot, q_dot, r_dot
    )

    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl
    model.x    = x
    model.xdot = xdot
    model.u    = u

    return model
