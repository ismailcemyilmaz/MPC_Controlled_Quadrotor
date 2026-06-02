"""
APF Force Field Visualization
Plots the APF vector field, potential surface, and drone path.

Usage:
    python3 plot_apf_field.py [log_path] [--world original|dense]
    python3 plot_apf_field.py                                       # dense world, lidar log
    python3 plot_apf_field.py logs/mpc/slalom_reactive/mpc_log.npz --world original
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


WORLDS = {
    'original': {
        'obstacles': [
            (np.array([3.0, 0.0, 1.5]), 0.4),
            (np.array([6.0, 1.5, 1.5]), 0.4),
            (np.array([9.0, -1.0, 1.5]), 0.4),
        ],
        'goal': np.array([12.0, 0.0]),
        'R_drone': 0.65,
        'xlim': (-1, 14),
        'ylim': (-4, 4),
        'default_log': 'slalom_reactive',
    },
    'dense': {
        'obstacles': [
            (np.array([3.0, -1.5, 1.5]), 0.4),
            (np.array([6.0,  1.5, 1.5]), 0.4),
            (np.array([9.0, -1.5, 1.5]), 0.4),
            (np.array([12.0, 1.5, 1.5]), 0.4),
            (np.array([15.0,-1.5, 1.5]), 0.4),
        ],
        'goal': np.array([18.0, 0.0]),
        'R_drone': 0.50,
        'xlim': (-1, 20),
        'ylim': (-4, 4),
        'default_log': 'slalom_reactive_lidar',
    },
}


def compute_signs(obstacles, start, goal):
    line_dir = goal - start
    signs = []
    for obs_pos, _ in obstacles:
        cross = line_dir[0] * obs_pos[1] - line_dir[1] * obs_pos[0]
        signs.append(+1 if cross >= 0 else -1)
    return signs


def apf_force(pos_2d, goal_2d, obstacles, obs_signs,
              k_att=1.0, k_rep=0.8, d0=2.5, R_drone=0.50):
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
            sign = obs_signs[idx]
            tangent = sign * np.array([-diff[1], diff[0]]) / dist
            f_rep += strength * (radial + 0.5 * tangent)
        elif margin <= 0.01:
            f_rep += k_rep * 100.0 * (diff / max(dist, 0.01))
    return f_att + f_rep


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ws = os.path.normpath(os.path.join(here, '..', '..'))

    world_name = 'dense'
    log_path = None

    for arg in sys.argv[1:]:
        if arg.startswith('--world'):
            continue
        elif arg in WORLDS:
            world_name = arg
        elif os.path.exists(arg):
            log_path = arg
        elif sys.argv[sys.argv.index(arg) - 1] == '--world':
            world_name = arg

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--world' and i < len(sys.argv) - 1:
            world_name = sys.argv[i + 1]

    w = WORLDS[world_name]
    obstacles = w['obstacles']
    goal = w['goal']
    R_drone = w['R_drone']
    xlim = w['xlim']
    ylim = w['ylim']

    if log_path is None:
        log_path = os.path.join(ws, 'logs', 'mpc', w['default_log'], 'mpc_log.npz')

    if not os.path.exists(log_path):
        print(f"Log not found: {log_path}")
        sys.exit(1)

    print(f"World: {world_name} ({len(obstacles)} obstacles)")
    print(f"Log: {log_path}")

    d = np.load(log_path)
    x = d['x']

    start = np.array([0.0, 0.0])
    obs_signs = compute_signs(obstacles, start, goal)

    colors = ['#e63946', '#f4a261', '#2a9d8f', '#264653', '#9b5de5']

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # --- Plot 1: Force field + drone path ---
    ax = axes[0]
    ax.set_title('APF Force Field + Drone Path', fontsize=14, fontweight='bold')

    n_arrows_x, n_arrows_y = 35, 20
    xs = np.linspace(xlim[0], xlim[1], n_arrows_x)
    ys = np.linspace(ylim[0], ylim[1], n_arrows_y)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pos = np.array([X[i, j], Y[i, j]])
            inside = any(np.linalg.norm(pos - op[:2]) < r + 0.1
                         for op, r in obstacles)
            if not inside:
                f = apf_force(pos, goal, obstacles, obs_signs, R_drone=R_drone)
                fn = np.linalg.norm(f)
                if fn > 0:
                    U[i, j] = f[0] / fn
                    V[i, j] = f[1] / fn

    ax.quiver(X, Y, U, V, alpha=0.3, color='steelblue', scale=25)

    for i, (obs_pos, obs_r) in enumerate(obstacles):
        c = colors[i % len(colors)]
        circle = plt.Circle((obs_pos[0], obs_pos[1]), obs_r,
                             color=c, alpha=0.7, zorder=5)
        ax.add_patch(circle)
        safe = plt.Circle((obs_pos[0], obs_pos[1]), obs_r + R_drone,
                           color=c, alpha=0.1, fill=True,
                           linestyle='--', zorder=4)
        ax.add_patch(safe)
        side = 'N' if obs_pos[1] > 0 else 'S'
        ax.text(obs_pos[0], obs_pos[1] + obs_r + 0.3, f'obs{i+1} ({side})',
                ha='center', fontsize=9, fontweight='bold')

    ax.plot(x[:, 0], x[:, 1], 'k-', linewidth=2, label='Drone path', zorder=10)
    ax.plot(x[0, 0], x[0, 1], 'go', markersize=10, label='Start', zorder=11)
    ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal', zorder=11)

    for i, (obs_pos, obs_r) in enumerate(obstacles):
        dists = np.sqrt((x[:, 0] - obs_pos[0])**2 +
                        (x[:, 1] - obs_pos[1])**2)
        idx = np.argmin(dists)
        c = colors[i % len(colors)]
        ax.plot([x[idx, 0], obs_pos[0]], [x[idx, 1], obs_pos[1]],
                '--', color=c, linewidth=1, alpha=0.7)
        mx = (x[idx, 0] + obs_pos[0]) / 2
        my = (x[idx, 1] + obs_pos[1]) / 2
        ax.text(mx, my, f'{np.min(dists):.2f}m', fontsize=8, color=c,
                fontweight='bold')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Potential heatmap ---
    ax2 = axes[1]
    ax2.set_title('APF Potential Field (log scale)', fontsize=14, fontweight='bold')

    xs2 = np.linspace(xlim[0], xlim[1], 250)
    ys2 = np.linspace(ylim[0], ylim[1], 150)
    X2, Y2 = np.meshgrid(xs2, ys2)
    P = np.zeros_like(X2)

    for i in range(X2.shape[0]):
        for j in range(X2.shape[1]):
            pos = np.array([X2[i, j], Y2[i, j]])
            U_att = 0.5 * 1.0 * np.linalg.norm(pos - goal)**2
            U_rep = 0.0
            for obs_pos, obs_r in obstacles:
                dist = np.linalg.norm(pos - obs_pos[:2])
                margin = dist - obs_r - R_drone
                if margin < 2.5 and margin > 0.01:
                    U_rep += 0.5 * 0.8 * (1.0/margin - 1.0/2.5)**2
                elif margin <= 0.01:
                    U_rep += 100.0
            P[i, j] = U_att + U_rep

    im = ax2.pcolormesh(X2, Y2, np.log1p(P), cmap='hot_r', shading='auto')
    plt.colorbar(im, ax=ax2, label='log(1 + U)')

    for i, (obs_pos, obs_r) in enumerate(obstacles):
        circle = plt.Circle((obs_pos[0], obs_pos[1]), obs_r,
                             color='white', alpha=0.8, zorder=5)
        ax2.add_patch(circle)

    ax2.plot(x[:, 0], x[:, 1], 'cyan', linewidth=2,
             label='Drone path', zorder=10)
    ax2.plot(x[0, 0], x[0, 1], 'go', markersize=8, zorder=11)
    ax2.plot(goal[0], goal[1], 'w*', markersize=12, zorder=11)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_aspect('equal')
    ax2.legend(loc='upper left')

    plt.tight_layout()
    out = os.path.join(here, 'plots', f'apf_field_{world_name}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
