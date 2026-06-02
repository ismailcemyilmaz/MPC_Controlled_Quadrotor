"""
Backflip analysis plots for paper.

Usage:
    python3 plot_backflip.py [log_path]
    python3 plot_backflip.py                    # uses best run
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

here = os.path.dirname(os.path.abspath(__file__))
ws = os.path.normpath(os.path.join(here, '..', '..'))

DEFAULT_LOG = os.path.join(ws, 'logs', 'mpc', 'backflip_ilc', 'mpc_log_best.npz')

MASS = 1.280
G = 9.81


def find_phases(x, t):
    q_rate = x[:, 11]

    # Flip start: pitch rate > 3 rad/s
    flip_start = next((i for i in range(len(q_rate)) if abs(q_rate[i]) > 3.0), None)

    # Popup start: search backward from flip_start for when vz starts rising
    popup_start = flip_start
    for i in range(flip_start - 1, 0, -1):
        if x[i, 5] < 0.3:
            popup_start = i
            break

    # Flip end: pitch rate < 1 rad/s after flip
    flip_end = next((i for i in range(flip_start, len(q_rate))
                     if abs(q_rate[i]) < 1.0), flip_start)

    # Coast: pitch rate stops increasing
    coast_start = None
    for i in range(flip_start, flip_end):
        if q_rate[i] > 7.0 and i + 5 < len(q_rate):
            if q_rate[i+5] - q_rate[i] < 0.5:
                coast_start = i
                break
    if coast_start is None:
        coast_start = flip_start + (flip_end - flip_start) // 3

    # Decel: pitch rate starts decreasing
    decel_start = None
    for i in range(coast_start, flip_end):
        if q_rate[i] > q_rate[i+1] + 0.3:
            decel_start = i
            break
    if decel_start is None:
        decel_start = flip_start + 2 * (flip_end - flip_start) // 3

    # Recovery settled: v_lat < 0.5
    settled = None
    for i in range(flip_end, len(x)):
        vlat = np.sqrt(x[i, 3]**2 + x[i, 4]**2)
        dt = t[i] - t[flip_end]
        if dt > 1.0 and vlat < 0.5 and abs(x[i, 5]) < 0.5:
            settled = i
            break

    return {
        'popup': popup_start,
        'flip_start': flip_start,
        'coast': coast_start,
        'decel': decel_start,
        'flip_end': flip_end,
        'settled': settled or len(x) - 1,
    }


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    if not os.path.exists(log_path):
        print(f"Not found: {log_path}")
        sys.exit(1)

    d = np.load(log_path)
    x = d['x']; t = d['t']; u = d['u']
    print(f"Log: {log_path}  samples={len(x)}")

    ph = find_phases(x, t)
    t0_flip = t[ph['flip_start']]

    # Time relative to flip start
    t_rel = t - t0_flip

    # Hover = state just before popup (stable position before flip sequence)
    hover_pos = x[ph['popup'], :3].copy()
    # Use z at popup start (should be ~10m, not the peak during popup)
    print(f"Hover position: ({hover_pos[0]:.2f}, {hover_pos[1]:.2f}, {hover_pos[2]:.2f})")

    # Phase colors
    phase_colors = {
        'Popup': '#e63946',
        'Accel': '#f4a261',
        'Coast': '#2a9d8f',
        'Decel': '#264653',
        'Recovery': '#9b5de5',
    }

    phase_spans = [
        ('Popup', t_rel[ph['popup']], t_rel[ph['flip_start']]),
        ('Accel', t_rel[ph['flip_start']], t_rel[ph['coast']]),
        ('Coast', t_rel[ph['coast']], t_rel[ph['decel']]),
        ('Decel', t_rel[ph['decel']], t_rel[ph['flip_end']]),
        ('Recovery', t_rel[ph['flip_end']], t_rel[ph['settled']]),
    ]

    def shade_phases(ax):
        for name, t_start, t_end in phase_spans:
            ax.axvspan(t_start, t_end, alpha=0.12, color=phase_colors[name])

    def add_phase_labels(ax, y_pos):
        for name, t_start, t_end in phase_spans:
            mid = (t_start + t_end) / 2
            ax.text(mid, y_pos, name, ha='center', va='bottom',
                    fontsize=7, color=phase_colors[name], fontweight='bold',
                    rotation=0)

    # Crop: 1s before popup to 2s after settled
    i_start = max(0, ph['popup'] - int(1.0 / 0.002))
    i_end = min(len(x), ph['settled'] + int(2.0 / 0.002))

    # ==================== FIGURE 1: Main analysis ====================
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle('Quadrotor Backflip — Lupashin Bang-Coast-Bang',
                 fontsize=14, fontweight='bold')

    # --- Position ---
    ax = axes[0]
    shade_phases(ax)
    ax.plot(t_rel[i_start:i_end], x[i_start:i_end, 0] - hover_pos[0],
            label='x (lateral)', linewidth=1.5)
    ax.plot(t_rel[i_start:i_end], x[i_start:i_end, 1] - hover_pos[1],
            label='y (lateral)', linewidth=1.5)
    ax.plot(t_rel[i_start:i_end], x[i_start:i_end, 2] - hover_pos[2],
            label='z (altitude)', linewidth=1.5)
    ax.set_ylabel('Position [m]\n(relative to hover)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Position', fontsize=11)

    # --- Velocity ---
    ax = axes[1]
    shade_phases(ax)
    ax.plot(t_rel[i_start:i_end], x[i_start:i_end, 3],
            label='vx', linewidth=1.5)
    ax.plot(t_rel[i_start:i_end], x[i_start:i_end, 4],
            label='vy', linewidth=1.5)
    ax.plot(t_rel[i_start:i_end], x[i_start:i_end, 5],
            label='vz', linewidth=1.5)
    ax.set_ylabel('Velocity [m/s]')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Velocity', fontsize=11)

    # --- Attitude (quaternion → angle from upright) ---
    ax = axes[2]
    shade_phases(ax)
    qw = x[i_start:i_end, 6]
    angle_from_upright = 2 * np.degrees(np.arccos(np.clip(np.abs(qw), 0, 1)))
    ax.plot(t_rel[i_start:i_end], angle_from_upright,
            color='#e63946', linewidth=1.5, label='Attitude error')
    ax.set_ylabel('Angle from\nupright [deg]')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Attitude Error', fontsize=11)
    ax.set_ylim([-5, 200])

    # --- Angular rate (pitch) ---
    ax = axes[3]
    shade_phases(ax)
    ax.plot(t_rel[i_start:i_end], x[i_start:i_end, 11],
            color='#264653', linewidth=1.5, label='Pitch rate (q)')
    ax.plot(t_rel[i_start:i_end], np.linalg.norm(x[i_start:i_end, 10:13], axis=1),
            color='#aaa', linewidth=1, linestyle='--', label='|omega|')
    ax.set_ylabel('Angular rate\n[rad/s]')
    ax.set_xlabel('Time relative to flip start [s]')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Angular Rates', fontsize=11)

    plt.tight_layout()
    out1 = os.path.join(here, 'plots', 'backflip_analysis.png')
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    plt.savefig(out1, dpi=150)
    print(f'Saved: {out1}')

    # ==================== FIGURE 2: 3D trajectory + XY drift ====================
    fig2, (ax3d, ax_xy) = plt.subplots(1, 2, figsize=(16, 7),
                                        subplot_kw={'projection': None})
    fig2.suptitle('Backflip Trajectory', fontsize=14, fontweight='bold')

    # 3D trajectory
    ax3d = fig2.add_subplot(121, projection='3d')
    flip_region = slice(ph['popup'], min(ph['settled'] + 300, len(x)))

    # Color by phase
    segments = [
        ('Popup', slice(ph['popup'], ph['flip_start']), phase_colors['Popup']),
        ('Flip', slice(ph['flip_start'], ph['flip_end']), phase_colors['Accel']),
        ('Recovery', slice(ph['flip_end'], ph['settled']), phase_colors['Recovery']),
    ]
    for name, sl, color in segments:
        ax3d.plot(x[sl, 0], x[sl, 1], x[sl, 2],
                  color=color, linewidth=2, label=name)

    ax3d.scatter(*hover_pos, color='green', s=80, zorder=10, marker='o', label='Hover')
    ax3d.scatter(*x[ph['flip_end'], :3], color='red', s=80, zorder=10, marker='x',
                 label='Flip exit')
    if ph['settled'] < len(x):
        ax3d.scatter(*x[ph['settled'], :3], color='purple', s=80, zorder=10,
                     marker='s', label='Settled')

    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.legend(fontsize=8, loc='upper left')
    ax3d.set_title('3D Trajectory', fontsize=11)

    # XY drift plot (only popup + flip + recovery)
    ax_xy = fig2.add_subplot(122)
    for name, sl, color in segments:
        if name == 'Climb':
            continue
        ax_xy.plot(x[sl, 0] - hover_pos[0], x[sl, 1] - hover_pos[1],
                   color=color, linewidth=2, label=name)

    ax_xy.plot(0, 0, 'go', markersize=12, label='Hover', zorder=10)
    xe = x[ph['flip_end']]
    ax_xy.plot(xe[0] - hover_pos[0], xe[1] - hover_pos[1], 'rx',
               markersize=12, markeredgewidth=2, label='Flip exit', zorder=10)
    if ph['settled'] < len(x):
        xs = x[ph['settled']]
        ax_xy.plot(xs[0] - hover_pos[0], xs[1] - hover_pos[1], 'ms',
                   markersize=10, label='Settled', zorder=10)

    # Drift circle annotations
    for r in [1, 2, 3, 5]:
        circle = plt.Circle((0, 0), r, fill=False, color='#ccc',
                             linestyle='--', linewidth=0.8)
        ax_xy.add_patch(circle)
        ax_xy.text(r * 0.707, r * 0.707, f'{r}m', fontsize=7, color='#aaa')

    ax_xy.set_xlabel('X drift [m]')
    ax_xy.set_ylabel('Y drift [m]')
    ax_xy.set_aspect('equal')
    ax_xy.legend(fontsize=9, loc='upper right')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.set_title('XY Drift (relative to hover)', fontsize=11)

    sl_drift = slice(ph['popup'], ph['settled'])
    lim = max(abs(x[sl_drift, 0] - hover_pos[0]).max(),
              abs(x[sl_drift, 1] - hover_pos[1]).max()) + 0.5
    lim = max(lim, 3.0)
    ax_xy.set_xlim(-lim, lim)
    ax_xy.set_ylim(-lim, lim)

    plt.tight_layout()
    out2 = os.path.join(here, 'plots', 'backflip_trajectory.png')
    plt.savefig(out2, dpi=150)
    print(f'Saved: {out2}')

    # ==================== FIGURE 3: Recovery detail ====================
    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig3.suptitle('Post-Flip Recovery Detail', fontsize=14, fontweight='bold')

    rec_start = ph['flip_end']
    rec_end = min(ph['settled'] + 300, len(x))
    t_rec = t[rec_start:rec_end] - t[rec_start]

    # Lateral drift distance
    ax = axes3[0]
    dx = x[rec_start:rec_end, 0] - hover_pos[0]
    dy = x[rec_start:rec_end, 1] - hover_pos[1]
    drift_dist = np.sqrt(dx**2 + dy**2)
    ax.plot(t_rec, drift_dist, 'k-', linewidth=2, label='XY drift distance')
    ax.fill_between(t_rec, 0, drift_dist, alpha=0.15, color='steelblue')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Settled (0.8m)')
    peak_t = t_rec[np.argmax(drift_dist)]
    peak_d = np.max(drift_dist)
    ax.annotate(f'Peak: {peak_d:.1f}m', xy=(peak_t, peak_d),
                xytext=(peak_t + 0.5, peak_d + 0.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')
    ax.set_ylabel('Drift [m]')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Lateral Drift from Hover Position', fontsize=11)

    # Velocity
    ax = axes3[1]
    vlat = np.sqrt(x[rec_start:rec_end, 3]**2 + x[rec_start:rec_end, 4]**2)
    ax.plot(t_rec, vlat, color='#e63946', linewidth=1.5, label='v_lateral')
    ax.plot(t_rec, x[rec_start:rec_end, 5], color='#2a9d8f',
            linewidth=1.5, label='vz')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Velocity [m/s]')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Recovery Velocities', fontsize=11)

    # Altitude
    ax = axes3[2]
    ax.plot(t_rec, x[rec_start:rec_end, 2], color='#264653', linewidth=2)
    ax.axhline(y=hover_pos[2], color='green', linestyle='--', alpha=0.7,
               label=f'Hover ({hover_pos[2]:.1f}m)')
    ax.set_ylabel('Altitude [m]')
    ax.set_xlabel('Time after flip exit [s]')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Altitude Recovery', fontsize=11)

    plt.tight_layout()
    out3 = os.path.join(here, 'plots', 'backflip_recovery.png')
    plt.savefig(out3, dpi=150)
    print(f'Saved: {out3}')

    print('\nDone — 3 figures saved to plots/')


if __name__ == '__main__':
    main()
