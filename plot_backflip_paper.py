"""
Paper-ready backflip plots — single column (3.5 inch) format.

Usage:
    python3 plot_backflip_paper.py [log_path]
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.0,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.3,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

here = os.path.dirname(os.path.abspath(__file__))
ws = os.path.normpath(os.path.join(here, '..', '..'))
DEFAULT_LOG = os.path.join(ws, 'logs', 'mpc', 'backflip_ilc', 'mpc_log_best.npz')

COL_W = 3.5  # single column width in inches


def find_phases(x, t):
    q_rate = x[:, 11]
    flip_start = next((i for i in range(len(q_rate)) if abs(q_rate[i]) > 3.0), None)
    popup_start = flip_start
    for i in range(flip_start - 1, 0, -1):
        if x[i, 5] < 0.3:
            popup_start = i
            break
    flip_end = next((i for i in range(flip_start, len(q_rate))
                     if abs(q_rate[i]) < 1.0), flip_start)
    coast_start = None
    for i in range(flip_start, flip_end):
        if q_rate[i] > 7.0 and i + 5 < len(q_rate):
            if q_rate[i + 5] - q_rate[i] < 0.5:
                coast_start = i
                break
    if coast_start is None:
        coast_start = flip_start + (flip_end - flip_start) // 3
    decel_start = None
    for i in range(coast_start, flip_end):
        if q_rate[i] > q_rate[i + 1] + 0.3:
            decel_start = i
            break
    if decel_start is None:
        decel_start = flip_start + 2 * (flip_end - flip_start) // 3
    settled = None
    for i in range(flip_end, len(x)):
        vlat = np.sqrt(x[i, 3]**2 + x[i, 4]**2)
        dt = t[i] - t[flip_end]
        if dt > 1.0 and vlat < 0.5 and abs(x[i, 5]) < 0.5:
            settled = i
            break
    return {
        'popup': popup_start, 'flip_start': flip_start,
        'coast': coast_start, 'decel': decel_start,
        'flip_end': flip_end, 'settled': settled or len(x) - 1,
    }


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    if not os.path.exists(log_path):
        print(f"Not found: {log_path}")
        sys.exit(1)

    d = np.load(log_path)
    x = d['x']; t = d['t']
    ph = find_phases(x, t)
    t0 = t[ph['flip_start']]
    t_rel = t - t0
    hover_pos = x[ph['popup'], :3]

    phase_colors = {
        'Popup': '#e63946', 'Accel': '#f4a261',
        'Coast': '#2a9d8f', 'Decel': '#264653', 'Recovery': '#9b5de5',
    }
    spans = [
        ('Popup', t_rel[ph['popup']], t_rel[ph['flip_start']]),
        ('Accel', t_rel[ph['flip_start']], t_rel[ph['coast']]),
        ('Coast', t_rel[ph['coast']], t_rel[ph['decel']]),
        ('Decel', t_rel[ph['decel']], t_rel[ph['flip_end']]),
        ('Recovery', t_rel[ph['flip_end']], t_rel[ph['settled']]),
    ]

    def shade(ax):
        for name, ts, te in spans:
            ax.axvspan(ts, te, alpha=0.10, color=phase_colors[name])

    i0 = max(0, ph['popup'] - int(0.5 / 0.002))
    i1 = min(len(x), ph['settled'] + int(1.0 / 0.002))
    sl = slice(i0, i1)

    # ===== FIGURE 1: 3-panel analysis =====
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 4.5), sharex=True)

    # Panel 1: Attitude error + pitch rate (twin axis)
    ax = axes[0]
    shade(ax)
    qw = np.abs(x[sl, 6])
    att_err = 2 * np.degrees(np.arccos(np.clip(qw, 0, 1)))
    ln1 = ax.plot(t_rel[sl], att_err, color='#e63946', label='Attitude error')
    ax.set_ylabel('Att. error [°]')
    ax.set_ylim([-5, 195])
    ax.set_yticks([0, 45, 90, 135, 180])

    ax2 = ax.twinx()
    ln2 = ax2.plot(t_rel[sl], x[sl, 11], color='#264653', alpha=0.6,
                   linewidth=0.8, label='Pitch rate')
    ax2.set_ylabel('q [rad/s]', color='#264653')
    ax2.tick_params(axis='y', labelcolor='#264653')
    ax2.set_ylim([-3, 12])

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # Phase labels at top
    for name, ts, te in spans:
        mid = (ts + te) / 2
        ax.text(mid, 188, name, ha='center', va='top',
                fontsize=5.5, color=phase_colors[name], fontweight='bold')

    # Panel 2: Lateral position (drift from hover)
    ax = axes[1]
    shade(ax)
    dx = x[sl, 0] - hover_pos[0]
    dy = x[sl, 1] - hover_pos[1]
    drift = np.sqrt(dx**2 + dy**2)
    ax.plot(t_rel[sl], dx, color='#1f77b4', label='Δx')
    ax.plot(t_rel[sl], dy, color='#ff7f0e', label='Δy')
    ax.plot(t_rel[sl], drift, color='black', linewidth=0.8,
            linestyle='--', alpha=0.5, label='|Δxy|')
    ax.set_ylabel('Drift [m]')
    ax.legend(loc='upper right', ncol=3, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Altitude
    ax = axes[2]
    shade(ax)
    ax.plot(t_rel[sl], x[sl, 2], color='#2a9d8f', linewidth=1.2)
    ax.axhline(y=hover_pos[2], color='gray', linestyle=':', linewidth=0.6)
    ax.text(t_rel[i1 - 1], hover_pos[2] + 0.15, f'hover ({hover_pos[2]:.0f}m)',
            fontsize=6, color='gray', ha='right')
    ax.set_ylabel('Altitude [m]')
    ax.set_xlabel('Time relative to flip start [s]')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(h_pad=0.3)
    out1 = os.path.join(here, 'plots', 'backflip_paper_analysis.png')
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    fig.savefig(out1)
    fig.savefig(out1.replace('.png', '.pdf'))
    print(f'Saved: {out1}')

    # ===== FIGURE 2: XY drift top-view =====
    fig2, ax = plt.subplots(figsize=(COL_W, COL_W))

    seg_plot = [
        ('Popup', slice(ph['popup'], ph['flip_start'])),
        ('Flip', slice(ph['flip_start'], ph['flip_end'])),
        ('Recovery', slice(ph['flip_end'], ph['settled'])),
    ]
    for name, sl2 in seg_plot:
        c = phase_colors.get(name, phase_colors.get('Accel'))
        ax.plot(x[sl2, 0] - hover_pos[0], x[sl2, 1] - hover_pos[1],
                color=c, linewidth=1.2, label=name)

    ax.plot(0, 0, 'o', color='#2a9d8f', markersize=6, zorder=10, label='Hover')
    xe = x[ph['flip_end']]
    ax.plot(xe[0] - hover_pos[0], xe[1] - hover_pos[1], 'x',
            color='#e63946', markersize=7, markeredgewidth=1.5, zorder=10,
            label='Flip exit')
    xs = x[ph['settled']]
    ax.plot(xs[0] - hover_pos[0], xs[1] - hover_pos[1], 's',
            color='#9b5de5', markersize=5, zorder=10, label='Settled')

    for r in [1, 2, 3]:
        circle = plt.Circle((0, 0), r, fill=False, color='#ddd',
                             linestyle='--', linewidth=0.5)
        ax.add_patch(circle)
        ax.text(r * 0.71, r * 0.71, f'{r}m', fontsize=5.5, color='#bbb')

    ax.set_xlabel('Δx [m]')
    ax.set_ylabel('Δy [m]')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.2)

    sl_all = slice(ph['popup'], ph['settled'])
    lim = max(abs(x[sl_all, 0] - hover_pos[0]).max(),
              abs(x[sl_all, 1] - hover_pos[1]).max()) + 0.5
    lim = max(lim, 3.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    plt.tight_layout()
    out2 = os.path.join(here, 'plots', 'backflip_paper_xy.png')
    fig2.savefig(out2)
    fig2.savefig(out2.replace('.png', '.pdf'))
    print(f'Saved: {out2}')

    print('\nDone — paper figures saved (PNG + PDF)')


if __name__ == '__main__':
    main()
