"""
plot_mpc_log.py — Quadrotor MPC Log Görselleştirici
====================================================
Kullanım:
    python3 plot_mpc_log.py                          # son log (logs/mpc/*/mpc_log.npz)
    python3 plot_mpc_log.py path/to/mpc_log.npz
    python3 plot_mpc_log.py logs/mpc/run5/mpc_log.npz --save

Her grafik başlığında ilişkili MPC parametresi belirtilmiştir.
"""

import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Argüman ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='MPC log plotter')
parser.add_argument('log', nargs='?', default=None,
                    help='.npz log dosyası yolu (verilmezse en son log bulunur)')
parser.add_argument('--save', action='store_true',
                    help='PNG olarak kaydet, ekranda gösterme')
parser.add_argument('--dpi', type=int, default=150)
args = parser.parse_args()

# ── Log dosyasını bul ─────────────────────────────────────────────────────────
if args.log:
    log_path = args.log
else:
    candidates = sorted(glob.glob('logs/mpc/hover/mpc_log.npz', recursive=True))
    if not candidates:
        # Çalışma dizininde ara
        candidates = sorted(glob.glob('logs/mpc/hover/mpc_log.npz', recursive=True))
    if not candidates:
        print("Hata: mpc_log.npz bulunamadı. Yol belirtin: python3 plot_mpc_log.py <dosya>")
        sys.exit(1)
    log_path = candidates[-1]

print(f"[plot] Log: {log_path}")
data      = np.load(log_path)
t         = data['t']           # (N,)
x         = data['x']           # (N, 13)  [px py pz vx vy vz qw qx qy qz p q r]
xref      = data['xref']        # (N, 13)
u_raw     = data['u']           # (M, 4)   [f tau_x tau_y tau_z]
mpc_times = data['mpc_times']   # (M,)
n_obs     = data['n_obs'] if 'n_obs' in data else np.zeros(len(u_raw))

# ── u'yu t zamanına hizala ────────────────────────────────────────────────────
# u ve mpc_times MPC adımlarında kaydedilmiş (daha seyrek)
t_u = np.linspace(t[0], t[-1], len(u_raw))

# Kolayca erişim
px, py, pz     = x[:,0], x[:,1], x[:,2]
vx, vy, vz     = x[:,3], x[:,4], x[:,5]
qw, qx, qy, qz = x[:,6], x[:,7], x[:,8], x[:,9]

rpx, rpy, rpz   = xref[:,0], xref[:,1], xref[:,2]
rvx, rvy, rvz   = xref[:,3], xref[:,4], xref[:,5]
rqw,rqx,rqy,rqz = xref[:,6], xref[:,7], xref[:,8], xref[:,9]

f_total = u_raw[:,0]
tau_x   = u_raw[:,1]
tau_y   = u_raw[:,2]
tau_z   = u_raw[:,3]

# ── Hesaplamalar ──────────────────────────────────────────────────────────────

# 1. Pozisyon hatası  (Q_pos)
pos_err = np.linalg.norm(x[:,:3] - xref[:,:3], axis=1)
ex = px - rpx;  ey = py - rpy;  ez = pz - rpz

# 2. Hız hatası  (Q_vel)
vel_err = np.linalg.norm(x[:,3:6] - xref[:,3:6], axis=1)
evx = vx - rvx;  evy = vy - rvy;  evz = vz - rvz

# 3. İvme (referans xref'ten elde: a = dvref/dt) — sonlu fark
dt      = np.diff(t, prepend=t[0])
dt      = np.where(dt < 1e-6, 1e-6, dt)
ax_act  = np.gradient(vx, t)
ay_act  = np.gradient(vy, t)
az_act  = np.gradient(vz, t) + 9.81   # yerçekimi çıkar
ax_ref  = np.gradient(rvx, t)
ay_ref  = np.gradient(rvy, t)
az_ref  = np.gradient(rvz, t) + 9.81
acc_err = np.sqrt((ax_act-ax_ref)**2 + (ay_act-ay_ref)**2 + (az_act-az_ref)**2)

# 4. Quaternion hatası → SO(3) uzaklığı  (Q_att)
# q_err = q_ref^{-1} ⊗ q_act;  hata = 2*arccos(|q_err_w|)
def quat_inv(w,x,y,z):
    return w,-x,-y,-z
def quat_mul(w1,x1,y1,z1, w2,x2,y2,z2):
    return (w1*w2-x1*x2-y1*y2-z1*z2,
            w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2,
            w1*z2+x1*y2-y1*x2+z1*w2)

riw,rix,riy,riz = quat_inv(rqw,rqx,rqy,rqz)
ew,ex_q,ey_q,ez_q = quat_mul(riw,rix,riy,riz, qw,qx,qy,qz)
att_err_rad = 2.0 * np.arccos(np.clip(np.abs(ew), 0, 1))
att_err_deg = np.degrees(att_err_rad)

# Euler açıları (görselleştirme için)
roll  = np.degrees(np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2)))
pitch = np.degrees(np.arcsin(np.clip(2*(qw*qy-qz*qx),-1,1)))
yaw   = np.degrees(np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2)))

rroll  = np.degrees(np.arctan2(2*(rqw*rqx+rqy*rqz), 1-2*(rqx**2+rqy**2)))
rpitch = np.degrees(np.arcsin(np.clip(2*(rqw*rqy-rqz*rqx),-1,1)))
ryaw   = np.degrees(np.arctan2(2*(rqw*rqz+rqx*rqy), 1-2*(rqy**2+rqz**2)))

# 5. Angular rate hatası  (Q_omega / Q_omega_r)
omega_err = np.linalg.norm(x[:,10:13] - xref[:,10:13], axis=1)
ep_rate = x[:,10]-xref[:,10];  eq_rate = x[:,11]-xref[:,11];  er_rate = x[:,12]-xref[:,12]

# 6. Thrust: f range, f_min ihlali  (R_f, f_min, f_max_scale)
MASS  = 1.28;  G = 9.81
f_hover   = MASS * G
f_min_val = 0.05 * f_hover
f_max_val = 2.5  * f_hover
at_fmin   = f_total <= f_min_val * 1.02
at_fmax   = f_total >= f_max_val * 0.98
f_err     = f_total - f_hover           # hover'dan sapma

# 7. Torque: tau satürasyon  (R_tau, R_tau_z, tau_max, tau_z_max)
tau_max_val  = 0.80
tau_z_max_val= 0.12
sat_xy = np.abs(tau_x) >= tau_max_val*0.99
sat_z  = np.abs(tau_z) >= tau_z_max_val*0.99

# 8. Landing cone  (alpha_land, W_land)
alpha_land = 2.0
land_cone  = vz + alpha_land * pz      # >= 0 olmalı
land_violation = np.clip(-land_cone, 0, None)  # ihlal miktarı

# 9. Engel slack (W_obs / W_land etkisi için gösterim)
# n_obs zaten loglarda var

# ── Stil ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   '#F7F7F7',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'grid.linewidth':   0.5,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'sans-serif',
    'font.size':        9,
    'axes.titlesize':   10,
    'axes.titleweight': 'bold',
    'axes.labelsize':   8,
    'lines.linewidth':  1.4,
    'legend.fontsize':  8,
    'legend.framealpha':0.7,
})

C_ACT  = '#2B6CB0'   # mavi   — gerçek
C_REF  = '#D69E2E'   # amber  — referans
C_ERR  = '#C53030'   # kırmızı — hata
C_F    = '#276749'   # yeşil  — thrust
C_LAND = '#744210'   # kahve  — landing cone
C_OBS  = '#553C9A'   # mor    — engel
C_SAT  = '#E53E3E'   # kırmızı — satürasyon

tag = os.path.basename(os.path.dirname(log_path))

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Tracking Errors (Q_pos, Q_vel, Q_att, Q_omega)
# ════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(4, 3, figsize=(16, 13))
fig1.suptitle(f'Tracking Errors — {tag}', fontsize=13, fontweight='bold', y=0.98)

# ── Row 0: Position Error  (Q_pos) ───────────────────────────────────────────
axes[0,0].plot(t, pos_err, color=C_ERR)
axes[0,0].set_title('Position Error  ‖Δp‖  // Q_pos')
axes[0,0].set_ylabel('m')
axes[0,0].axhline(0, color='k', lw=0.6, ls='--')

axes[0,1].plot(t, ex, color=C_ACT, label='ex')
axes[0,1].plot(t, ey, color=C_REF, label='ey')
axes[0,1].plot(t, ez, color=C_ERR, label='ez')
axes[0,1].set_title('Position Error per Axis  // Q_pos')
axes[0,1].set_ylabel('m')
axes[0,1].legend()

axes[0,2].plot(t, pz, color=C_ACT, label='z actual')
axes[0,2].plot(t, rpz, color=C_REF, ls='--', label='z ref')
axes[0,2].set_title('Altitude  z(t)  // Q_pos')
axes[0,2].set_ylabel('m')
axes[0,2].legend()

# ── Row 1: Velocity Error  (Q_vel) ───────────────────────────────────────────
axes[1,0].plot(t, vel_err, color=C_ERR)
axes[1,0].set_title('Velocity Error  ‖Δv‖  // Q_vel')
axes[1,0].set_ylabel('m/s')

axes[1,1].plot(t, evx, color=C_ACT, label='evx')
axes[1,1].plot(t, evy, color=C_REF, label='evy')
axes[1,1].plot(t, evz, color=C_ERR, label='evz')
axes[1,1].set_title('Velocity Error per Axis  // Q_vel')
axes[1,1].set_ylabel('m/s')
axes[1,1].legend()

axes[1,2].plot(t, vz, color=C_ACT, label='vz actual')
axes[1,2].plot(t, rvz, color=C_REF, ls='--', label='vz ref')
axes[1,2].axhline(0, color='k', lw=0.5, ls=':')
axes[1,2].set_title('Vertical Velocity  vz(t)  // Q_vel')
axes[1,2].set_ylabel('m/s')
axes[1,2].legend()

# ── Row 2: Acceleration Error  (Q_vel — ivme tracking) ───────────────────────
axes[2,0].plot(t, acc_err, color=C_ERR)
axes[2,0].set_title('Acceleration Error  ‖Δa‖  // Q_vel (ivme)')
axes[2,0].set_ylabel('m/s²')

axes[2,1].plot(t, ax_act-ax_ref, color=C_ACT, label='eax')
axes[2,1].plot(t, ay_act-ay_ref, color=C_REF, label='eay')
axes[2,1].plot(t, az_act-az_ref, color=C_ERR, label='eaz')
axes[2,1].set_title('Acceleration Error per Axis  // Q_vel')
axes[2,1].set_ylabel('m/s²')
axes[2,1].legend()

axes[2,2].plot(t, az_act, color=C_ACT, label='az actual')
axes[2,2].plot(t, az_ref, color=C_REF, ls='--', label='az ref')
axes[2,2].set_title('Vertical Acceleration  az(t)  // Q_vel')
axes[2,2].set_ylabel('m/s²')
axes[2,2].legend()

# ── Row 3: Attitude Error  (Q_att) ───────────────────────────────────────────
axes[3,0].plot(t, att_err_deg, color=C_ERR)
axes[3,0].set_title('Attitude Error  2·arccos|qe_w|  // Q_att')
axes[3,0].set_ylabel('deg')
axes[3,0].set_xlabel('t (s)')

axes[3,1].plot(t, roll,  color=C_ACT,  label='roll act')
axes[3,1].plot(t, rroll, color=C_ACT,  ls='--', alpha=0.5, label='roll ref')
axes[3,1].plot(t, pitch, color=C_REF,  label='pitch act')
axes[3,1].plot(t, rpitch,color=C_REF,  ls='--', alpha=0.5, label='pitch ref')
axes[3,1].set_title('Roll / Pitch  // Q_att')
axes[3,1].set_ylabel('deg')
axes[3,1].set_xlabel('t (s)')
axes[3,1].legend(ncol=2)

axes[3,2].plot(t, yaw,  color=C_ACT, label='yaw act')
axes[3,2].plot(t, ryaw, color=C_REF, ls='--', label='yaw ref')
axes[3,2].set_title('Yaw  // Q_att')
axes[3,2].set_ylabel('deg')
axes[3,2].set_xlabel('t (s)')
axes[3,2].legend()

for ax in axes.flat:
    ax.set_xlabel('t (s)')

fig1.tight_layout()

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Angular Rate + Control Inputs (Q_omega, R_f, R_tau, R_tau_z)
# ════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(3, 3, figsize=(16, 10))
fig2.suptitle(f'Angular Rates & Control Inputs — {tag}', fontsize=13, fontweight='bold', y=0.98)

# ── Row 0: Angular Rate Error  (Q_omega / Q_omega_r) ─────────────────────────
axes2[0,0].plot(t, omega_err, color=C_ERR)
axes2[0,0].set_title('Angular Rate Error  ‖Δω‖  // Q_omega / Q_omega_r')
axes2[0,0].set_ylabel('rad/s')

axes2[0,1].plot(t, ep_rate, color=C_ACT, label='ep (roll)')
axes2[0,1].plot(t, eq_rate, color=C_REF, label='eq (pitch)')
axes2[0,1].plot(t, er_rate, color=C_ERR, label='er (yaw)')
axes2[0,1].set_title('Angular Rate Error per Axis  // Q_omega / Q_omega_r')
axes2[0,1].set_ylabel('rad/s')
axes2[0,1].legend()

axes2[0,2].plot(t, x[:,10], color=C_ACT, label='p actual')
axes2[0,2].plot(t, xref[:,10], color=C_ACT, ls='--', alpha=0.5, label='p ref')
axes2[0,2].plot(t, x[:,11], color=C_REF, label='q actual')
axes2[0,2].plot(t, xref[:,11], color=C_REF, ls='--', alpha=0.5, label='q ref')
axes2[0,2].set_title('Body Rates p, q  // Q_omega')
axes2[0,2].set_ylabel('rad/s')
axes2[0,2].legend(ncol=2)

# ── Row 1: Thrust  (R_f, f_min, f_max_scale) ─────────────────────────────────
axes2[1,0].plot(t_u, f_total, color=C_F)
axes2[1,0].axhline(f_hover,   color='k',    lw=1,   ls='--', label=f'f_hover={f_hover:.1f}N')
axes2[1,0].axhline(f_min_val, color=C_SAT,  lw=1,   ls=':',  label=f'f_min={f_min_val:.2f}N')
axes2[1,0].axhline(f_max_val, color=C_LAND, lw=1,   ls=':',  label=f'f_max={f_max_val:.1f}N')
axes2[1,0].fill_between(t_u, f_total, f_hover, where=(at_fmin), color=C_SAT, alpha=0.25, label='at f_min')
axes2[1,0].set_title('Total Thrust f  // R_f, f_min, f_max_scale')
axes2[1,0].set_ylabel('N')
axes2[1,0].legend(fontsize=7)

axes2[1,1].plot(t_u, f_err, color=C_ERR)
axes2[1,1].axhline(0, color='k', lw=0.7, ls='--')
axes2[1,1].fill_between(t_u, f_err, 0, where=(f_err<0), color=C_SAT, alpha=0.2, label='sub-hover')
axes2[1,1].set_title('Thrust Deviation  f − f_hover  // R_f')
axes2[1,1].set_ylabel('N')
axes2[1,1].legend()

# Thrust histogram
axes2[1,2].hist(f_total, bins=30, color=C_F, edgecolor='white', linewidth=0.3)
axes2[1,2].axvline(f_hover,   color='k',   lw=1.5, ls='--', label=f'hover')
axes2[1,2].axvline(f_min_val, color=C_SAT, lw=1.5, ls=':',  label=f'f_min')
axes2[1,2].axvline(f_max_val, color=C_LAND,lw=1.5, ls=':',  label=f'f_max')
axes2[1,2].set_title('Thrust Distribution  // R_f, f_min, f_max_scale')
axes2[1,2].set_xlabel('f (N)')
axes2[1,2].set_ylabel('count')
axes2[1,2].legend()

# ── Row 2: Torques  (R_tau, R_tau_z, tau_max, tau_z_max) ─────────────────────
axes2[2,0].plot(t_u, tau_x, color=C_ACT, label='τx')
axes2[2,0].plot(t_u, tau_y, color=C_REF, label='τy')
axes2[2,0].axhline( tau_max_val, color=C_SAT, lw=1, ls=':', label=f'±{tau_max_val}')
axes2[2,0].axhline(-tau_max_val, color=C_SAT, lw=1, ls=':')
axes2[2,0].fill_between(t_u, tau_max_val, tau_max_val*1.05, color=C_SAT, alpha=0.15)
axes2[2,0].fill_between(t_u,-tau_max_val,-tau_max_val*1.05, color=C_SAT, alpha=0.15)
axes2[2,0].set_title('Roll/Pitch Torques  τx, τy  // R_tau, tau_max')
axes2[2,0].set_ylabel('Nm')
axes2[2,0].legend()

axes2[2,1].plot(t_u, tau_z, color=C_ACT)
axes2[2,1].axhline( tau_z_max_val, color=C_SAT, lw=1, ls=':', label=f'±{tau_z_max_val}')
axes2[2,1].axhline(-tau_z_max_val, color=C_SAT, lw=1, ls=':')
axes2[2,1].fill_between(t_u, np.where(sat_z, tau_z, np.nan),  tau_z_max_val,  color=C_SAT, alpha=0.25, label='saturated')
axes2[2,1].set_title('Yaw Torque  τz  // R_tau_z, tau_z_max')
axes2[2,1].set_ylabel('Nm')
axes2[2,1].legend()

# Saturation timeline
sat_xy_f = sat_xy.astype(float)
sat_z_f  = sat_z.astype(float)
axes2[2,2].fill_between(t_u, sat_xy_f, step='mid', color=C_ACT, alpha=0.6, label='τx/τy sat')
axes2[2,2].fill_between(t_u, -sat_z_f, step='mid', color=C_REF, alpha=0.6, label='τz sat')
axes2[2,2].axhline(0, color='k', lw=0.5)
axes2[2,2].set_ylim(-1.3, 1.3)
axes2[2,2].set_yticks([-1, 0, 1])
axes2[2,2].set_yticklabels(['τz sat', '0', 'τxy sat'])
sat_frac_xy = sat_xy.mean()*100
sat_frac_z  = sat_z.mean()*100
axes2[2,2].set_title(f'Saturation Timeline  // tau_max, tau_z_max\n'
                     f'τxy: {sat_frac_xy:.1f}%  τz: {sat_frac_z:.1f}%')
axes2[2,2].legend()

for ax in axes2.flat:
    ax.set_xlabel('t (s)')

fig2.tight_layout()

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Landing Cone + MPC Solver + Summary (alpha_land, W_land, W_obs)
# ════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 3, figsize=(16, 7))
fig3.suptitle(f'Constraints & Solver — {tag}', fontsize=13, fontweight='bold', y=0.98)

# ── Landing cone  (alpha_land, W_land) ───────────────────────────────────────
axes3[0,0].plot(t, land_cone, color=C_LAND, label='vz + α·z')
axes3[0,0].axhline(0, color='k', lw=1, ls='--', label='constraint = 0')
axes3[0,0].fill_between(t, land_cone, 0, where=(land_cone<0), color=C_SAT, alpha=0.3, label='violation')
axes3[0,0].set_title(f'Landing Cone  vz + α·z ≥ 0  (α={alpha_land})  // alpha_land, W_land')
axes3[0,0].set_ylabel('m/s')
axes3[0,0].legend()

axes3[0,1].plot(t, land_violation, color=C_SAT)
axes3[0,1].fill_between(t, land_violation, color=C_SAT, alpha=0.3)
axes3[0,1].set_title('Landing Cone Violation  max(−(vz+α·z), 0)  // W_land')
axes3[0,1].set_ylabel('m/s')

# Descent rate vs altitude (landing phase)
axes3[0,2].scatter(pz, vz, c=t, cmap='viridis', s=4, alpha=0.7)
cone_z = np.linspace(0, max(pz.max(), 0.5), 100)
axes3[0,2].plot(cone_z, -alpha_land*cone_z, color=C_LAND, lw=1.5, ls='--', label=f'limit: vz = −{alpha_land}·z')
axes3[0,2].axhline(0, color='k', lw=0.5)
axes3[0,2].set_xlabel('z (m)')
axes3[0,2].set_ylabel('vz (m/s)')
axes3[0,2].set_title('Phase Plot: vz vs z  // alpha_land, W_land')
axes3[0,2].legend()

# ── MPC Solver Performance ────────────────────────────────────────────────────
axes3[1,0].plot(t_u, mpc_times, color=C_ACT, lw=1)
axes3[1,0].axhline(np.mean(mpc_times), color=C_REF, lw=1.5, ls='--',
                   label=f'mean={np.mean(mpc_times):.1f}ms')
axes3[1,0].axhline(50, color=C_SAT, lw=1, ls=':', label='budget 50ms')
axes3[1,0].fill_between(t_u, mpc_times, 50, where=(mpc_times>50), color=C_SAT, alpha=0.4, label='over budget')
axes3[1,0].set_title('MPC Solve Time per Step')
axes3[1,0].set_ylabel('ms')
axes3[1,0].legend()

axes3[1,1].hist(mpc_times, bins=25, color=C_ACT, edgecolor='white', lw=0.3)
axes3[1,1].axvline(np.mean(mpc_times), color=C_REF, lw=1.5, ls='--', label=f'mean={np.mean(mpc_times):.1f}ms')
axes3[1,1].axvline(50, color=C_SAT, lw=1.5, ls=':', label='50ms budget')
axes3[1,1].set_title('Solve Time Distribution')
axes3[1,1].set_xlabel('ms')
axes3[1,1].set_ylabel('count')
axes3[1,1].legend()

# ── Obstacle Slot Usage  (W_obs) ─────────────────────────────────────────────
axes3[1,2].step(t_u, n_obs, color=C_OBS, where='post')
axes3[1,2].fill_between(t_u, n_obs, step='post', color=C_OBS, alpha=0.25)
axes3[1,2].set_ylim(-0.2, max(n_obs.max()+0.5, 1.5))
axes3[1,2].set_title('Active Obstacles  // W_obs, n_obs_max')
axes3[1,2].set_ylabel('count')

for ax in axes3.flat:
    ax.set_xlabel('t (s)')

fig3.tight_layout()

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Summary Dashboard
# ════════════════════════════════════════════════════════════════════════════
fig4 = plt.figure(figsize=(16, 5))
fig4.suptitle(f'Summary — {tag}', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(1, 4, figure=fig4, wspace=0.35)

# RMS table
rms_pos  = np.sqrt(np.mean(pos_err**2))
rms_vel  = np.sqrt(np.mean(vel_err**2))
rms_att  = np.sqrt(np.mean(att_err_deg**2))
rms_omg  = np.sqrt(np.mean(omega_err**2))
rms_acc  = np.sqrt(np.mean(acc_err**2))

ax_txt = fig4.add_subplot(gs[0])
ax_txt.axis('off')
metrics = [
    ("RMS pos error",      f"{rms_pos:.4f} m",     "Q_pos"),
    ("RMS vel error",      f"{rms_vel:.4f} m/s",   "Q_vel"),
    ("RMS acc error",      f"{rms_acc:.4f} m/s²",  "Q_vel"),
    ("RMS att error",      f"{rms_att:.2f} °",     "Q_att"),
    ("RMS omega error",    f"{rms_omg:.4f} rad/s", "Q_omega"),
    ("z_max",              f"{pz.max():.3f} m",    "—"),
    ("τ_sat (xy)",         f"{sat_xy.mean()*100:.1f}%", "tau_max"),
    ("τ_sat (z)",          f"{sat_z.mean()*100:.1f}%",  "tau_z_max"),
    ("f @ f_min",          f"{at_fmin.mean()*100:.1f}%","f_min"),
    ("MPC mean",           f"{mpc_times.mean():.1f} ms","—"),
    ("Landing violations", f"{(land_violation>1e-3).sum()}", "W_land"),
]
col_labels = ["Metrik", "Değer", "Parametre"]
table_data = [[m, v, p] for m,v,p in metrics]
tbl = ax_txt.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='left')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.35)
for (r,c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    if r == 0:
        cell.set_facecolor('#D6E4F0')
        cell.set_text_props(fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#F0F4F8')
ax_txt.set_title('Error Metrics', fontweight='bold', fontsize=9, pad=4)

# Pos error bar over time
ax_pe = fig4.add_subplot(gs[1])
ax_pe.plot(t, pos_err, color=C_ERR, lw=1)
ax_pe.set_title('‖Δp‖  // Q_pos', fontsize=9, fontweight='bold')
ax_pe.set_xlabel('t (s)'); ax_pe.set_ylabel('m')

# Attitude error bar over time
ax_ae = fig4.add_subplot(gs[2])
ax_ae.plot(t, att_err_deg, color=C_ERR, lw=1)
ax_ae.set_title('Att Error (deg)  // Q_att', fontsize=9, fontweight='bold')
ax_ae.set_xlabel('t (s)'); ax_ae.set_ylabel('deg')

# Torque usage
ax_tu = fig4.add_subplot(gs[3])
bins = np.linspace(-tau_max_val*1.05, tau_max_val*1.05, 30)
ax_tu.hist(tau_x, bins=bins, color=C_ACT, alpha=0.6, label='τx', edgecolor='none')
ax_tu.hist(tau_y, bins=bins, color=C_REF, alpha=0.6, label='τy', edgecolor='none')
ax_tu.axvline( tau_max_val, color=C_SAT, lw=1.5, ls=':', label='limit')
ax_tu.axvline(-tau_max_val, color=C_SAT, lw=1.5, ls=':')
ax_tu.set_title('Torque Distribution  // R_tau, tau_max', fontsize=9, fontweight='bold')
ax_tu.set_xlabel('Nm')
ax_tu.legend()

fig4.tight_layout()

# ── Kaydet veya Göster ────────────────────────────────────────────────────────
if args.save:
    base = log_path.replace('.npz', '')
    for i, fig in enumerate([fig1, fig2, fig3, fig4], 1):
        fname = f'{base}_plot{i}.png'
        fig.savefig(fname, dpi=args.dpi, bbox_inches='tight')
        print(f'[plot] Kaydedildi: {fname}')
else:
    plt.show()
