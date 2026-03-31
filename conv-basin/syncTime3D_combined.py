"""
Combined 3D + convergence animation for the D=3 hybrid oscillator.
Layout:
  Top-left:  3D view of trajectories in x-coords [-pi,pi)^3
  Top-right: x_i(t) - circular mean over time — CONVERGENCE IS DIRECTLY VISIBLE
             as all three lines approach 0
  Bottom-left:  x_1 vs x_2  (pairwise synchronization)
  Bottom-right: x_1 vs x_3  (pairwise synchronization)
Convergence = lines in top-right panel approach 0,
              dots in bottom panels approach the diagonal.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hybrid_tools import torus_spread_x

# ── Model ─────────────────────────────────────────────────────────
DELTA       = 0.5
D           = 3
PI          = np.pi
COORD_GUARD = 0.5
SUM_GUARD_U = (1.0 + D * PI) / (2.0 * PI)
U_STAR      = SUM_GUARD_U / D

# ── CONFIG ────────────────────────────────────────────────────────
FPS       = 60
T_END     = 80.0
TRAIL_LEN = 0.15   # torus path length in 3D trail

x_star = np.array([1/3, 1/3, 1/3])
r_max  = 1.0 / np.sqrt(D * (D-1))
v1     = np.array([ 1, -1,  0]) / np.sqrt(2)
v2     = np.array([ 1,  1, -2]) / np.sqrt(6)

# Initial conditions in x coords — all start ON sum guard inside S_D
INITIALS_X = [
    x_star,
    x_star + 0.3 * r_max * v1,
    x_star + 0.7 * r_max * v1,
    x_star + 0.3 * r_max * v1 + 0.3 * r_max * v2,
]
INITIALS_U  = [(x + PI) / (2*PI) for x in INITIALS_X]
COLORS      = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
LABELS      = [f'P{i+1}  $\\|v\\|_T={torus_spread_x(x):.3f}$'
               for i, x in enumerate(INITIALS_X)]


class OdePC:
    def __init__(self, fun):
        self._fun = fun

    def __call__(self, y0, t0, t1=None, dt=None, tTol=1e-6,
                 pars=None, allclose=np.allclose, withDy=False):
        ts  = list(np.arange(t0, t1, dt))
        y   = [np.asarray(y0, dtype=float)]
        t   = [ts.pop(0)]
        h   = min(dt, ts[0] - t0)
        dy0 = [self._fun(t[0], y[0], pars)]
        while ts:
            th = t[-1] + h
            yh = y[-1] + h * dy0[-1]
            dy = self._fun(th, yh, pars)
            if not allclose(dy, dy0[-1]) and h > tTol:
                h /= 2.0
                continue
            if th < ts[0]:
                h0 = h
            else:
                th  = ts.pop(0)
                h0  = th - t[-1]
                yh  = y[-1] + h0 * dy0[-1]
                dy1 = self._fun(th, yh, pars)
                if h0 > tTol and not allclose(dy1, dy):
                    h0  = tTol
                    th  = t[-1] + h0
                    yh  = y[-1] + h0 * dy0[-1]
                    dy1 = self._fun(th, yh, pars)
                dy = dy1
            dy0.append(dy)
            y.append(y[-1] + dy * h0)
            t.append(th)
            h = min(h * 1.5, dt)
        if withDy:
            return np.asarray(t), np.asarray(y), np.asarray(dy0)
        return np.asarray(t), np.asarray(y)


def rhs_3d(t, u, pars=None):
    delta = (pars or {}).get('delta', DELTA)
    u_w   = np.mod(u, 1.0)
    fast  = 1.0 / (2.0 * PI)
    slow  = (1.0 - delta) / (2.0 * PI)
    return np.where((u_w < COORD_GUARD) | (u_w.sum() > SUM_GUARD_U), fast, slow)


def wrap_signed(u):
    return (u + 0.5) % 1.0 - 0.5

def cumlen_torus(A):
    if len(A) <= 1:
        return np.array([0.0])
    d = np.diff(A, axis=0)
    d = (d + 0.5) % 1.0 - 0.5
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(d, axis=1))))

def split_at_tile_crossings(pu_tail, pw_tail):
    """Same as syncTime4D.py."""
    tiles   = np.floor(pu_tail)
    crosses = np.any(np.diff(tiles, axis=0) != 0, axis=1)
    x, y    = pw_tail[:, 0].copy(), pw_tail[:, 1].copy()
    if np.any(crosses):
        for j in np.where(crosses)[0][::-1]:
            x = np.insert(x, j + 1, np.nan)
            y = np.insert(y, j + 1, np.nan)
    return x, y

def get_segments(raw_seg, u_seg):
    """Split 3D trail at wrap points."""
    if len(raw_seg) < 2:
        return [raw_seg]
    tiles    = np.floor((u_seg + PI) / (2.0 * PI)).astype(int)
    jump_idx = np.where(np.any(np.diff(tiles, axis=0) != 0, axis=1))[0]
    if len(jump_idx) == 0:
        return [raw_seg]
    cuts = np.concatenate([[0], jump_idx + 1, [len(raw_seg)]])
    return [raw_seg[cuts[k]:cuts[k+1]] for k in range(len(cuts)-1)
            if cuts[k+1] > cuts[k]]


# ── Integrate ─────────────────────────────────────────────────────
print("Integrating with OdePC...")
ode  = OdePC(rhs_3d)
pars = dict(delta=DELTA)
raw_trajs, raw_times = [], []
for u0, x0 in zip(INITIALS_U, INITIALS_X):
    t, y, _ = ode(u0, t0=0.0, t1=T_END, dt=0.001, tTol=1e-6, pars=pars, withDy=True)
    raw_trajs.append(y)
    raw_times.append(t)
    v_norm = torus_spread_x(x0)
    print(f"  {np.round(u0,3)}  ||v||={v_norm:.4f}  pts={len(t)}")

# ── Interpolate ───────────────────────────────────────────────────
n_frames  = int(T_END * FPS)
uniform_t = np.linspace(0, T_END, n_frames)

interp_u, unwrap_u = [], []
for i in range(len(INITIALS_U)):
    funcs = [interp1d(raw_times[i], raw_trajs[i][:, d], kind='linear',
                      bounds_error=False, fill_value='extrapolate')
             for d in range(D)]
    T = np.column_stack([f(uniform_t) for f in funcs])
    theta    = 2.0 * PI * np.mod(T, 1.0)
    theta_un = np.unwrap(theta, axis=0)
    interp_u.append(np.mod(T, 1.0))
    unwrap_u.append(theta_un / (2.0 * PI))

interp_x = [2.0 * PI * U - PI for U in interp_u]
unwrap_x  = [2.0 * PI * U - PI for U in unwrap_u]
cumlen    = [cumlen_torus(W) for W in interp_u]

# Transverse component: v_i(t) = x_i(t) - circular mean  (torus-correct)
# Use circular mean in u coords * 2pi for proper wrapping
def circ_mean_u(u_row):
    theta = 2*PI*u_row
    return np.arctan2(np.sin(theta).mean(), np.cos(theta).mean()) % (2*PI)

transverse = []  # v_i(t) = x_i(t) - x_mean_circular(t)
for i in range(len(INITIALS_U)):
    U   = interp_u[i]
    phi = np.array([circ_mean_u(U[k]) for k in range(n_frames)])   # circular mean angle
    phi_u = phi / (2*PI)
    phi_x = 2*PI*phi_u - PI
    x_i   = interp_x[i]
    v_i   = x_i - phi_x[:, None]   # shape (n_frames, 3)
    # wrap v_i to [-pi, pi] for display
    v_i   = ((v_i + PI) % (2*PI)) - PI
    transverse.append(v_i)


# ── Figure layout ─────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

ax3d   = fig.add_subplot(gs[0, 0], projection='3d')
ax_v   = fig.add_subplot(gs[0, 1])   # transverse components vs time
ax_12  = fig.add_subplot(gs[1, 0])   # x1 vs x2
ax_13  = fig.add_subplot(gs[1, 1])   # x1 vs x3

# ── 3D panel ──────────────────────────────────────────────────────
ax3d.set_xlim(-PI, PI); ax3d.set_ylim(-PI, PI); ax3d.set_zlim(-PI, PI)
ax3d.set_xlabel('$x_1$', fontsize=9); ax3d.set_ylabel('$x_2$', fontsize=9)
ax3d.set_zlabel('$x_3$', fontsize=9)
ax3d.set_title('3D view  ($x$ coordinates)', fontsize=10)
tri = Poly3DCollection([[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]], alpha=0.18)
tri.set_facecolor('royalblue'); tri.set_edgecolor('royalblue')
ax3d.add_collection3d(tri)
lo, hi = -PI, PI
for col, plane in zip(['darkorange','purple','saddlebrown'],
                      [([0,0,0,0],[lo,lo,hi,hi],[lo,hi,hi,lo]),
                       ([lo,lo,hi,hi],[0,0,0,0],[lo,hi,hi,lo]),
                       ([lo,lo,hi,hi],[lo,hi,hi,lo],[0,0,0,0])]):
    sq = Poly3DCollection([list(zip(*plane))], alpha=0.06)
    sq.set_facecolor(col); sq.set_edgecolor(col); ax3d.add_collection3d(sq)
t_d = np.linspace(-PI, PI/3+0.2, 100)
ax3d.plot(t_d, t_d, t_d, 'k-', lw=2.5)
ax3d.scatter([1/3],[1/3],[1/3], color='black', s=50, zorder=10)
for x0, col in zip(INITIALS_X, COLORS):
    ax3d.scatter(*x0, color=col, s=60, marker='^', zorder=12)

# ── Transverse panel ──────────────────────────────────────────────
ax_v.set_xlim(0, T_END); ax_v.set_ylim(-PI/2, PI/2)
ax_v.axhline(0, color='black', lw=1.5, ls='-', alpha=0.4)
ax_v.set_xlabel('$t$', fontsize=10)
ax_v.set_ylabel('$x_i - \\bar{x}$', fontsize=10)
ax_v.set_title('Transverse component $x_i - \\bar{x}$\n'
               'Convergence = all lines → 0', fontsize=9)
ax_v.grid(True, alpha=0.3)

# ── Pairwise panels ───────────────────────────────────────────────
for ax, xl, yl, title in [
        (ax_12, '$x_1$', '$x_2$', '$x_1$ vs $x_2$\n(sync = approach diagonal)'),
        (ax_13, '$x_1$', '$x_3$', '$x_1$ vs $x_3$')]:
    ax.set_xlim(-PI, PI); ax.set_ylim(-PI, PI)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel(xl, fontsize=10); ax.set_ylabel(yl, fontsize=10)
    ax.set_title(title, fontsize=9)
    ax.plot([-PI, PI], [-PI, PI], 'k-', lw=1.2, alpha=0.4)
    ax.axvline(0, color='orange', lw=0.8, ls='--', alpha=0.4)
    ax.axhline(0, color='purple', lw=0.8, ls='--', alpha=0.4)
    ax.scatter([1/3],[1/3], s=40, c='black', marker='*', zorder=5)
    for x0, col in zip(INITIALS_X, COLORS):
        ax.scatter([x0[0]], [x0[1 if ax is ax_12 else 2]],
                   color=col, s=50, marker='^', zorder=6)

# ── Artists ───────────────────────────────────────────────────────
MAX_SEGS    = 8
seg_lines_3d = [[ax3d.plot([],[],[], color=COLORS[i], lw=1.8, alpha=0.85)[0]
                 for _ in range(MAX_SEGS)]
                for i in range(len(INITIALS_X))]
heads_3d     = [ax3d.plot([],[],[], 'o', color=COLORS[i], ms=9, zorder=20)[0]
                for i in range(len(INITIALS_X))]

# Transverse panel: 3 lines per agent (one per coordinate)
COORD_STYLES = ['-', '--', ':']
COORD_NAMES  = ['$x_1$', '$x_2$', '$x_3$']
v_lines = [[ax_v.plot([],[], ls=COORD_STYLES[d],
                       color=COLORS[i], lw=1.5, alpha=0.85,
                       animated=True)[0]
            for d in range(D)]
           for i in range(len(INITIALS_X))]
v_heads = [ax_v.plot([],[], 'o', color=COLORS[i], ms=5,
                      animated=True, zorder=4)[0]
           for i in range(len(INITIALS_X))]

# Pairwise panels
trails_12 = [ax_12.plot([],[],'-', color=COLORS[i], lw=1.8, alpha=0.8,
                          animated=True, zorder=3)[0]
             for i in range(len(INITIALS_X))]
heads_12  = [ax_12.plot([],[],'o', color=COLORS[i], ms=7,
                          animated=True, zorder=4)[0]
             for i in range(len(INITIALS_X))]
trails_13 = [ax_13.plot([],[],'-', color=COLORS[i], lw=1.8, alpha=0.8,
                          animated=True, zorder=3)[0]
             for i in range(len(INITIALS_X))]
heads_13  = [ax_13.plot([],[],'o', color=COLORS[i], ms=7,
                          animated=True, zorder=4)[0]
             for i in range(len(INITIALS_X))]

# Legend
handles = [plt.Line2D([0],[0],color=COLORS[i],lw=2) for i in range(len(INITIALS_X))]
fig.legend(handles, LABELS, loc='lower center', ncol=4,
           fontsize=8, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=[0, 0.04, 1, 1])

TAIL_FRAMES = int(FPS * 12)   # 12 seconds of trail in 3D
TAIL_T      = 15.0             # seconds of transverse history shown

def update(frame):
    t_now = uniform_t[frame]
    art   = []

    for i in range(len(INITIALS_X)):
        cl        = cumlen[i][:frame + 1]
        start_3d  = np.searchsorted(cl, max(0.0, cl[-1] - TRAIL_LEN), 'left')

        # ── 3D ────────────────────────────────────────────────────
        raw_tail = interp_x[i][start_3d:frame+1]
        u_tail   = unwrap_x[i][start_3d:frame+1]
        heads_3d[i].set_data([interp_x[i][frame,0]], [interp_x[i][frame,1]])
        heads_3d[i].set_3d_properties([interp_x[i][frame,2]])
        parts = get_segments(raw_tail, u_tail)
        for k, ln in enumerate(seg_lines_3d[i]):
            if k < len(parts) and len(parts[k]) > 1:
                p = parts[k]
                ln.set_data(p[:,0], p[:,1]); ln.set_3d_properties(p[:,2])
            else:
                ln.set_data([], []); ln.set_3d_properties([])
        art += seg_lines_3d[i] + [heads_3d[i]]

        # ── Transverse panel ──────────────────────────────────────
        t_start_v = max(0, t_now - TAIL_T)
        f_start_v = max(0, frame - int(TAIL_T * FPS))
        t_seg = uniform_t[f_start_v:frame+1]
        v_seg = transverse[i][f_start_v:frame+1]   # (T, 3)
        for d in range(D):
            v_lines[i][d].set_data(t_seg, v_seg[:, d])
        v_heads[i].set_data([t_now], [transverse[i][frame, 0]])
        art += v_lines[i] + [v_heads[i]]
        # update x-axis window
        ax_v.set_xlim(max(0, t_now - TAIL_T), max(TAIL_T, t_now))

        # ── Pairwise x1-x2 ────────────────────────────────────────
        pu_12 = unwrap_u[i][start_3d:frame+1][:, [0,1]]
        pw_12 = interp_u[i][start_3d:frame+1][:, [0,1]]
        x12_u, y12_u = split_at_tile_crossings(pu_12, pw_12)
        # convert to x coords: x = 2*pi*u - pi
        trails_12[i].set_data(2*PI*x12_u - PI, 2*PI*y12_u - PI)
        h12 = interp_x[i][frame]
        heads_12[i].set_data([h12[0]], [h12[1]])
        art += [trails_12[i], heads_12[i]]

        # ── Pairwise x1-x3 ────────────────────────────────────────
        pu_13 = unwrap_u[i][start_3d:frame+1][:, [0,2]]
        pw_13 = interp_u[i][start_3d:frame+1][:, [0,2]]
        x13_u, y13_u = split_at_tile_crossings(pu_13, pw_13)
        trails_13[i].set_data(2*PI*x13_u - PI, 2*PI*y13_u - PI)
        heads_13[i].set_data([h12[0]], [h12[2]])
        art += [trails_13[i], heads_13[i]]

    fig.suptitle(
        f'3D Hybrid Oscillator  |  $\\delta={DELTA}$  |  $t={t_now:.1f}$\n'
        f'Top-right: convergence is visible as $x_i - \\bar{{x}} \\to 0$',
        fontsize=11)
    return art


print(f"Animating {n_frames} frames at {FPS} fps...")
anim = FuncAnimation(fig, update, frames=n_frames,
                     interval=1000 / FPS, blit=False, repeat=True)
anim.save('basin_3d_combined.mp4', writer='ffmpeg', fps=FPS, bitrate=2000)
print("Saved basin_3d_combined.mp4")
