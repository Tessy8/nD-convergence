"""
3D basin of attraction animation using OdePC (same integrator as syncTime3D.py).
Shows trajectories moving through the 3D torus box [-pi, pi)^3 with
accurate event detection at every switching surface crossing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Model ─────────────────────────────────────────────────────────
DELTA       = 0.5
D           = 3
PI          = np.pi
COORD_GUARD = 0.5                              # in u: u_i < 0.5 => fast (x_i < 0)
SUM_GUARD_U = (1.0 + D * PI) / (2.0 * PI)    # sum(u) > this => fast
U_STAR      = SUM_GUARD_U / D                 # diagonal meets sum guard
FAST_U      = 1.0 / (2.0 * PI)
SLOW_U      = (1.0 - DELTA) / (2.0 * PI)

# ── CONFIG ────────────────────────────────────────────────────────
FPS         = 60
T_END       = 60.0
TRAIL_LEN   = 0.15    # cumulative torus path length shown in trail

INITIALS = [
    np.array([U_STAR, U_STAR, U_STAR]),        # on diagonal at x*
    np.array([0.60,   0.58,   0.53  ]),        # close — converges
    np.array([0.45,   0.63,   0.68  ]),        # mixed region
    np.array([0.20,   0.72,   0.80  ]),        # far from diagonal
]
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']


class OdePC:
    def __init__(self, fun):
        self._fun = fun

    def __call__(self, y0, t0, t1=None, dt=None, tTol=1e-6,
                 pars=None, allclose=np.allclose, withDy=False):
        if t1 is None:
            ts = list(t0)
            assert len(ts) > 1 and all(np.diff(ts) > 0)
        else:
            assert dt is not None
            ts = list(np.arange(t0, t1, dt))
        y   = [np.asarray(y0, dtype=float)]
        t   = [ts.pop(0)]
        if dt is None:
            dt = ts[-1]
        h   = min(dt, ts[0] - t0)
        dy0 = [self._fun(t[0], y[0], pars)]
        while ts:
            th  = t[-1] + h
            yh  = y[-1] + h * dy0[-1]
            dy  = self._fun(th, yh, pars)
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
            return (np.asarray(t), np.asarray(y), np.asarray(dy0))
        return (np.asarray(t), np.asarray(y))


def rhs_3d(t, u, pars=None):
    delta = (pars or {}).get('delta', DELTA)
    fast  = 1.0 / (2.0 * PI)
    slow  = (1.0 - delta) / (2.0 * PI)
    u_w   = np.mod(u, 1.0)
    return np.where((u_w < COORD_GUARD) | (u_w.sum() > SUM_GUARD_U), fast, slow)


def wrap_signed(u):
    return (u + 0.5) % 1.0 - 0.5

def cumlen_torus(A):
    if len(A) <= 1:
        return np.array([0.0])
    d   = np.diff(A, axis=0)
    d   = (d + 0.5) % 1.0 - 0.5
    seg = np.linalg.norm(d, axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))

def get_segments(raw_seg, u_seg):
    """Split raw_seg at torus wraps (tile crossings in unwrapped coords)."""
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
for u0 in INITIALS:
    t, y, _ = ode(u0, t0=0.0, t1=T_END, dt=0.001, tTol=1e-6,
                  pars=pars, withDy=True)
    raw_trajs.append(y)
    raw_times.append(t)
    print(f"  {np.round(u0,3)}  pts={len(t)}  "
          f"spread={np.abs(y[-1]-y[-1].mean()).max():.4f}")

# ── Interpolate to uniform time grid ──────────────────────────────
n_frames    = int(T_END * FPS)
uniform_t   = np.linspace(0, T_END, n_frames)

interp_u = []   # wrapped [0,1)
unwrap_u = []   # unwrapped (for segment detection)
for i in range(len(INITIALS)):
    funcs = [interp1d(raw_times[i], raw_trajs[i][:, d],
                      kind='linear', bounds_error=False,
                      fill_value='extrapolate')
             for d in range(D)]
    T = np.column_stack([f(uniform_t) for f in funcs])
    # unwrap in torus (theta = 2pi*u)
    theta    = 2.0 * PI * np.mod(T, 1.0)
    theta_un = np.unwrap(theta, axis=0)
    u_un     = theta_un / (2.0 * PI)
    interp_u.append(np.mod(T, 1.0))
    unwrap_u.append(u_un)

cumlen = [cumlen_torus(W) for W in interp_u]

# Convert from u in [0,1) to x in [-pi, pi)
def u_to_x(u):
    return 2.0 * PI * u - PI

interp_x = [u_to_x(U) for U in interp_u]
unwrap_x  = [u_to_x(U) for U in unwrap_u]


# ── Figure ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 9))
ax  = fig.add_subplot(111, projection='3d')
ax.set_xlim(-PI, PI); ax.set_ylim(-PI, PI); ax.set_zlim(-PI, PI)
ax.set_xlabel('$x_1$', fontsize=11); ax.set_ylabel('$x_2$', fontsize=11)
ax.set_zlabel('$x_3$', fontsize=11)

# Sum guard triangle (x1+x2+x3=1, xi>=0) in x-coords
tri_verts = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
tri = Poly3DCollection([tri_verts], alpha=0.18)
tri.set_facecolor('royalblue'); tri.set_edgecolor('royalblue')
ax.add_collection3d(tri)

# Coordinate guard planes (x_i = 0) as faint squares
lo, hi = -PI, PI
for col, plane in zip(['darkorange', 'purple', 'saddlebrown'],
                      [([0,0,0,0],[lo,lo,hi,hi],[lo,hi,hi,lo]),
                       ([lo,lo,hi,hi],[0,0,0,0],[lo,hi,hi,lo]),
                       ([lo,lo,hi,hi],[lo,hi,hi,lo],[0,0,0,0])]):
    sq = Poly3DCollection([list(zip(*plane))], alpha=0.07)
    sq.set_facecolor(col); sq.set_edgecolor(col)
    ax.add_collection3d(sq)

# Diagonal
t_d = np.linspace(-PI, PI/3 + 0.1, 100)
ax.plot(t_d, t_d, t_d, 'k-', lw=2.5, label='Diagonal')
# x* = (1/3, 1/3, 1/3)
ax.scatter([1/3], [1/3], [1/3], color='black', s=70, zorder=10)

# Starting markers
for i, (u0, col) in enumerate(zip(INITIALS, COLORS)):
    x0 = u_to_x(u0)
    ax.scatter(*x0, color=col, s=70, marker='^', zorder=12,
               label=f'P{i+1} {np.round(u0,2)}')

# ── Trail artists: MAX_SEGS separate Line3D per trajectory ────────
MAX_SEGS  = 8
seg_lines = []
heads     = []
for i in range(len(INITIALS)):
    col = COLORS[i]
    segs = [ax.plot([], [], [], color=col, lw=2.0, alpha=0.85)[0]
            for _ in range(MAX_SEGS)]
    seg_lines.append(segs)
    head, = ax.plot([], [], [], 'o', color=col, ms=9, zorder=20)
    heads.append(head)


def update(frame):
    all_art = []
    for i in range(len(INITIALS)):
        # find start of trail using cumulative path length
        cl        = cumlen[i][:frame + 1]
        start_idx = np.searchsorted(cl, max(0.0, cl[-1] - TRAIL_LEN), 'left')

        raw_tail = interp_x[i][start_idx:frame + 1]
        u_tail   = unwrap_x[i][start_idx:frame + 1]   # unwrapped x for detection

        # head at current position
        heads[i].set_data([interp_x[i][frame, 0]], [interp_x[i][frame, 1]])
        heads[i].set_3d_properties([interp_x[i][frame, 2]])

        # split into continuous segments at wrap points
        parts = get_segments(raw_tail, u_tail)
        for k, ln in enumerate(seg_lines[i]):
            if k < len(parts) and len(parts[k]) > 1:
                p = parts[k]
                ln.set_data(p[:, 0], p[:, 1])
                ln.set_3d_properties(p[:, 2])
            else:
                ln.set_data([], [])
                ln.set_3d_properties([])
        all_art += seg_lines[i] + [heads[i]]

    ax.set_title(f'3D Hybrid Oscillator  |  $\\delta={DELTA}$  |  '
                 f'$t={uniform_t[frame]:.1f}$', fontsize=12)
    return all_art


ax.legend(fontsize=8, loc='upper left')
plt.tight_layout()

print(f"Animating {n_frames} frames...")
anim = FuncAnimation(fig, update, frames=n_frames,
                     interval=1000 / FPS, blit=False, repeat=True)
anim.save('basin_3d_animation.mp4', writer='ffmpeg', fps=FPS, bitrate=2000)
print("Saved basin_3d_animation.mp4")
