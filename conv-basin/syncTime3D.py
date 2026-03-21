import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

# ── Model parameters ──────────────────────────────────────────────────────────
# Model: x_i = 2*pi*u_i - pi  in [-pi, pi)
# x_i is FAST if x_i < 0  OR  sum(x) > 1
# In u coords [0,1):
#   x_i < 0        <=>  u_i < 0.5
#   sum(x) > 1     <=>  sum(u) > (1 + D*pi) / (2*pi)
DELTA         = 0.5
D             = 3
COORD_GUARD   = 0.5                              # u_i < this => fast
SUM_GUARD_U   = (1.0 + D * np.pi) / (2.0 * np.pi)   # ≈ 1.659 for D=3
FAST_SPEED    = 1.0 / (2.0 * np.pi)
SLOW_SPEED    = (1.0 - DELTA) / (2.0 * np.pi)

# u* = where diagonal meets sum guard (in u coords)
U_STAR        = SUM_GUARD_U / D                  # ≈ 0.553


class OdePC:
    """Predictor-corrector ODE solver with automatic event detection.
    Identical to OdePC in syncTime4D.py."""
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
            return (np.asarray(t, dtype=float),
                    np.asarray(y, dtype=float),
                    np.asarray(dy0, dtype=float))
        return (np.asarray(t, dtype=float),
                np.asarray(y, dtype=float))


def rhs_3d(t, u, pars=None):
    """
    Vector field for the D=3 hybrid model in u in [0,1)^3.

    Dynamics (in x = 2*pi*u - pi coordinates):
      x_dot_i = 1         if x_i < 0  OR  sum(x) > 1
      x_dot_i = 1 - delta  otherwise

    Translating to u = (x + pi) / (2*pi):
      u_dot_i = 1/(2*pi)         if u_i < 0.5  OR  sum(u) > SUM_GUARD_U
      u_dot_i = (1-delta)/(2*pi)  otherwise
    """
    delta = pars.get('delta', DELTA) if pars else DELTA
    fast  = 1.0 / (2.0 * np.pi)
    slow  = (1.0 - delta) / (2.0 * np.pi)
    u_w   = np.mod(u, 1.0)          # wrap before checking condition
    u_sum = u_w.sum()
    return np.where((u_w < COORD_GUARD) | (u_sum > SUM_GUARD_U), fast, slow)


def wrap_signed(u):
    return (u + 0.5) % 1.0 - 0.5

def torus_diff_nd(A):
    return wrap_signed(np.diff(A, axis=0))

def cumlen_torus_nd(A):
    if len(A) <= 1:
        return np.array([0.0], float)
    seg = np.linalg.norm(torus_diff_nd(A), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))

def split_at_tile_crossings(pu_tail, pw_tail):
    """
    Identical logic to syncTime4D.py:
    pu_tail: (m,2) unwrapped projected coords
    pw_tail: (m,2) wrapped projected coords in [0,1)^2
    Returns x,y with NaNs at torus crossings.
    """
    tiles   = np.floor(pu_tail)
    crosses = np.any(np.diff(tiles, axis=0) != 0, axis=1)
    x, y    = pw_tail[:, 0].copy(), pw_tail[:, 1].copy()
    if np.any(crosses):
        for j in np.where(crosses)[0][::-1]:
            x = np.insert(x, j + 1, np.nan)
            y = np.insert(y, j + 1, np.nan)
    return x, y


def create_continuous_trajectories():
    """Integrate the D=3 hybrid model for several agents using OdePC."""
    # Initial conditions in u in [0,1)^3
    # u* ≈ 0.553 is where the diagonal meets the sum guard
    initials = [
        np.array([0.553, 0.553, 0.553]),    # exactly on diagonal at x*
        np.array([0.60,  0.58,  0.53 ]),    # close to diagonal — converges
        np.array([0.45,  0.63,  0.68 ]),    # one coord below guard — mixed region
        np.array([0.20,  0.72,  0.80 ]),    # far from diagonal
    ]
    ode  = OdePC(rhs_3d)
    pars = dict(delta=DELTA)

    trajs, times = [], []
    for u0 in initials:
        t, y, _ = ode(u0, t0=0.0, t1=60.0, dt=0.001, tTol=1e-6,
                      pars=pars, withDy=True)
        trajs.append(y)
        times.append(t)
    return trajs, times


def animate_3d_agents():
    trajs, times = create_continuous_trajectories()
    n_agents     = len(trajs)

    total_time  = max(times[i][-1] for i in range(n_agents))
    fps         = 60
    n_frames    = int(total_time * fps)
    uniform_t   = np.linspace(0, total_time, n_frames)

    # ── Interpolate to uniform time ───────────────────────────────
    interp_trajs = []
    for i in range(n_agents):
        funcs = [interp1d(times[i], trajs[i][:, d], kind='linear',
                          bounds_error=False, fill_value='extrapolate')
                 for d in range(D)]
        interp_trajs.append(np.column_stack([f(uniform_t) for f in funcs]))

    # ── Unwrap each dimension (same as syncTime4D.py) ─────────────
    unwrapped_trajs = []
    for T in interp_trajs:
        theta    = 2.0 * np.pi * np.mod(T, 1.0)
        theta_un = np.unwrap(theta, axis=0)
        unwrapped_trajs.append(theta_un / (2.0 * np.pi))   # cycles, unwrapped

    wrapped_trajs = [U - np.floor(U) for U in unwrapped_trajs]   # back to [0,1)

    # Cumulative path length on the 3D torus
    cumlen = [cumlen_torus_nd(W) for W in wrapped_trajs]

    # ── Figure: 3 panels — one per coordinate pair ───────────────
    # Mirrors syncTime4D.py's 6-panel layout, reduced to 3 for 3D.
    pairs  = [(0, 1), (0, 2), (1, 2)]
    titles = ['$u_1$–$u_2$  ($x_1$–$x_2$)',
              '$u_1$–$u_3$  ($x_1$–$x_3$)',
              '$u_2$–$u_3$  ($x_2$–$x_3$)']
    xlbls  = ['$u_1$', '$u_1$', '$u_2$']
    ylbls  = ['$u_2$', '$u_3$', '$u_3$']

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.tight_layout(pad=3.0)

    for ax, title, xl, yl in zip(axs, titles, xlbls, ylbls):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
        ax.set_title(title, fontsize=11)
        # coordinate guards (u_i = 0.5 lines)
        ax.axvline(COORD_GUARD, color='orange', lw=1.2, alpha=0.55,
                   ls='--', label='coord guard')
        ax.axhline(COORD_GUARD, color='purple', lw=1.2, alpha=0.55, ls='--')
        # sum guard projected: u_i + u_j = SUM_GUARD_U - u_k
        # at u_k = U_STAR the boundary is u_i + u_j = SUM_GUARD_U - U_STAR ≈ 1.106
        sg = SUM_GUARD_U - U_STAR
        ax.plot([max(0, sg-1), min(1, sg)],
                [min(1, sg), max(0, sg-1)],
                color='steelblue', lw=1.4, alpha=0.55, ls='--',
                label='sum guard (at $u^*$)')
        # diagonal u_i = u_j
        ax.plot([0, 1], [0, 1], 'k-', lw=1.0, alpha=0.3)
        # x* projection
        ax.scatter([U_STAR], [U_STAR], s=60, c='black', marker='*', zorder=5)

    # ── Trail and head artists ─────────────────────────────────────
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    trails = [[ax.plot([], [], '-', color=colors[k], alpha=0.75,
                       linewidth=2, animated=True, zorder=3)[0]
               for ax in axs]
              for k in range(n_agents)]
    points = [[ax.plot([], [], 'o', color=colors[k], markersize=6,
                       animated=True, zorder=4)[0]
               for ax in axs]
              for k in range(n_agents)]

    def init():
        art = []
        for k in range(n_agents):
            for p in range(3):
                trails[k][p].set_data([], [])
                points[k][p].set_data([], [])
                art += [trails[k][p], points[k][p]]
        return art

    def update(frame):
        trail_len = 0.12   # cumulative torus path length shown in trail
        art = []
        for k in range(n_agents):
            cl        = cumlen[k][:frame + 1]
            start_idx = np.searchsorted(cl, max(0.0, cl[-1] - trail_len), 'left')

            for p, (i, j) in enumerate(pairs):
                # 2D projection of unwrapped and wrapped trajectories
                pu_tail = unwrapped_trajs[k][start_idx:frame + 1, :][:, [i, j]]
                pw_tail = wrapped_trajs[k][start_idx:frame + 1, :][:, [i, j]]

                # break line at torus wrap crossings (same as syncTime4D.py)
                x, y = split_at_tile_crossings(pu_tail, pw_tail)
                trails[k][p].set_data(x, y)

                # head: current wrapped position
                head = wrapped_trajs[k][frame]
                points[k][p].set_data([head[i]], [head[j]])
                art += [trails[k][p], points[k][p]]

        fig.suptitle(
            f'3D Hybrid Oscillator  |  $\\delta={DELTA}$  |  '
            f'$t={uniform_t[frame]:.1f}$',
            fontsize=13)
        return art

    anim = FuncAnimation(fig, update, frames=n_frames,
                         init_func=init, interval=1000 / fps,
                         blit=True, repeat=True)
    anim.save('basin_3d_animation.mp4', writer='ffmpeg', fps=fps, bitrate=2000)
    print("Saved basin_3d_animation.mp4")
    return anim


if __name__ == '__main__':
    animate_3d_agents()