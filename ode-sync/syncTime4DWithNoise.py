import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from integro_sde.sde import SDE
from statsmodels.graphics.gofplots import qqplot_2samples

lower_boundary = 0.3
upper_boundary = 0.7

AXIS_LABELS = ['w', 'x', 'y', 'z']
AXIS_COLORS = {'w': 'blue', 'x': 'red', 'y': 'green', 'z': 'orange'}

FAST        = 1.00
SLOW        = 0.30
BAND_CENTER = 0.50
BAND_HALF   = 0.15


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

        y = [np.asarray(y0, dtype=float)]
        t = [ts.pop(0)]

        if dt is None:
            dt = ts[-1]

        h = min(dt, ts[0] - t0)
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
                th = ts.pop(0)
                h0 = th - t[-1]
                yh = y[-1] + h0 * dy0[-1]
                dy1 = self._fun(th, yh, pars)
                if h0 > tTol and not allclose(dy1, dy):
                    h0 = tTol
                    th = t[-1] + h0
                    yh = y[-1] + h0 * dy0[-1]
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
        else:
            return (np.asarray(t, dtype=float),
                    np.asarray(y, dtype=float))

def wrap_signed(u):
    """Map real u to [-0.5, 0.5)"""
    return (u + 0.5) % 1.0 - 0.5

def circular_mean(u):
    """Mean on the unit circle; returns value in [0,1)."""
    theta = 2*np.pi*np.mod(u, 1.0)
    c, s = np.mean(np.cos(theta)), np.mean(np.sin(theta))
    ang = np.arctan2(s, c) % (2*np.pi)
    return ang / (2*np.pi)

def torus_diff_nd(A):
    """Forward differences"""
    d = np.diff(A, axis=0)
    return wrap_signed(d)

def cumlen_torus_nd(A):
    """Cumulative path length"""
    if len(A) <= 1: return np.array([0.0], float)
    seg = np.linalg.norm(torus_diff_nd(A), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))

def segment_wrap_jumps(x, y, jump=0.5):
    """Insert NaNs to avoid weird lines"""
    ddx, ddy = np.diff(x), np.diff(y)
    jumps = (np.abs(ddx) > jump) | (np.abs(ddy) > jump)
    if not np.any(jumps): return x, y
    x2, y2 = x.copy(), y.copy()
    for j in np.where(jumps)[0][::-1]:
        x2 = np.insert(x2, j+1, np.nan)
        y2 = np.insert(y2, j+1, np.nan)
    return x2, y2

def rhs_one(coords, speed=0.5, K_phase=2.0, gamma=0.12):
    """Contraction to phase center with a hard fast/slow phase gate."""
    coords = np.mod(coords, 1.0)

    def zone(c):
        if c < lower_boundary: return 0
        elif c <= upper_boundary: return 1
        else: return 2
    zones = [zone(c) for c in coords]

    center = circular_mean(coords) 
    diffs  = wrap_signed(coords - center)
    spread = np.max(np.abs(diffs))

    s0, s1 = 0.02, 0.15
    g = np.clip((spread - s0) / (s1 - s0), 0.0, 1.0)
    base = np.ones(4)
    if not all(z == 1 for z in zones):
        for i in range(4):
            if (zones[i] == 0) and any(zones[j] == 1 for j in range(4) if j != i):
                base[i] = 1.0 + 0.5 * g

    dphase = np.abs(wrap_signed(center - BAND_CENTER))
    gain = SLOW if dphase <= BAND_HALF else FAST

    v = gain * speed * base - gamma * diffs
    return v, center

def rhs_all(t, Y, pars):
    """Coupled agent RHS"""
    n = pars["n_agents"]
    speed   = pars.get("speed", 0.4)
    K_phase = pars.get("K_phase", 2.0) 
    gamma   = pars.get("gamma", 3.0)
    k_couple= pars.get("k_couple", 0.2)

    Y = np.asarray(Y)
    dY = np.zeros_like(Y)
    centers = np.zeros(n)

    for i in range(n):
        v_i, phi_i = rhs_one(Y[4*i:4*(i+1)], speed=speed, K_phase=K_phase, gamma=gamma)
        dY[4*i:4*(i+1)] = v_i
        centers[i] = phi_i

    phi_bar = circular_mean(centers)
    for i in range(n):
        phase_err = wrap_signed(phi_bar - centers[i])
        dY[4*i:4*(i+1)] += k_couple * phase_err
    return dY

def build_p_from_segment(T, Y, n_agents, t_start, t_end, torus=True):
    """Make a DxN matrix p from a time window [t_start,t_end]."""
    i0 = np.searchsorted(T, t_start, side='left')
    i1 = np.searchsorted(T, t_end,   side='right')
    seg = Y[i0:i1]                               # (S, 4*n_agents)
    S = seg.shape[0]
    X = seg.reshape(S, n_agents, 4)              # (S, A, D=4)
    if torus:
        X = np.mod(X, 1.0)                       # keep on [0,1)

    # p has shape (D, N) with columns = 4D points
    p = X.transpose(2, 0, 1).reshape(4, -1)      # (4, S*A)
    return p

def distance_from_diagonal(p):
    """
    Circular version of distance-from-diagonal:
    for each 4D point (a column of p in [0,1)),
    compute circular mean on the circle and the std of
    minimal circular differences to that mean.
    Returns one value per column.
    """
    # p: (D, N) in [0,1)
    p = np.mod(p, 1.0)
    theta = 2 * np.pi * p           # (D, N)
    c = np.cos(theta).mean(axis=0)  # (N,)
    s = np.sin(theta).mean(axis=0)  # (N,)
    ang = np.arctan2(s, c)          # mean direction, shape (N,)
    center = ang / (2 * np.pi)      # back to [0,1) “phase”

    # minimal circular differences in [-0.5, 0.5)
    diffs = wrap_signed(p - center[None, :])
    return diffs.std(axis=0)

def gaussian_reference(D, M):
    """Reference from i.i.d. N(0,1): same metric as data."""
    return np.random.randn(D, M).std(axis=0)

def plot_cdf(x, label):
    xs = np.sort(x)
    qs = np.linspace(0, 1, xs.size, endpoint=True)
    plt.plot(qs, xs, label=label, linewidth=2)

def create_continuous_trajectories(noise_strength=0.0):
    """Integrate rhs_all for 4 agents using SDE; noise_strength=0 reproduces ODE"""
    initials = [
        np.array([0.00, 0.25, 0.10, 0.05]),
        np.array([0.18, 0.05, 0.30, 0.08]),
        np.array([0.27, 0.12, 0.02, 0.32]),
        np.array([0.06, 0.20, 0.18, 0.12]),
    ]
    n = len(initials)
    Y0 = np.concatenate(initials).astype(np.float64, copy=False)
    pars = dict(n_agents=n, speed=0.5, K_phase=0.3, gamma=0.12, k_couple=0.05)

    def d(x):
        return rhs_all(0.0, x, pars).astype(np.float64, copy=False)

    def s(x, dw):
        return (noise_strength * dw).astype(np.float64, copy=False)

    sde = SDE(d, s, sdim=len(Y0))

    # integrate
    t = np.arange(0.0, 500, 0.01, dtype=np.float64)
    T, Y, W = sde.integrateAt(t, Y0, dtype=np.float64)
    print("T_end =", T[-1], "  Var(W[:,0]) =", np.var(W[:,0]))

    trajs = [Y[:, 4*i:4*(i+1)] for i in range(n)]
    times = [T for _ in range(n)]
    return trajs, times, T, Y, n

def build_biased_P(D, ix, iy, small=0.03, big=1.0, col_norm=0.9):
    """
    2×D projection with x-axis ~ dim ix and y-axis ~ dim iy.
    Two large entries (≈1) at columns ix (row 0) and iy (row 1),
    all other entries small (~0.2/D). Columns are normalized to col_norm.
    """
    if small is None:
        small = 0.2 / D

    P = np.full((2, D), small, dtype=float)
    P[0, ix] = big   # x-axis strongly aligned with dim ix
    P[1, iy] = big   # y-axis strongly aligned with dim iy

    # keep columns' norms consistent (and positive)
    norms = np.linalg.norm(P, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    P = P / norms * col_norm
    return P

def make_projection_set(num=6, n=4, base_seed=None):
    """
    Biased projections: each view highlights a different pair of dims.
    For 4D, this cycles through (w,x), (y,z), (w,y), (x,z), (w,z), (x,y).
    """
    # choose pairs to emphasize (wrap/cycle if num > len(pairs))
    pairs = [(0,1),(2,3),(0,2),(1,3),(0,3),(1,2)]
    Ps = []
    for v in range(num):
        ix, iy = pairs[v % len(pairs)]
        Ps.append(build_biased_P(n, ix, iy, small=0.2/n, big=1.0, col_norm=0.9))
    return Ps

def draw_axes_for_P(ax, P, length=0.25):
    """Draw projected w/x/y/z axes; keep relative magnitudes (big long, small short)."""
    # global scale so the largest column length maps to `length`
    col_norms = np.linalg.norm(P, axis=0)
    max_col = max(col_norms.max(), 1e-12)
    scale = length / max_col

    for k, lab in enumerate(AXIS_LABELS):
        vec = P[:, k] * scale                    # preserve relative sizes
        ax.arrow(0, 0, vec[0], vec[1],
                 head_width=0.02, head_length=0.03,
                 fc=AXIS_COLORS[lab], ec=AXIS_COLORS[lab],
                 linewidth=1.5, alpha=0.95, length_includes_head=True)
        ax.text(vec[0]*1.1, vec[1]*1.1, lab,
                color=AXIS_COLORS[lab], fontsize=10, fontweight="bold")

def split_at_tile_crossings(pu_tail, pw_tail):
    """
    pu_tail: (m,2) unwrapped projected coords in R^2
    pw_tail: (m,2) wrapped projected coords in [0,1)^2
    Returns x,y arrays with NaNs inserted at any tile crossing.
    """
    tiles = np.floor(pu_tail)
    crosses = np.any(np.diff(tiles, axis=0) != 0, axis=1) 
    x, y = pw_tail[:, 0].copy(), pw_tail[:, 1].copy()
    if np.any(crosses):
        idxs = np.where(crosses)[0]
        for j in idxs[::-1]:
            x = np.insert(x, j+1, np.nan)
            y = np.insert(y, j+1, np.nan)
    return x, y

def project_and_clip_line(P, L4):
    p0_4d, p1_4d = L4[0], L4[-1]
    e0 = (P @ p0_4d).astype(float)
    e1 = (P @ p1_4d).astype(float)

    mid = 0.5 * (e0 + e1)
    shift = np.floor(mid)
    e0 -= shift; e1 -= shift

    x0, y0 = e0; x1, y1 = e1
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [ x0, 1.0 - x0,  y0, 1.0 - y0]

    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > u1: return None
                if t > u0: u0 = t
            else:
                if t < u0: return None
                if t < u1: u1 = t

    q0 = np.array([x0 + u0*dx, y0 + u0*dy])
    q1 = np.array([x0 + u1*dx, y0 + u1*dy])
    return np.vstack([q0, q1])

def build_intersection_lines_4d(boundaries=(0.3, 0.7), n_points=60):
    """Returns 4D polylines made by fixing 3 coords to boundaries, varying the 4th"""
    lo, hi = boundaries
    dims = [0, 1, 2, 3]
    v = np.linspace(0.0, 1.0, n_points)

    lines_4d = []
    for free_dim in dims:
        Q = np.zeros((n_points, 4))
        Q[:, free_dim] = v
        for d in dims:
            if d != free_dim:
                Q[:, d] = lo
        lines_4d.append(Q)

        Q = np.zeros((n_points, 4))
        Q[:, free_dim] = v
        for d in dims:
            if d != free_dim:
                Q[:, d] = hi
        lines_4d.append(Q)

    return lines_4d

def animate_continuous_agents():
    """Run simulation projected to 6 views, and animate"""
    PROJECTIONS_2x4 = make_projection_set(num=6, n=4, base_seed=17)
    trajs, times, T, Y, n_agents = create_continuous_trajectories(noise_strength=0)

    # Gaussian check (early and late)
    Tend = T[-1]
    p_early = build_p_from_segment(T, Y, n_agents, t_start=T[0],      t_end=Tend*0.05)
    p_late  = build_p_from_segment(T, Y, n_agents, t_start=Tend*0.9, t_end=Tend)

    data_early = distance_from_diagonal(p_early)
    data_late  = distance_from_diagonal(p_late)

    ref_early = gaussian_reference(4, M=data_early.size)
    ref_late  = gaussian_reference(4, M=data_late.size)

    plt.figure(figsize=(10, 4))

    # QQ plot: simulation vs Gaussian reference (early)
    plt.subplot(1, 2, 1)
    qqplot_2samples(data_early, ref_early, ax=plt.gca())
    plt.gca().lines[-1].set_label("early")
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    lo = min(xmin, ymin)
    hi = max(xmax, ymax)
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label="y=x")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.title("QQ: early vs Gaussian ref")
    plt.grid(alpha=0.3)
    plt.legend()

    # QQ plot: simulation vs Gaussian reference (late)
    plt.subplot(1, 2, 2)
    qqplot_2samples(data_late, ref_late, ax=plt.gca())
    plt.gca().lines[-1].set_label("late")
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    lo = min(xmin, ymin)
    hi = max(xmax, ymax)
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label="y=x")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.title("QQ: late vs Gaussian ref")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("qq_compare_early_late.png", dpi=160)

    total_time = max(times[i][-1] for i in range(n_agents))
    fps = 30
    n_frames = int(total_time * fps)
    uniform_times = np.linspace(0, total_time, n_frames)

    interpolated_trajs = []
    for i in range(n_agents):
        raw_traj, raw_times = trajs[i], times[i]
        interp_funcs = [interp1d(raw_times, raw_traj[:, d], kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
                        for d in range(4)]
        interpolated_trajs.append(np.column_stack([f(uniform_times) for f in interp_funcs]))

    unwrapped_trajs = []
    for T in interpolated_trajs:
        theta = 2*np.pi*np.mod(T, 1.0) 
        theta_un = np.unwrap(theta, axis=0)
        cycles_un = theta_un / (2*np.pi) 
        unwrapped_trajs.append(cycles_un)

    projected_unwrapped = [
        [ (P @ cycles_un.T).T.astype(np.float32)
        for cycles_un in unwrapped_trajs ]
        for P in PROJECTIONS_2x4
    ]

    projected_wrapped = [
        [ pu - np.floor(pu) for pu in pu_set ]
        for pu_set in projected_unwrapped
    ]

    def torus_cumlen_from_unwrapped(cycles_un):
        d = np.diff(cycles_un, axis=0)
        d = (d + 0.5) % 1.0 - 0.5
        seg = np.linalg.norm(d, axis=1)
        return np.concatenate(([0.0], np.cumsum(seg)))

    cumlen4d = [torus_cumlen_from_unwrapped(c_un) for c_un in unwrapped_trajs]

    # figure
    rows, cols = 2, 3
    fig, axs = plt.subplots(rows, cols, figsize=(18, 12))
    fig.tight_layout()

    pairs = [(0,1),(2,3),(0,2),(1,3),(0,3),(1,2)]
    for idx, (ax, P) in enumerate(zip(axs.flat, PROJECTIONS_2x4), start=1):
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1);      ax.set_ylim(0, 1)
        draw_axes_for_P(ax, P, length=0.2)
        i, j = pairs[(idx-1) % len(pairs)]
        ax.set_title(f"Projection {idx}: {AXIS_LABELS[i]}–{AXIS_LABELS[j]}")


    lines_4d = build_intersection_lines_4d(boundaries=(lower_boundary, upper_boundary), n_points=8)

    for s, (ax, P) in enumerate(zip(axs.flat, PROJECTIONS_2x4)):
        for L4 in lines_4d:
            seg = project_and_clip_line(P, L4)
            if seg is not None:
                ax.plot(seg[:,0], seg[:,1], color="purple", linewidth=0.8, alpha=0.28, zorder=1)


    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    trails = [[ax.plot([], [], '-', color=colors[k], alpha=0.7, linewidth=2,
                       animated=True, zorder=3)[0] for ax in axs.flat]
              for k in range(n_agents)]
    points = [[ax.plot([], [], 'o', color=colors[k], markersize=5,
                       animated=True, zorder=4)[0] for ax in axs.flat]
              for k in range(n_agents)]

    def init():
        """Init blit artists"""
        art = []
        for k in range(n_agents):
            for a in range(len(axs.flat)):
                trails[k][a].set_data([], [])
                points[k][a].set_data([], [])
                art += [trails[k][a], points[k][a]]
        return art

    def update(frame):
        trail_len = 0.1
        art = []
        for k in range(n_agents):
            cl4 = cumlen4d[k][:frame+1]
            start_idx = np.searchsorted(cl4, max(0.0, cl4[-1] - trail_len), 'left')

            for s, ax in enumerate(axs.flat):
                pu = projected_unwrapped[s][k][:frame+1] 
                pw = projected_wrapped[s][k][:frame+1] 

                tail_u = pu[start_idx:]
                tail_w = pw[start_idx:]

                x, y = split_at_tile_crossings(tail_u, tail_w)
                trails[k][s].set_data(x, y)

                head = pw[-1]
                points[k][s].set_data([head[0]], [head[1]])
                art += [trails[k][s], points[k][s]]
        return art

    anim = FuncAnimation(fig, update, frames=len(uniform_times),
                         init_func=init, interval=1000/fps, blit=True, repeat=True)
    anim.save("agents_animation_4D_with_ltl_noise_and gaussian_plot.mp4", writer="ffmpeg", fps=fps, bitrate=2000)
    return anim


if __name__ == "__main__":
    animate_continuous_agents()
