import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

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

def create_continuous_trajectories():
    """Integrate rhs_all for 4 agents; returns agent trajectories"""
    initials = [
        np.array([0.0, 0.2, 0.15, 0.1]),
        np.array([0.2, 0.15, 0.1, 0.0]),
        np.array([0.15, 0.1, 0.0, 0.2]),
        np.array([0.05, 0.1, 0.1, 0.2]),
    ]
    n = len(initials)
    Y0 = np.concatenate(initials)

    ode = OdePC(rhs_all)
    pars = dict(n_agents=n, speed=0.5, K_phase=0.3, gamma=0.12, k_couple=0.05)
    t, Y, _ = ode(Y0, t0=0.0, t1=20.0, dt=0.001, tTol=1e-6, pars=pars, withDy=True)

    trajs = [Y[:, 4*i:4*(i+1)] for i in range(n)]
    times = [t for _ in range(n)]
    return trajs, times

def make_projection(n=4, seed=None, min_norm=0.90, max_norm=0.98, cmin=0.90, max_its=6):
    """Random 2 x n with columns in first quadrant and aligned with (1,1)"""
    rng = np.random.default_rng(seed)
    P = rng.random((2, n)) + 0.15
    P /= np.linalg.norm(P, axis=0, keepdims=True)

    u = np.array([1.0, 1.0]) / np.sqrt(2.0)
    for k in range(n):
        p, its = P[:, k], 0
        while p.dot(u) < cmin and its < max_its:
            lam = 0.35 + 0.25 * rng.random()
            p = (1.0 - lam) * p + lam * u
            p /= np.linalg.norm(p)
            its += 1
        P[:, k] = p

    scales = rng.uniform(min_norm, max_norm, size=(1, n))
    return P * scales

def make_projection_set(num=6, n=4, base_seed=17):
    """Create projection matrices for 6 subplots"""
    return [make_projection(n=n, seed=base_seed + s) for s in range(num)]

def draw_axes_for_P(ax, P, length=0.25):
    """Draw projected w/x/y/z axes as colored arrows from the origin"""
    for k, lab in enumerate(AXIS_LABELS):
        vec = P[:, k]
        if np.linalg.norm(vec) > 1e-9:
            vec = (vec / np.linalg.norm(vec)) * length
        ax.arrow(0, 0, vec[0], vec[1],
                 head_width=0.02, head_length=0.03,
                 fc=AXIS_COLORS[lab], ec=AXIS_COLORS[lab],
                 linewidth=1.5, alpha=0.9, length_includes_head=True)
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
    trajs, times = create_continuous_trajectories()
    n_agents = len(trajs)

    total_time = max(times[i][-1] for i in range(n_agents))
    fps = 100
    n_frames = int(total_time * fps)
    uniform_times = np.linspace(0, total_time, n_frames)

    interpolated_trajs = []
    for i in range(n_agents):
        raw_traj, raw_times = trajs[i], times[i]
        interp_funcs = [interp1d(raw_times, raw_traj[:, d], kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
                        for d in range(4)]
        interpolated_trajs.append(np.column_stack([f(uniform_times) for f in interp_funcs]))

    # wrapped_trajs = [np.mod(T, 1.0) for T in interpolated_trajs]

    # projected_trajs = [[np.mod((P @ T_wr.T).T, 1.0).astype(np.float32)
    #                 for T_wr in wrapped_trajs]
    #                for P in PROJECTIONS_2x4]

    # cumlen4d = [cumlen_torus_nd(T_wr) for T_wr in wrapped_trajs]

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

    for idx, (ax, P) in enumerate(zip(axs.flat, PROJECTIONS_2x4), start=1):
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1);      ax.set_ylim(0, 1)
        draw_axes_for_P(ax, P, length=0.2)
        ax.set_title(f"Projection {idx}")

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
    anim.save("agents_animation_4D.mp4", writer="ffmpeg", fps=fps, bitrate=2000)
    return anim


if __name__ == "__main__":
    animate_continuous_agents()
