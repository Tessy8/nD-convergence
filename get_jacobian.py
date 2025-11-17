import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from integro_sde.sde import SDE
from plotmisc.stats import qqplot

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
    """Map real u to [-0.5, 0.5)."""
    return (u + 0.5) % 1.0 - 0.5


def circular_mean(u):
    """Mean on the unit circle; returns value in [0,1)."""
    theta = 2 * np.pi * np.mod(u, 1.0)
    c, s = np.mean(np.cos(theta)), np.mean(np.sin(theta))
    ang = np.arctan2(s, c) % (2 * np.pi)
    return ang / (2 * np.pi)


def torus_diff_nd(A):
    d = np.diff(A, axis=0)
    return wrap_signed(d)


def cumlen_torus_nd(A):
    if len(A) <= 1:
        return np.array([0.0], float)
    seg = np.linalg.norm(torus_diff_nd(A), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def segment_wrap_jumps(x, y, jump=0.5):
    ddx, ddy = np.diff(x), np.diff(y)
    jumps = (np.abs(ddx) > jump) | (np.abs(ddy) > jump)
    if not np.any(jumps):
        return x, y
    x2, y2 = x.copy(), y.copy()
    for j in np.where(jumps)[0][::-1]:
        x2 = np.insert(x2, j + 1, np.nan)
        y2 = np.insert(y2, j + 1, np.nan)
    return x2, y2


def rhs_one(coords, speed=0.5, K_phase=2.0, gamma=0.12):
    """Contraction to phase center with a hard fast/slow phase gate."""
    coords = np.mod(coords, 1.0)
    D = coords.size

    def zone(c):
        if c < lower_boundary:
            return 0
        elif c <= upper_boundary:
            return 1
        else:
            return 2

    zones = [zone(c) for c in coords]

    center = circular_mean(coords)
    diffs  = wrap_signed(coords - center)
    spread = np.max(np.abs(diffs))

    s0, s1 = 0.02, 0.15
    g = np.clip((spread - s0) / (s1 - s0), 0.0, 1.0)

    base = np.ones(D)
    if not all(z == 1 for z in zones):
        for i in range(D):
            if (zones[i] == 0) and any(zones[j] == 1 for j in range(D) if j != i):
                base[i] = 1.0 + 0.5 * g

    dphase = np.abs(wrap_signed(center - BAND_CENTER))
    gain = SLOW if dphase <= BAND_HALF else FAST

    v = gain * speed * base - gamma * diffs
    return v, center


def compute_zones_vector(Y, lower=lower_boundary, upper=upper_boundary):
    Y_mod = np.mod(Y, 1.0)
    zones = np.empty_like(Y_mod, dtype=np.int8)
    zones[Y_mod < lower] = 0
    mid_mask = (Y_mod >= lower) & (Y_mod <= upper)
    zones[mid_mask] = 1
    zones[Y_mod > upper] = 2
    return zones


# ------------------------ Robust event & saltation helpers ------------------

_TINY_TIME = 1e-12
_TINY_FLOW = 1e-6
_TINY_PUSH = 1e-6
_POST_EVENT_DT = 1e-3
_MAX_EVENTS_PER_STEP = 64
_GAIN_CLIP = 10.0


def _first_guard_event(Y, F, dt_max,
                       lower=lower_boundary, upper=upper_boundary):
    """Earliest guard crossing within [0, dt_max] via linear extrapolation."""
    events = []
    for idx, (y, f) in enumerate(zip(Y, F)):
        if abs(f) < _TINY_FLOW:
            continue
        y0 = y % 1.0
        if f > 0:
            if y0 < lower:
                t = (lower - y0) / f
                if 0 < t <= dt_max:
                    events.append((t, idx, lower, +1))
            elif y0 <= upper:
                t = (upper - y0) / f
                if 0 < t <= dt_max:
                    events.append((t, idx, upper, +1))
        else:
            if y0 > upper:
                t = (upper - y0) / f
                if 0 < t <= dt_max:
                    events.append((t, idx, upper, -1))
            elif y0 >= lower:
                t = (lower - y0) / f
                if 0 < t <= dt_max:
                    events.append((t, idx, lower, -1))
    if not events:
        return None
    events.sort(key=lambda e: e[0])
    return events[0]


def _eval_F_plus_after_crossing(fun, t, Y_cross, idx, direction, pars):
    Yp = np.array(Y_cross, dtype=float, copy=True)
    Yp[idx] = Yp[idx] + (_TINY_PUSH if direction > 0 else -_TINY_PUSH)
    return fun(t, Yp, pars)


def _apply_saltation_update(J, deltaF, row, denom):
    gain = np.linalg.norm(deltaF) / max(abs(denom), _TINY_FLOW)
    if gain > _GAIN_CLIP:
        deltaF = deltaF * (_GAIN_CLIP / gain)
    J += np.outer(deltaF, row) / denom
    return J


# ============================ System RHS (multi-agent) ======================

def make_initials_for_dim(D):
    base4 = [
        np.array([0.00, 0.25, 0.10, 0.05]),
        np.array([0.18, 0.05, 0.10, 0.08]),
        np.array([0.27, 0.12, 0.02, 0.12]),
        np.array([0.06, 0.20, 0.18, 0.12]),
    ]
    if D == 4:
        return [b.copy() for b in base4]
    if D == 2:
        return [b[:2].copy() for b in base4]
    if D == 7:
        extras = [
            np.array([0.31, 0.37, 0.33]),
            np.array([0.31, 0.27, 0.43]),
            np.array([0.41, 0.47, 0.23]),
            np.array([0.41, 0.38, 0.53]),
        ]
        return [np.concatenate([b, e]) for b, e in zip(base4, extras)]
    raise ValueError(f"Unsupported dimension D={D}")


def rhs_all_D(t, Y, pars):
    n_agents = pars["n_agents"]
    D        = pars.get("dim", 4)

    speed    = pars.get("speed",   0.4)
    K_phase  = pars.get("K_phase", 2.0)
    gamma    = pars.get("gamma",   3.0)
    k_couple = pars.get("k_couple", 0.2)

    Y = np.asarray(Y)
    dY = np.zeros_like(Y)
    centers = np.zeros(n_agents)

    for i in range(n_agents):
        sl = slice(D * i, D * (i + 1))
        v_i, phi_i = rhs_one(Y[sl], speed=speed, K_phase=K_phase, gamma=gamma)
        dY[sl] = v_i
        centers[i] = phi_i

    phi_bar = circular_mean(centers)
    for i in range(n_agents):
        phase_err = wrap_signed(phi_bar - centers[i])
        sl = slice(D * i, D * (i + 1))
        dY[sl] += k_couple * phase_err
    return dY


def jacobian_rhs_all_D_fd(Y, pars, eps=1e-6):
    Y = np.asarray(Y, dtype=float)
    N = Y.size
    F0 = rhs_all_D(0.0, Y, pars)
    J = np.zeros((N, N), dtype=float)
    for j in range(N):
        dY = np.zeros_like(Y)
        dY[j] = eps
        F1 = rhs_all_D(0.0, Y + dY, pars)
        J[:, j] = (F1 - F0) / eps
    return J


# ============================ Mean-phase section tools ======================

def phi_bar_of_state(Y, pars):
    """Mean over agents of each agent's circular mean; returns in [0,1)."""
    D = pars.get("dim", 4)
    n_agents = pars["n_agents"]
    centers = []
    for i in range(n_agents):
        sl = slice(D * i, D * (i + 1))
        centers.append(circular_mean(np.mod(Y[sl], 1.0)))
    return circular_mean(np.array(centers))


def unwrap_phi_series(T, Y, pars):
    """Compute unwrapped mean-phase (in cycles) along the trajectory."""
    ph = np.array([phi_bar_of_state(Yk, pars) for Yk in Y])
    theta = 2 * np.pi * ph
    theta_un = np.unwrap(theta)
    return theta_un / (2 * np.pi)  # cycles


def find_section_crossings(T, Y, pars, phi_star=BAND_CENTER, max_crossings=None):
    """Times t where unwrapped mean-phase crosses m + phi_star."""
    u = unwrap_phi_series(T, Y, pars)  # cycles
    ts = []
    # choose starting integer so first target > u[0]
    m = np.floor(u[0] - phi_star) + 1
    target = m + phi_star
    for k in range(1, len(T)):
        if u[k-1] < target <= u[k]:
            # linear interpolation on u
            a = (target - u[k-1]) / (u[k] - u[k-1])
            t_hit = T[k-1] + a * (T[k] - T[k-1])
            ts.append(t_hit)
            if (max_crossings is not None) and (len(ts) >= max_crossings):
                break
            target += 1.0  # next crossing
    return ts


def grad_phi_fd(Y, pars, eps=1e-6):
    """Finite-difference gradient of phi_bar (handle wrap via angles)."""
    base_theta = 2 * np.pi * phi_bar_of_state(Y, pars)
    N = len(Y)
    g = np.zeros(N, dtype=float)
    for j in range(N):
        Yp = Y.copy()
        Yp[j] += eps
        th = 2 * np.pi * phi_bar_of_state(Yp, pars)
        dth = (th - base_theta + np.pi) % (2 * np.pi) - np.pi
        g[j] = dth / eps / (2 * np.pi)  # gradient of cycles, not radians
    return g


# ======================= Jacobian over a return (section->section) =========

def integrate_J_over_interval(Y_start, t_a, t_b, pars, dt_max=5e-3, eps_J=1e-6):
    """
    Integrate variational dynamics from t_a to t_b with saltations at guards.
    Returns (J, Y_end).
    """
    N = Y_start.size
    Ya = Y_start.copy()
    t_now = t_a
    J = np.eye(N, dtype=float)

    # bootstrap flow
    Fa = rhs_all_D(t_now, Ya, pars)

    while t_b - t_now > _TINY_TIME:
        step_cap = min(dt_max, t_b - t_now)
        # guard before end of this micro-step?
        ev = _first_guard_event(Ya, Fa, step_cap,
                                lower=lower_boundary, upper=upper_boundary)
        if ev is None:
            # purely continuous micro-step
            A = jacobian_rhs_all_D_fd(Ya, pars, eps=eps_J)
            dt = step_cap
            J = (np.eye(N) + dt * A) @ J
            Ya = Ya + dt * Fa
            t_now += dt
            Fa = rhs_all_D(t_now, Ya, pars)
            continue

        # go to guard
        dt_hit, idx, c, direction = ev
        if dt_hit > _TINY_TIME:
            A = jacobian_rhs_all_D_fd(Ya, pars, eps=eps_J)
            J = (np.eye(N) + dt_hit * A) @ J
            Ya = Ya + dt_hit * Fa
            t_now += dt_hit

        # saltation at guard
        F_plus = _eval_F_plus_after_crossing(rhs_all_D, t_now, Ya, idx, direction, pars)
        deltaF = F_plus - Fa
        denom  = 0.5 * (Fa[idx] + F_plus[idx])
        if abs(denom) >= _TINY_FLOW:
            row = J[idx, :]
            J = _apply_saltation_update(J, deltaF, row, denom)

        Ya = Ya  # already at guard
        Fa = F_plus

        # de-bounce hop
        if t_b - t_now > _TINY_TIME:
            hop = min(_POST_EVENT_DT, t_b - t_now)
            A = jacobian_rhs_all_D_fd(Ya, pars, eps=eps_J)
            J = (np.eye(N) + hop * A) @ J
            Ya = Ya + hop * Fa
            t_now += hop
            Fa = rhs_all_D(t_now, Ya, pars)

    return J, Ya


def interpolate_state_at_time(T, Y, t_star):
    """Linear interpolate state on precomputed (T,Y)."""
    k = np.searchsorted(T, t_star, side='right')
    k = np.clip(k, 1, len(T) - 1)
    a = (t_star - T[k-1]) / (T[k] - T[k-1] + 1e-16)
    return (1 - a) * Y[k-1] + a * Y[k]


# --- Poincaré section helpers (phi_bar == 0.5 by default) ---

def _phi_bar_of_state(Y, pars):
    """Global phase: circular-mean of agents' internal centers."""
    n_agents = pars["n_agents"]
    D = pars["dim"]
    Y = np.asarray(Y)
    centers = np.empty(n_agents, float)
    for i in range(n_agents):
        sl = slice(D * i, D * (i + 1))
        centers[i] = circular_mean(Y[sl])
    return circular_mean(centers)


def section_value_phi(Y, pars, phi_star=0.5):
    """Section s(Y)=wrap_signed(phi_bar(Y)-phi_star). Zero at returns."""
    return wrap_signed(_phi_bar_of_state(Y, pars) - phi_star)


def grad_section_fd(Y, pars, phi_star=0.5, eps=1e-6):
    """
    Finite-diff gradient of s(Y) with wrap-aware differencing.
    Returns g ≈ ∇s(Y) of shape (N,).
    """
    Y = np.asarray(Y, float)
    N = Y.size
    base = section_value_phi(Y, pars, phi_star)
    g = np.empty(N, float)
    for j in range(N):
        Yp = Y.copy()
        Yp[j] += eps
        sj = section_value_phi(Yp, pars, phi_star)
        # wrap-aware increment
        ds = wrap_signed(sj - base)
        g[j] = ds / eps
    return g


def oblique_projector(g, f, tol=1e-9):
    """
    P = I - f g^T / (g^T f).  Projects along flow f onto the section normal g.
    Clamps near-grazing (|g·f| small) for numerical robustness.
    """
    g = np.asarray(g, float)
    f = np.asarray(f, float)
    N = f.size
    dot = float(np.dot(g, f))
    if abs(dot) < tol:
        dot = np.copysign(tol, dot if dot != 0.0 else 1.0)
    return np.eye(N) - np.outer(f, g) / dot


# --------------------- Bouligand derivative (flow with saltation) ----------

def B_derivative_flow_with_saltation(
        Y_start, t_a, t_b, pars,
        dt_max=5e-3, eps_J=1e-6):
    """
    Bouligand derivative of the hybrid flow map from t_a to t_b
    at base point Y_start, using:
      * continuous variational dynamics dJ/dt = A(Y) J,
      * saltation rank-one updates at each guard crossing.

    Returns:
      J_B : (N,N) Bouligand derivative of the flow map Phi_{t_b,t_a}
      Y_b : state Phi_{t_b,t_a}(Y_start)
    """
    J_B, Y_b = integrate_J_over_interval(
        Y_start, t_a, t_b, pars,
        dt_max=dt_max, eps_J=eps_J
    )
    return J_B, Y_b


def jacobians_poincare_full_D(
        Y0, pars, T_total, dt=5e-3, eps_J=1e-6,
        phi_star=BAND_CENTER, max_returns=10, sec_tol=1e-9):
    """
    Build the TRUE Poincaré return-map B-derivative for h(Y)=phi_bar(Y)-phi_star=0:
      1) integrate once coarsely to find section-crossing times,
      2) for each consecutive pair (a,b) of section times:
           - interpolate state at a (Ya),
           - compute B-derivative of flow a→b with saltations: J_B,
           - compute oblique projectors at the endpoints and sandwich:
             J_ret = P_out @ J_B @ P_in.
    Returns:
      Js: list of N×N Jacobians (section→section)
      Ts: list of (t_in, t_out) time intervals for each return.
    """
    # 1) coarse pass to get (T,Y,dY) and section times
    ode = OdePC(rhs_all_D)
    T, Y, dY = ode(Y0, t0=  0.0, t1=T_total, dt=dt, pars=pars, withDy=True)

    section_ts = find_section_crossings(T, Y, pars, phi_star=phi_star,
                                        max_crossings=max_returns + 1)
    if len(section_ts) < 2:
        return [], []

    Js, Ts = [], []
    for a, b in zip(section_ts[:-1], section_ts[1:]):
        # 2a) state on the section (interpolated)
        Ya = interpolate_state_at_time(T, Y, a)

        # 2b) projection at the IN endpoint (onto section tangent along flow)
        f_in = rhs_all_D(a, Ya, pars)                          # flow at the hit
        g_in = grad_section_fd(Ya, pars, phi_star=phi_star)    # ∇h at the hit
        P_in = oblique_projector(g_in, f_in, tol=sec_tol)      # I - f g^T / (g^T f)

        # 2c) Bouligand derivative of hybrid flow a→b with saltations
        J_B, Yb = B_derivative_flow_with_saltation(
            Ya, a, b, pars, dt_max=dt, eps_J=eps_J
        )

        # 2d) projection at the OUT endpoint
        f_out = rhs_all_D(b, Yb, pars)
        g_out = grad_section_fd(Yb, pars, phi_star=phi_star)
        P_out = oblique_projector(g_out, f_out, tol=sec_tol)

        # 2e) return-map B-derivative restricted to the section
        J_ret = P_out @ J_B @ P_in

        Js.append(J_ret)
        Ts.append((a, b))

    return Js, Ts


# ============================= Analyses / viz ===============================

def analyze_cycles_poincare_multiD(
        D_list=(2, 4, 7),
        T_total=20.0,
        dt=5e-3,
        eps_J=1e-6,
        phi_star=BAND_CENTER,
        max_returns=6):
    """
    For each D:
      * build system
      * compute Poincaré return Jacobians to mean-phase section (phi_star)
      * print eigenvalues and transverse contraction
    """
    for D in D_list:
        initials = make_initials_for_dim(D)
        n_agents = len(initials)
        Y0 = np.concatenate(initials).astype(np.float64, copy=False)

        pars = dict(
            n_agents=n_agents,
            dim=D,
            speed=0.5,
            K_phase=0.3,
            gamma=0.12,
            k_couple=0.05,
        )

        Js, Ts = jacobians_poincare_full_D(
            Y0, pars, T_total=T_total, dt=dt, eps_J=eps_J,
            phi_star=phi_star, max_returns=max_returns
        )

        print(f"\n=== D={D} (Poincaré at phi*={phi_star}) ===")
        print(f"Returns found: {len(Js)} over [0, {T_total}]")

        for (J, (t_s, t_e)) in zip(Js, Ts):
            eigvals, _ = np.linalg.eig(J)
            mags = np.abs(eigvals)

            tol_neutral = 1e-3
            # ignore near-1 neutral and near-0 (projection) modes
            neutral_mask = (np.abs(mags - 1.0) < tol_neutral) | (mags < 1e-6)
            transverse_mask = ~neutral_mask

            c_perp = mags[transverse_mask].max() if np.any(transverse_mask) else 1.0
            print(f"  Return [{t_s:.3f} → {t_e:.3f}] Δt={t_e - t_s:.3f}")
            print(f"    eigenvalues: {eigvals}")
            print(f"    transverse contraction c_perp ≈ {c_perp}")


if __name__ == "__main__":
    # Use the TRUE return map (section-to-section), not a fixed-time map
    analyze_cycles_poincare_multiD(
        D_list=(7, 4, 2),
        T_total=30.0,   # enough time to collect several returns
        dt=5e-3,
        eps_J=1e-6,
        phi_star=BAND_CENTER,  # section at mean phase = 0.5
        max_returns=50
    )
