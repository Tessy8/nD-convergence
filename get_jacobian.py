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


def rhs_one(coords, delta=0.1):
    """
    Professor's simple hybrid model, implemented in [0,1) coordinates.

    Let u_i = coords[i] in [0,1).
    Interpret x_i = 2*pi*u_i - pi in [-pi, pi).
    Then:
      if x_i < 0 and sum_j x_j > 1:  dx_i/dt = 1
      else:                          dx_i/dt = 1 - delta

    In u-coordinates, du_i/dt = dx_i/dt / (2*pi).
    """
    coords = np.mod(coords, 1.0)
    D = coords.size

    # angles in [-pi, pi)
    x = 2.0 * np.pi * coords - np.pi
    sum_x = np.sum(x)

    # speeds in u-space
    fast_u = 1.0 / (2.0 * np.pi)
    slow_u = (1.0 - delta) / (2.0 * np.pi)

    v = np.full(D, slow_u, dtype=float)
    if sum_x > 1.0:
        mask = (x < 0.0)
        v[mask] = fast_u

    # phase "center" (used only for section / plotting)
    center = circular_mean(coords)
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


def _first_guard_event(Y, F, dt_max, pars):
    """
    Earliest guard crossing within [0, dt_max] for the simple model.

    Guards (in x-coordinates):
      1) x_i = 0   <=> u_i = 0.5
      2) sum_i x_i = 1  <=> sum_i u_i = (1 + D*pi)/(2*pi)

    Y: current state in u-coordinates (in R^N, typically mod 1)
    F: current du/dt (same shape)
    """
    Y = np.asarray(Y, dtype=float)
    F = np.asarray(F, dtype=float)
    D = pars.get("dim", 4)
    N = Y.size

    # Work in local u-coordinates (no wrapping over one step).
    u = np.mod(Y, 1.0)
    events = []

    # 1) Coordinate guards: u_i = 0.5
    for idx, (ui, fi) in enumerate(zip(u, F)):
        if abs(fi) < _TINY_FLOW:
            continue
        t_hit = (0.5 - ui) / fi
        if 0.0 < t_hit <= dt_max:
            direction = np.sign(fi)  # +1 if crossing upward, -1 otherwise
            events.append((t_hit, "xi", idx, direction))

    # 2) Sum guard: sum x_i = 1  <=> sum u_i = (1 + D*pi)/(2*pi)
    sum_u  = np.sum(u)
    sum_F  = np.sum(F)
    if abs(sum_F) >= _TINY_FLOW:
        c_sum = (1.0 + D * np.pi) / (2.0 * np.pi)
        t_sum = (c_sum - sum_u) / sum_F
        if 0.0 < t_sum <= dt_max:
            direction = np.sign(sum_F)
            events.append((t_sum, "sum", None, direction))

    if not events:
        return None
    events.sort(key=lambda e: e[0])
    return events[0]


def _eval_F_plus_after_crossing(fun, t, Y_cross, event_kind, idx, direction, pars):
    """
    Evaluate F^+ just beyond a guard, by nudging along the event normal.

    event_kind: "xi"  => plane x_i = 0  (normal e_i)
                "sum" => plane sum x_i = 1  (normal all-ones)
    """
    Yp = np.array(Y_cross, dtype=float, copy=True)

    if event_kind == "xi":
        Yp[idx] += direction * _TINY_PUSH

    elif event_kind == "sum":
        N = Yp.size
        Yp += (direction * _TINY_PUSH / np.sqrt(N))

    else:
        raise ValueError(f"Unknown event_kind={event_kind!r}")

    return fun(t, Yp, pars)


def _apply_saltation_update(J, deltaF, n, F_minus):
    """
    General saltation update (no reset case):

      Xi = I + (deltaF * n^T) / (n^T F_minus)
      J  <- Xi @ J

    We implement this as:
      row = n^T J
      J  += (deltaF * row^T) / (n^T F_minus)

    with a clip on the gain for robustness.
    """
    denom = float(np.dot(n, F_minus))
    gain  = np.linalg.norm(deltaF) / max(abs(denom), _TINY_FLOW)

    if gain > _GAIN_CLIP:
        deltaF = deltaF * (_GAIN_CLIP / gain)

    row = n @ J  # shape (N,)
    if abs(denom) < _TINY_FLOW:
        denom = np.copysign(_TINY_FLOW, denom if denom != 0.0 else 1.0)
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

# def make_initials_for_dim(D):
#     """
#     Return a single D-dimensional initial condition (one agent).
#     Reuse the old base4 seed for consistency.
#     """
#     base4_0 = np.array([0.09, 0.37, 0.10, 0.25])

#     if D == 2:
#         return [base4_0[:2].copy()]

#     if D == 4:
#         return [base4_0.copy()]

#     if D == 7:
#         extras0 = np.array([0.31, 0.37, 0.33])
#         return [np.concatenate([base4_0, extras0])]

#     # generic fallback: small deviation from diagonal
#     u0 = 0.2 + 0.05 * np.linspace(0, 1, D)
#     return [u0]


def rhs_all_D(t, Y, pars):
    """
    Apply the simple hybrid model blockwise to each agent.

    Typically we'll use n_agents = 1 so the whole state is just one D-block.
    """
    n_agents = pars["n_agents"]
    D        = pars.get("dim", 4)
    delta    = pars.get("delta", 0.4)

    Y = np.asarray(Y, dtype=float)
    dY = np.zeros_like(Y)
    centers = np.zeros(n_agents)

    for i in range(n_agents):
        sl = slice(D * i, D * (i + 1))
        v_i, phi_i = rhs_one(Y[sl], delta=delta)
        dY[sl] = v_i
        centers[i] = phi_i

    # No inter-agent coupling in this toy model
    return dY


def jacobian_rhs_all_D_fd(Y, pars, eps=1e-6):
    """
    For the professor's model, the vector field is piecewise constant
    in each region, so the continuous Jacobian is zero.
    All the action comes from saltation matrices at the guards.
    """
    N = len(Y)
    return np.zeros((N, N), dtype=float)


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
    (Simple model: continuous Jacobian is zero, only saltation jumps.)
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
        ev = _first_guard_event(Ya, Fa, step_cap, pars)
        if ev is None:
            # purely continuous micro-step: dJ/dt = 0, so J stays constant
            dt = step_cap
            Ya = Ya + dt * Fa
            t_now += dt
            Fa = rhs_all_D(t_now, Ya, pars)
            continue

        dt_hit, kind, idx, direction = ev

        # advance to the guard
        if dt_hit > _TINY_TIME:
            Ya = Ya + dt_hit * Fa
            t_now += dt_hit

        # saltation at guard
        F_plus = _eval_F_plus_after_crossing(rhs_all_D, t_now, Ya,
                                             kind, idx, direction, pars)
        deltaF = F_plus - Fa

        # normal for this event
        if kind == "xi":
            n = np.zeros(N, dtype=float)
            n[idx] = 1.0
        elif kind == "sum":
            n = np.ones(N, dtype=float)
        else:
            raise ValueError(f"Unknown guard kind={kind!r}")

        # apply saltation
        J = _apply_saltation_update(J, deltaF, n, Fa)

        Ya = Ya       # already at the guard
        Fa = F_plus   # post-event vector field

        # small hop after the event to avoid sticking exactly on the guard
        if t_b - t_now > _TINY_TIME:
            hop = min(_POST_EVENT_DT, t_b - t_now)
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
      * build system with the simple hybrid model
      * compute Poincaré return Jacobians
      * print eigenvalues and transverse contraction
    """
    for D in D_list:
        initials = make_initials_for_dim(D)
        n_agents = len(initials)  # typically 1 now
        Y0 = np.concatenate(initials).astype(np.float64, copy=False)

        pars = dict(
            n_agents=n_agents,
            dim=D,
            delta=0.5,      # <-- professor's (1 - δ) parameter
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
        T_total=5000.0,   # enough time to collect several returns
        dt=5e-3,
        eps_J=1e-6,
        phi_star=BAND_CENTER,  # section at mean phase = 0.5
        max_returns=1000
    )
