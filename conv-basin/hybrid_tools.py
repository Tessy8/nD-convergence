import numpy as np

PI = np.pi


def wrap_x(x):
    """Wrap x-coordinates into [-pi, pi)."""
    return ((np.asarray(x) + PI) % (2.0 * PI)) - PI


def circular_mean_x(x):
    """Circular mean of x-coordinates in [-pi, pi)."""
    x = np.asarray(x, dtype=float)
    return np.arctan2(np.sin(x).mean(), np.cos(x).mean())


def torus_residual_x(x):
    """Centered residuals on the torus, still expressed in [-pi, pi)."""
    center = circular_mean_x(x)
    return wrap_x(np.asarray(x, dtype=float) - center)


def torus_spread_x(x):
    """Max coordinate deviation from the circular center."""
    return np.abs(torus_residual_x(x)).max()


def inner_radius(dim):
    """Analytical inner-basin radius from the current notes."""
    return 1.0 / np.sqrt(dim * (dim - 1))


def vfield_x(x, guard_offset, delta, sum_guard=1.0):
    """Hybrid vector field in x-coordinates."""
    x = np.asarray(x, dtype=float)
    return np.where((x < guard_offset) | (x.sum() > sum_guard), 1.0, 1.0 - delta)


def is_in_simplex(x, tol=1e-9):
    """Simplex under the sum guard: x_i >= 0 and sum x_i <= 1."""
    x = np.asarray(x, dtype=float)
    return bool(np.all(x >= -tol) and x.sum() <= 1.0 + tol)


def inner_basin_width(x):
    """One-sided inner-basin width W = mean(x) - min(x) used in the notes."""
    x = np.asarray(x, dtype=float)
    return float(x.mean() - x.min())


def is_in_inner_contracting_region(x, tol=1e-9):
    """Paper-style inner basin: simplex plus W <= 1/D with arithmetic mean."""
    x = np.asarray(x, dtype=float)
    return bool(is_in_simplex(x, tol=tol) and inner_basin_width(x) <= (1.0 / len(x)) + tol)


def wrap_event_mask(traj_unwrapped):
    """True where a trajectory crosses a torus tile boundary between samples."""
    traj_unwrapped = np.asarray(traj_unwrapped, dtype=float)
    tiles = np.floor((traj_unwrapped + PI) / (2.0 * PI)).astype(int)
    if len(tiles) <= 1:
        return np.zeros(0, dtype=bool)
    return np.any(np.diff(tiles, axis=0) != 0, axis=1)


def cumulative_wrap_events(traj_unwrapped):
    """Cumulative count of tile-crossing events along a trajectory."""
    jumps = wrap_event_mask(traj_unwrapped)
    return np.concatenate(([0], np.cumsum(jumps.astype(int))))


def first_true_index(mask, start=0):
    """Return first index >= start where mask is true, or -1."""
    idx = np.flatnonzero(np.asarray(mask)[start:])
    if len(idx) == 0:
        return -1
    return int(start + idx[0])


def summarize_trajectory(traj_wrapped, traj_unwrapped, traj_times, conv_tol=0.05):
    """Extract event-based wrapped-entry diagnostics from a trajectory."""
    traj_wrapped = np.asarray(traj_wrapped, dtype=float)
    traj_unwrapped = np.asarray(traj_unwrapped, dtype=float)
    traj_times = np.asarray(traj_times, dtype=float)

    simplex_mask = np.array([is_in_simplex(x) for x in traj_wrapped], dtype=bool)
    contracting_mask = np.array([is_in_inner_contracting_region(x) for x in traj_wrapped], dtype=bool)
    spread = np.array([torus_spread_x(x) for x in traj_wrapped], dtype=float)
    wrap_counts = cumulative_wrap_events(traj_unwrapped)

    initial_in_simplex = bool(simplex_mask[0])
    enters_simplex = bool(np.any(simplex_mask))
    enters_contracting = bool(np.any(contracting_mask))
    enters_simplex_later = bool(np.any(simplex_mask[1:])) if len(simplex_mask) > 1 else False
    enters_contracting_later = bool(np.any(contracting_mask[1:])) if len(contracting_mask) > 1 else False

    first_simplex_idx = first_true_index(simplex_mask)
    first_contracting_idx = first_true_index(contracting_mask)

    wrap_events_before_first_simplex = int(wrap_counts[first_simplex_idx]) if first_simplex_idx >= 0 else -1
    wrap_events_before_first_contracting = int(wrap_counts[first_contracting_idx]) if first_contracting_idx >= 0 else -1

    first_simplex_after_wrap = bool(
        (not initial_in_simplex)
        and first_simplex_idx > 0
        and wrap_events_before_first_simplex > 0
    )

    first_simplex_exit_idx = -1
    same_visit_contracting_idx = -1
    wrapped_entry_to_inner_same_visit = False
    if first_simplex_after_wrap:
        first_simplex_exit_idx = first_true_index(~simplex_mask, start=first_simplex_idx + 1)
        visit_end = len(simplex_mask) if first_simplex_exit_idx == -1 else first_simplex_exit_idx
        same_visit_contracting_idx = first_true_index(contracting_mask, start=first_simplex_idx)
        if same_visit_contracting_idx != -1 and same_visit_contracting_idx < visit_end:
            wrapped_entry_to_inner_same_visit = True
        else:
            same_visit_contracting_idx = -1

    wrap_events_between_simplex_and_contracting = -1
    time_to_contracting_after_simplex = np.inf
    if wrapped_entry_to_inner_same_visit:
        wrap_events_between_simplex_and_contracting = (
            int(wrap_counts[same_visit_contracting_idx]) - wrap_events_before_first_simplex
        )
        time_to_contracting_after_simplex = (
            traj_times[same_visit_contracting_idx] - traj_times[first_simplex_idx]
        )

    return {
        "initial_in_simplex": initial_in_simplex,
        "enters_simplex": enters_simplex,
        "enters_simplex_later": enters_simplex_later,
        "enters_contracting": enters_contracting,
        "enters_contracting_later": enters_contracting_later,
        "simplex_mask": simplex_mask,
        "contracting_mask": contracting_mask,
        "spread": spread,
        "wrap_counts": wrap_counts,
        "first_simplex_idx": first_simplex_idx,
        "first_contracting_idx": first_contracting_idx,
        "wrap_events_before_first_simplex": wrap_events_before_first_simplex,
        "wrap_events_before_first_contracting": wrap_events_before_first_contracting,
        "first_simplex_after_wrap": first_simplex_after_wrap,
        "first_simplex_exit_idx": first_simplex_exit_idx,
        "same_visit_contracting_idx": same_visit_contracting_idx,
        "wrapped_entry_to_inner_same_visit": wrapped_entry_to_inner_same_visit,
        "wrap_events_between_simplex_and_contracting": wrap_events_between_simplex_and_contracting,
        "time_to_contracting_after_simplex": time_to_contracting_after_simplex,
    }


def integrate_pc(x0, guard_offset, delta, t_max, dt, t_tol,
                 sum_guard=1.0,
                 conv_tol=0.05, conv_time=10.0,
                 store_every=1, keep_trajectory=True,
                 stop_on_convergence=True):
    """
    Predictor-corrector integrator with event detection for the x-system.

    Diagnostics are sampled on every accepted step so transient simplex or
    inner-basin visits are not missed when the returned trajectory is
    downsampled for plotting via ``store_every``.
    """
    x = np.array(x0, dtype=float)
    x_unwrapped = x.copy()
    h = dt
    t = 0.0
    dy0 = vfield_x(x, guard_offset, delta, sum_guard=sum_guard)
    conv_for = 0.0
    converged = False
    step = 0

    traj_wrapped = [x.copy()] if keep_trajectory else None
    traj_unwrapped = [x_unwrapped.copy()] if keep_trajectory else None
    traj_times = [0.0] if keep_trajectory else None

    diag_wrapped = [x.copy()]
    diag_unwrapped = [x_unwrapped.copy()]
    diag_times = [0.0]

    while t < t_max:
        x_trial = x + h * dy0
        dy1 = vfield_x(wrap_x(x_trial), guard_offset, delta, sum_guard=sum_guard)

        if not np.allclose(dy1, dy0) and h > t_tol:
            h /= 2.0
            continue

        x_unwrapped = x_unwrapped + h * dy0
        x = wrap_x(x + h * dy0)
        t += h
        dy0 = vfield_x(x, guard_offset, delta, sum_guard=sum_guard)
        h = min(h * 1.5, dt)

        step += 1
        diag_wrapped.append(x.copy())
        diag_unwrapped.append(x_unwrapped.copy())
        diag_times.append(t)

        if keep_trajectory and step % store_every == 0:
            traj_wrapped.append(x.copy())
            traj_unwrapped.append(x_unwrapped.copy())
            traj_times.append(t)

        # Note: convergence uses torus_spread_x, a symmetric circular-mean-based
        # closeness test for synchronization on the torus. Inner-basin membership
        # is checked separately with inner_basin_width, the one-sided arithmetic-
        # mean criterion W <= 1/D from the paper.
        if torus_spread_x(x) < conv_tol:
            conv_for += h
            if conv_for >= conv_time:
                converged = True
                if stop_on_convergence:
                    break
        else:
            conv_for = 0.0

    result = {
        "converged": converged,
        "final_wrapped": x,
        "final_unwrapped": x_unwrapped,
        "t_final": t,
    }

    diag_wrapped = np.asarray(diag_wrapped)
    diag_unwrapped = np.asarray(diag_unwrapped)
    diag_times = np.asarray(diag_times)
    result.update(summarize_trajectory(diag_wrapped, diag_unwrapped, diag_times, conv_tol=conv_tol))

    if keep_trajectory:
        traj_wrapped = np.asarray(traj_wrapped)
        traj_unwrapped = np.asarray(traj_unwrapped)
        traj_times = np.asarray(traj_times)
        result["traj_wrapped"] = traj_wrapped
        result["traj_unwrapped"] = traj_unwrapped
        result["traj_times"] = traj_times
    return result


def classify_points_pc(X, guard_offset, delta, t_max, dt, t_tol,
                       sum_guard=1.0,
                       conv_tol=0.05, conv_time=10.0, progress_every=None):
    """Classify points one-by-one using the same event-accurate integrator."""
    X = np.asarray(X, dtype=float)
    out = np.zeros(len(X), dtype=bool)
    for i, x0 in enumerate(X):
        out[i] = integrate_pc(
            x0,
            guard_offset=guard_offset,
            delta=delta,
            t_max=t_max,
            dt=dt,
            t_tol=t_tol,
            sum_guard=sum_guard,
            conv_tol=conv_tol,
            conv_time=conv_time,
            keep_trajectory=False,
        )["converged"]
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  classified {i + 1}/{len(X)} points")
    return out


def classify_points_with_diagnostics(X, guard_offset, delta, t_max, dt, t_tol,
                                     sum_guard=1.0,
                                     conv_tol=0.05, conv_time=10.0,
                                     progress_every=None):
    """Event-accurate basin classification plus event-based wrapped-entry diagnostics.

    Diagnostics are computed from the integrator's full internal history, so
    this grid sweep does not keep plotting trajectories in memory.
    """
    X = np.asarray(X, dtype=float)
    data = {
        "converged": np.zeros(len(X), dtype=bool),
        "initial_in_simplex": np.zeros(len(X), dtype=bool),
        "enters_simplex_later": np.zeros(len(X), dtype=bool),
        "enters_contracting_later": np.zeros(len(X), dtype=bool),
        "ever_enters_contracting": np.zeros(len(X), dtype=bool),
        "first_simplex_after_wrap": np.zeros(len(X), dtype=bool),
        "wrapped_entry_to_inner_same_visit": np.zeros(len(X), dtype=bool),
        "wrap_events_before_first_simplex": np.full(len(X), -1, dtype=int),
        "wrap_events_before_first_contracting": np.full(len(X), -1, dtype=int),
        "wrap_events_between_simplex_and_contracting": np.full(len(X), -1, dtype=int),
        "first_simplex_idx": np.full(len(X), -1, dtype=int),
        "first_simplex_exit_idx": np.full(len(X), -1, dtype=int),
        "same_visit_contracting_idx": np.full(len(X), -1, dtype=int),
        "time_to_contracting_after_simplex": np.full(len(X), np.inf, dtype=float),
    }
    for i, x0 in enumerate(X):
        result = integrate_pc(
            x0,
            guard_offset=guard_offset,
            delta=delta,
            t_max=t_max,
            dt=dt,
            t_tol=t_tol,
            sum_guard=sum_guard,
            conv_tol=conv_tol,
            conv_time=conv_time,
            keep_trajectory=False,
        )
        data["converged"][i] = result["converged"]
        data["initial_in_simplex"][i] = result["initial_in_simplex"]
        data["enters_simplex_later"][i] = result["enters_simplex_later"]
        data["enters_contracting_later"][i] = result["enters_contracting_later"]
        data["ever_enters_contracting"][i] = result["enters_contracting"]
        data["first_simplex_after_wrap"][i] = result["first_simplex_after_wrap"]
        data["wrapped_entry_to_inner_same_visit"][i] = result["wrapped_entry_to_inner_same_visit"]
        data["wrap_events_before_first_simplex"][i] = result["wrap_events_before_first_simplex"]
        data["wrap_events_before_first_contracting"][i] = result["wrap_events_before_first_contracting"]
        data["wrap_events_between_simplex_and_contracting"][i] = result["wrap_events_between_simplex_and_contracting"]
        data["first_simplex_idx"][i] = result["first_simplex_idx"]
        data["first_simplex_exit_idx"][i] = result["first_simplex_exit_idx"]
        data["same_visit_contracting_idx"][i] = result["same_visit_contracting_idx"]
        data["time_to_contracting_after_simplex"][i] = result["time_to_contracting_after_simplex"]
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  diagnosed {i + 1}/{len(X)} points")
    return data


def break_wraps_3d(traj_wrapped, traj_unwrapped):
    """Insert NaN rows at torus tile crossings for clean plotting."""
    traj_wrapped = np.asarray(traj_wrapped)
    traj_unwrapped = np.asarray(traj_unwrapped)
    tiles = np.floor((traj_unwrapped + PI) / (2.0 * PI)).astype(int)
    jumps = np.any(np.diff(tiles, axis=0) != 0, axis=1)
    if not np.any(jumps):
        return traj_wrapped
    out = traj_wrapped.tolist()
    nan_row = [np.nan, np.nan, np.nan]
    for j in np.where(jumps)[0][::-1]:
        out.insert(j + 1, nan_row)
    return np.asarray(out)


def slice_vector_field(x3_value, n, guard_offset, delta, margin=0.15, sum_guard=1.0):
    """Sample the 3D vector field on the plane x3 = const."""
    vals = np.linspace(-PI + margin, PI - margin, n)
    x1, x2 = np.meshgrid(vals, vals, indexing="ij")
    pts = np.stack([x1.ravel(), x2.ravel(), np.full(x1.size, x3_value)], axis=1)
    vf = np.array([vfield_x(p, guard_offset, delta, sum_guard=sum_guard) for p in pts])
    return vals, pts, vf
