"""
Literal professor model on x in [0, 2*pi)^3 with c = 2*pi.

The rule is implemented exactly componentwise:

    dx_k is slow when both
      1. x_k > 0
      2. x_1 + x_2 + x_3 < 2*pi

and fast otherwise.

Because the sampled grid lies in the interior of [0, 2*pi)^3, condition (1) is
true at every sampled point, so the interior dynamics reduce to a shared
fast/slow switch based only on the sum. The code still implements the
componentwise professor rule explicitly.

Outputs:
  - console summary
  - conv-basin/output/torus_0_2pi_literal_professor_c2pi_analysis.json
"""

import json
import os
import numpy as np

PI = np.pi
TWO_PI = 2.0 * np.pi

# configuration
DELTA         = 0.5
DT            = 0.05
T_TOL         = 1e-5
CONV_TOL      = 0.05
CONV_TIME     = 10.0
STANDARD_TMAX = 120.0
LONG_TMAX     = 400.0

GRID_N        = 9
GRID_MARGIN   = 0.35
COORD_GUARD_U = 0.0
SUM_GUARD_U   = TWO_PI

OUTPUT_DIR    = "conv-basin/output"
OUTPUT_JSON   = os.path.join(OUTPUT_DIR, "torus_0_2pi_literal_professor_c2pi_analysis.json")


def wrap_theta(x):
    """Wrap coordinates into [0, 2*pi)."""
    return np.mod(np.asarray(x, dtype=float), TWO_PI)


def circular_mean_theta(x):
    """Circular mean expressed back in [0, 2*pi)."""
    x = np.asarray(x, dtype=float)
    mean = np.arctan2(np.sin(x).mean(), np.cos(x).mean())
    return float(np.mod(mean, TWO_PI))


def torus_residual_theta(x):
    """Centered residuals on the torus, expressed in (-pi, pi]."""
    center = circular_mean_theta(x)
    return ((np.asarray(x, dtype=float) - center + PI) % TWO_PI) - PI


def torus_spread_theta(x):
    """Max deviation from the circular center on the torus."""
    return float(np.abs(torus_residual_theta(x)).max())


def vfield_theta(x, coord_guard_x, sum_guard, delta):
    """Literal professor vector field applied componentwise on x in [0, 2*pi)^3."""
    x = wrap_theta(x)
    sum_is_small = x.sum() < sum_guard
    coord_is_positive = x > coord_guard_x
    slow_mask = coord_is_positive & sum_is_small
    return np.where(slow_mask, 1.0 - delta, 1.0)


def integrate_pc_theta(x0, coord_guard_x, sum_guard, delta, t_max, dt, t_tol,
                       conv_tol=0.05, conv_time=10.0):
    """Predictor-corrector integrator in [0, 2*pi)^3."""
    x = wrap_theta(x0)
    x_unwrapped = np.array(x0, dtype=float)
    h = dt
    t = 0.0
    dx0 = vfield_theta(x, coord_guard_x, sum_guard, delta)
    conv_for = 0.0
    converged = False

    while t < t_max:
        x_trial = x + h * dx0
        dx1 = vfield_theta(x_trial, coord_guard_x, sum_guard, delta)

        if not np.allclose(dx1, dx0) and h > t_tol:
            h /= 2.0
            continue

        x_unwrapped = x_unwrapped + h * dx0
        x = wrap_theta(x + h * dx0)
        t += h
        dx0 = vfield_theta(x, coord_guard_x, sum_guard, delta)
        h = min(h * 1.5, dt)

        if torus_spread_theta(x) < conv_tol:
            conv_for += h
            if conv_for >= conv_time:
                converged = True
                break
        else:
            conv_for = 0.0

    return {
        "converged": bool(converged),
        "t_final": float(t),
        "final_wrapped": x.tolist(),
    }


def build_grid():
    vals = np.linspace(GRID_MARGIN, TWO_PI - GRID_MARGIN, GRID_N)
    grid = np.meshgrid(vals, vals, vals, indexing="ij")
    X = np.stack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()], axis=1)
    return vals, X


def classify_grid(points, t_max, label):
    conv = np.zeros(len(points), dtype=bool)
    t_final = np.zeros(len(points), dtype=float)

    print(f"Classifying {len(points)} points for {label} (t={t_max:.0f})")
    for i, x0 in enumerate(points):
        res = integrate_pc_theta(
            x0,
            coord_guard_x=COORD_GUARD_U,
            sum_guard=SUM_GUARD_U,
            delta=DELTA,
            t_max=t_max,
            dt=DT,
            t_tol=T_TOL,
            conv_tol=CONV_TOL,
            conv_time=CONV_TIME,
        )
        conv[i] = res["converged"]
        t_final[i] = res["t_final"]
        if (i + 1) % max(1, len(points) // 8) == 0:
            print(f"  finished {i + 1}/{len(points)} points")
    return conv, t_final


def sample_points(points, mask, limit=20):
    return [np.round(pt, 6).tolist() for pt in points[mask][:limit]]


def summarize_transition(points, conv_short, conv_long):
    gained = (~conv_short) & conv_long
    lost = conv_short & (~conv_long)
    both = conv_short & conv_long
    neither = (~conv_short) & (~conv_long)

    return {
        "counts": {
            "converged_t120": int(conv_short.sum()),
            "converged_t400": int(conv_long.sum()),
            "gained_by_more_time": int(gained.sum()),
            "lost_by_more_time": int(lost.sum()),
            "converged_both": int(both.sum()),
            "converged_neither": int(neither.sum()),
        },
        "fractions": {
            "fraction_converged_t120": float(conv_short.mean()),
            "fraction_converged_t400": float(conv_long.mean()),
            "fraction_nonconverged_t120_that_converge_t400": (
                float(gained.sum() / (~conv_short).sum()) if (~conv_short).sum() else 0.0
            ),
        },
        "sample_points": {
            "gained_by_more_time": sample_points(points, gained),
            "lost_by_more_time": sample_points(points, lost),
            "converged_both": sample_points(points, both),
            "converged_neither": sample_points(points, neither),
        },
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vals, X = build_grid()

    conv_120, tfin_120 = classify_grid(X, STANDARD_TMAX, "professor_model")
    conv_400, tfin_400 = classify_grid(X, LONG_TMAX, "professor_model")

    gained = (~conv_120) & conv_400
    inside_slow_simplex = np.all(X > COORD_GUARD_U, axis=1) & (X.sum(axis=1) <= SUM_GUARD_U)

    summary = {
        "model": {
            "torus_domain": [0.0, TWO_PI],
            "dimension": 3,
            "delta": DELTA,
            "dt": DT,
            "t_tol": T_TOL,
            "conv_tol": CONV_TOL,
            "conv_time": CONV_TIME,
            "model_name": "literal_professor_torus_model",
            "coordinate_guard_x": COORD_GUARD_U,
            "sum_guard_u": SUM_GUARD_U,
            "equation": "dx_k is slow iff x_k > 0 and x1 + x2 + x3 < 2*pi; fast otherwise",
            "slow_region": "x_k > 0 for all k and x1 + x2 + x3 < 2*pi",
            "fast_region": "for each k, dx_k is fast when x_k <= 0 or x1 + x2 + x3 >= 2*pi",
            "interpretation": (
                "This script implements the professor's rule componentwise. On "
                "the sampled interior grid x_k > 0 holds automatically, so the "
                "interior dynamics collapse to a shared switch based on the sum."
            ),
        },
        "grid": {
            "n_per_axis": GRID_N,
            "margin": GRID_MARGIN,
            "total_points": int(len(X)),
            "axis_values": np.round(vals, 6).tolist(),
        },
        "standard_vs_long_horizon": summarize_transition(X, conv_120, conv_400),
        "geometric_breakdown": {
            "points_inside_slow_simplex": int(inside_slow_simplex.sum()),
            "points_outside_slow_simplex": int((~inside_slow_simplex).sum()),
            "fraction_inside_slow_simplex": float(inside_slow_simplex.mean()),
            "expected_volume_fraction_simplex": float(1.0 / 6.0),
            "converged_t120_inside_slow_simplex": int((conv_120 & inside_slow_simplex).sum()),
            "converged_t120_outside_slow_simplex": int((conv_120 & (~inside_slow_simplex)).sum()),
            "converged_t400_inside_slow_simplex": int((conv_400 & inside_slow_simplex).sum()),
            "converged_t400_outside_slow_simplex": int((conv_400 & (~inside_slow_simplex)).sum()),
            "gained_by_more_time_inside_slow_simplex": int((gained & inside_slow_simplex).sum()),
            "gained_by_more_time_outside_slow_simplex": int((gained & (~inside_slow_simplex)).sum()),
        },
        "vector_field_check": {
            "value_at_center": vfield_theta(np.array([PI, PI, PI]), COORD_GUARD_U, SUM_GUARD_U, DELTA).tolist(),
            "value_near_origin": vfield_theta(np.array([0.2, 0.2, 0.2]), COORD_GUARD_U, SUM_GUARD_U, DELTA).tolist(),
            "value_on_boundary_example": vfield_theta(np.array([0.0, 0.2, 0.2]), COORD_GUARD_U, SUM_GUARD_U, DELTA).tolist(),
            "expected_behavior": (
                "Each component is tested separately against x_k > 0, but away "
                "from the boundary all sampled coordinates satisfy that test."
            ),
        },
        "timing": {
            "mean_t_final_t120": float(tfin_120.mean()),
            "mean_t_final_t400": float(tfin_400.mean()),
            "max_t_final_t120": float(tfin_120.max()),
            "max_t_final_t400": float(tfin_400.max()),
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary")
    print(f"  Grid points: {len(X)}")
    print(f"  Slow-simplex points: {inside_slow_simplex.sum()}/{len(X)}")
    print(f"  Converged by t=120: {conv_120.sum()}/{len(X)}")
    print(f"  Converged by t=400: {conv_400.sum()}/{len(X)}")
    print(f"  Gained by more time: {gained.sum()}")
    print(f"  Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
