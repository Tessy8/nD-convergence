"""
Separate experiment for the professor's SUM-GUARD variant in u coordinates.

Coordinates:
    u in [0, 2*pi)^3

Model:
    u_k is slow when u_k > 0 for all k and sum(u) < c_sum

Equivalently:
    FAST if (u_i < coord_guard) OR (u1 + u2 + u3 > c_sum)
    SLOW otherwise

This script compares two choices with the professor's coordinate guard:
    coord_guard = 0
    c_sum = 1 + 3*pi   (baseline remapped sum threshold)
    c_sum = 2*pi       (professor's requested sum threshold)

Outputs:
  - console summary
  - conv-basin/output/torus_0_2pi_sum_guard_2pi_analysis.json
"""

import json
import os
import numpy as np

PI = np.pi
TWO_PI = 2.0 * np.pi

# configuration
DELTA              = 0.5
DT                 = 0.05
T_TOL              = 1e-5
CONV_TOL           = 0.05
CONV_TIME          = 10.0
STANDARD_TMAX      = 120.0
LONG_TMAX          = 400.0

GRID_N             = 9
GRID_MARGIN        = 0.35
COORD_GUARD_U      = 0.0
SUM_GUARD_BASE_U   = 1.0 + 3.0 * PI
SUM_GUARD_TEST_U   = TWO_PI

OUTPUT_DIR         = "conv-basin/output"
OUTPUT_JSON        = os.path.join(OUTPUT_DIR, "torus_0_2pi_sum_guard_2pi_analysis.json")


def wrap_theta(x):
    return np.mod(np.asarray(x, dtype=float), TWO_PI)


def circular_mean_theta(x):
    x = np.asarray(x, dtype=float)
    mean = np.arctan2(np.sin(x).mean(), np.cos(x).mean())
    return float(np.mod(mean, TWO_PI))


def torus_residual_theta(x):
    center = circular_mean_theta(x)
    return ((np.asarray(x, dtype=float) - center + PI) % TWO_PI) - PI


def torus_spread_theta(x):
    return float(np.abs(torus_residual_theta(x)).max())


def vfield_theta(u, coord_guard_u, sum_guard_u, delta):
    """Hybrid vector field in u coordinates."""
    u = wrap_theta(u)
    return np.where((u < coord_guard_u) | (u.sum() > sum_guard_u), 1.0, 1.0 - delta)


def integrate_pc_theta(u0, coord_guard_u, sum_guard_u, delta, t_max, dt, t_tol,
                       conv_tol=0.05, conv_time=10.0):
    u = wrap_theta(u0)
    h = dt
    t = 0.0
    du0 = vfield_theta(u, coord_guard_u, sum_guard_u, delta)
    conv_for = 0.0
    converged = False

    while t < t_max:
        u_trial = u + h * du0
        du1 = vfield_theta(u_trial, coord_guard_u, sum_guard_u, delta)

        if not np.allclose(du1, du0) and h > t_tol:
            h /= 2.0
            continue

        u = wrap_theta(u + h * du0)
        t += h
        du0 = vfield_theta(u, coord_guard_u, sum_guard_u, delta)
        h = min(h * 1.5, dt)

        if torus_spread_theta(u) < CONV_TOL:
            conv_for += h
            if conv_for >= conv_time:
                converged = True
                break
        else:
            conv_for = 0.0

    return {
        "converged": bool(converged),
        "t_final": float(t),
    }


def build_grid():
    vals = np.linspace(GRID_MARGIN, TWO_PI - GRID_MARGIN, GRID_N)
    grid = np.meshgrid(vals, vals, vals, indexing="ij")
    X = np.stack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()], axis=1)
    return vals, X


def classify_grid(points, sum_guard_u, t_max, label):
    conv = np.zeros(len(points), dtype=bool)
    t_final = np.zeros(len(points), dtype=float)
    print(
        f"Classifying {len(points)} points for {label} "
        f"(coord guard={COORD_GUARD_U:.6f}, sum guard={sum_guard_u:.6f}, t={t_max:.0f})",
        flush=True,
    )

    for i, u0 in enumerate(points):
        res = integrate_pc_theta(
            u0,
            coord_guard_u=COORD_GUARD_U,
            sum_guard_u=sum_guard_u,
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
            print(f"  finished {i + 1}/{len(points)} points", flush=True)
    return conv, t_final


def sample_points(points, mask, limit=20):
    return [np.round(pt, 6).tolist() for pt in points[mask][:limit]]


def summarize_pair(points, conv_a, conv_b, label_a, label_b):
    both = conv_a & conv_b
    neither = (~conv_a) & (~conv_b)
    gained_b = (~conv_a) & conv_b
    lost_b = conv_a & (~conv_b)

    return {
        "counts": {
            f"converged_{label_a}": int(conv_a.sum()),
            f"converged_{label_b}": int(conv_b.sum()),
            "converged_both": int(both.sum()),
            "converged_neither": int(neither.sum()),
            f"nonconverged_{label_a}_that_converge_{label_b}": int(gained_b.sum()),
            f"nonconverged_{label_b}_that_converge_{label_a}": int(lost_b.sum()),
        },
        "fractions": {
            f"fraction_converged_{label_a}": float(conv_a.mean()),
            f"fraction_converged_{label_b}": float(conv_b.mean()),
            f"fraction_nonconverged_{label_a}_that_converge_{label_b}": (
                float(gained_b.sum() / (~conv_a).sum()) if (~conv_a).sum() else 0.0
            ),
            f"fraction_nonconverged_{label_b}_that_converge_{label_a}": (
                float(lost_b.sum() / (~conv_b).sum()) if (~conv_b).sum() else 0.0
            ),
        },
        "sample_points": {
            f"nonconverged_{label_a}_that_converge_{label_b}": sample_points(points, gained_b),
            f"nonconverged_{label_b}_that_converge_{label_a}": sample_points(points, lost_b),
            "converged_both": sample_points(points, both),
            "converged_neither": sample_points(points, neither),
        },
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vals, X = build_grid()

    base_120, base_t120 = classify_grid(X, SUM_GUARD_BASE_U, STANDARD_TMAX, "baseline_sum_guard")
    test_120, test_t120 = classify_grid(X, SUM_GUARD_TEST_U, STANDARD_TMAX, "sum_guard_2pi")
    base_400, base_t400 = classify_grid(X, SUM_GUARD_BASE_U, LONG_TMAX, "baseline_sum_guard")
    test_400, test_t400 = classify_grid(X, SUM_GUARD_TEST_U, LONG_TMAX, "sum_guard_2pi")

    inside_base_simplex = X.sum(axis=1) <= SUM_GUARD_BASE_U
    inside_test_simplex = X.sum(axis=1) <= SUM_GUARD_TEST_U

    summary = {
        "model": {
            "torus_domain": [0.0, TWO_PI],
            "dimension": 3,
            "delta": DELTA,
            "coordinate_guard_u": COORD_GUARD_U,
            "coordinate_guard_equation": "u_i = 0",
            "baseline_sum_guard_u": SUM_GUARD_BASE_U,
            "baseline_sum_guard_equation": "u1 + u2 + u3 = 1 + 3*pi",
            "test_sum_guard_u": SUM_GUARD_TEST_U,
            "test_sum_guard_equation": "u1 + u2 + u3 = 2*pi",
            "interpretation": (
                "Both runs use the professor's coordinate guard u_i > 0 in the "
                "interior; only the sum threshold changes."
            ),
        },
        "grid": {
            "n_per_axis": GRID_N,
            "margin": GRID_MARGIN,
            "total_points": int(len(X)),
            "axis_values": np.round(vals, 6).tolist(),
        },
        "geometry": {
            "points_inside_baseline_slow_simplex": int(inside_base_simplex.sum()),
            "points_inside_test_slow_simplex": int(inside_test_simplex.sum()),
            "fraction_inside_baseline_slow_simplex": float(inside_base_simplex.mean()),
            "fraction_inside_test_slow_simplex": float(inside_test_simplex.mean()),
            "expected_test_volume_fraction_simplex": float(1.0 / 6.0),
        },
        "standard_horizon_comparison": summarize_pair(
            X, base_120, test_120, "baseline_t120", "sumguard2pi_t120"
        ),
        "long_horizon_comparison": summarize_pair(
            X, base_400, test_400, "baseline_t400", "sumguard2pi_t400"
        ),
        "time_horizon_effect": {
            "baseline": summarize_pair(X, base_120, base_400, "baseline_t120", "baseline_t400"),
            "sumguard2pi": summarize_pair(X, test_120, test_400, "sumguard2pi_t120", "sumguard2pi_t400"),
        },
        "timing": {
            "baseline_mean_tfinal_t120": float(base_t120.mean()),
            "baseline_mean_tfinal_t400": float(base_t400.mean()),
            "sumguard2pi_mean_tfinal_t120": float(test_t120.mean()),
            "sumguard2pi_mean_tfinal_t400": float(test_t400.mean()),
        },
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary", flush=True)
    print(f"  Baseline sum guard, t=120: {base_120.sum()}/{len(X)}", flush=True)
    print(f"  Sum guard = 2pi, t=120:   {test_120.sum()}/{len(X)}", flush=True)
    print(f"  Baseline sum guard, t=400: {base_400.sum()}/{len(X)}", flush=True)
    print(f"  Sum guard = 2pi, t=400:    {test_400.sum()}/{len(X)}", flush=True)
    print(f"  Saved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
