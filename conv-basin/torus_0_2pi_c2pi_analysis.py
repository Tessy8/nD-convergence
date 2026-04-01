"""
Sweep the coordinate threshold c in the professor's rule on [0, 2*pi)^3.

Model:
    dx_k is slow iff x_k > c and x_1 + x_2 + x_3 < 2*pi
    dx_k is fast otherwise

The script evaluates a grid of candidate c values, then reports which c gives
the largest converging fraction at the standard and long horizons.

Outputs:
  - console summary
  - conv-basin/output/torus_0_2pi_coordinate_guard_sweep.json
"""

import json
import os
import numpy as np

PI = np.pi
TWO_PI = 2.0 * np.pi

DELTA         = 0.5
DT            = 0.05
T_TOL         = 1e-5
CONV_TOL      = 0.05
CONV_TIME     = 10.0
STANDARD_TMAX = 120.0
LONG_TMAX     = 400.0

GRID_N        = 9
GRID_MARGIN   = 0.35
SUM_GUARD_U   = TWO_PI

# Include the previous c=0 case, a uniform sweep across the interval, and c=2*pi.
C_VALUES      = np.concatenate([np.linspace(0.0, TWO_PI, 17), [TWO_PI]])

OUTPUT_DIR    = "conv-basin/output"
OUTPUT_JSON   = os.path.join(OUTPUT_DIR, "torus_0_2pi_coordinate_guard_sweep.json")


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


def build_grid():
    vals = np.linspace(GRID_MARGIN, TWO_PI - GRID_MARGIN, GRID_N)
    grid = np.meshgrid(vals, vals, vals, indexing="ij")
    points = np.stack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()], axis=1)
    return vals, points


def state_at_sum_guard(points, coord_guard_x, sum_guard):
    """
    Closed-form state when the trajectory first reaches the sum guard.

    Before sum(x) hits 2*pi, coordinates with x_k <= c move at speed 1 while
    coordinates with x_k > c move at speed 1-delta. After the sum guard is hit,
    every coordinate moves at the same fast speed, so torus offsets are frozen.
    """
    points = np.asarray(points, dtype=float)
    out = np.empty_like(points)
    for i, x0 in enumerate(points):
        s0 = float(x0.sum())
        if s0 >= sum_guard:
            out[i] = x0
            continue

        a = np.maximum(0.0, coord_guard_x - x0)
        a_sorted = np.sort(a)
        prev = 0.0
        base = s0
        active_fast = 3
        hit_time = None

        for nxt in a_sorted:
            slope = 3.0 * (1.0 - DELTA) + DELTA * active_fast
            if nxt > prev:
                if base + slope * (nxt - prev) >= sum_guard - 1e-12:
                    hit_time = prev + (sum_guard - base) / slope
                    break
                base += slope * (nxt - prev)
                prev = nxt
            active_fast -= 1

        if hit_time is None:
            slope = 3.0 * (1.0 - DELTA)
            hit_time = prev + (sum_guard - base) / slope

        out[i] = x0 + (1.0 - DELTA) * hit_time + DELTA * np.minimum(hit_time, a)
    return out


def sample_points(points, mask, limit=12):
    return [np.round(pt, 6).tolist() for pt in points[mask][:limit]]


def candidate_summary(points, coord_guard_x):
    slow_region = np.all(points > coord_guard_x, axis=1) & (points.sum(axis=1) < SUM_GUARD_U)
    state_on_guard = state_at_sum_guard(points, coord_guard_x, SUM_GUARD_U)
    spread = np.array([torus_spread_theta(x) for x in state_on_guard])
    conv = spread < CONV_TOL
    return {
        "c": float(coord_guard_x),
        "counts": {
            "points_in_slow_region": int(slow_region.sum()),
            "converged_t120": int(conv.sum()),
            "converged_t400": int(conv.sum()),
        },
        "fractions": {
            "fraction_in_slow_region": float(slow_region.mean()),
            "fraction_converged_t120": float(conv.mean()),
            "fraction_converged_t400": float(conv.mean()),
        },
        "timing": {
            "mean_t_final_t120": float(STANDARD_TMAX),
            "mean_t_final_t400": float(LONG_TMAX),
        },
        "samples": {
            "converged_t120": sample_points(points, conv),
            "converged_t400": sample_points(points, conv),
        },
    }


def best_entry(results, key):
    best = max(results, key=lambda item: (item["fractions"][key], -item["c"]))
    return {
        "c": best["c"],
        "fraction": best["fractions"][key],
        "count": best["counts"]["converged_t120" if key.endswith("t120") else "converged_t400"],
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vals, points = build_grid()

    results = [candidate_summary(points, c) for c in C_VALUES]

    summary = {
        "model": {
            "model_name": "literal_professor_coordinate_guard_sweep",
            "torus_domain": [0.0, TWO_PI],
            "dimension": 3,
            "delta": DELTA,
            "dt": DT,
            "t_tol": T_TOL,
            "conv_tol": CONV_TOL,
            "conv_time": CONV_TIME,
            "sum_guard_u": SUM_GUARD_U,
            "equation": "dx_k is slow iff x_k > c and x1 + x2 + x3 < 2*pi; fast otherwise",
            "evaluation_method": (
                "event-based closed form to the first time x1+x2+x3 reaches 2*pi; "
                "after that all coordinates move with the same speed so torus offsets are frozen"
            ),
            "c_values": np.round(C_VALUES, 6).tolist(),
        },
        "grid": {
            "n_per_axis": GRID_N,
            "margin": GRID_MARGIN,
            "total_points": int(len(points)),
            "axis_values": np.round(vals, 6).tolist(),
        },
        "best": {
            "t120": best_entry(results, "fraction_converged_t120"),
            "t400": best_entry(results, "fraction_converged_t400"),
        },
        "results": results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    best_t120 = summary["best"]["t120"]
    best_t400 = summary["best"]["t400"]
    print("\nBest c values", flush=True)
    print(
        f"  t=120: c={best_t120['c']:.6f}, "
        f"converged={best_t120['count']}/{len(points)} "
        f"({best_t120['fraction']:.4f})",
        flush=True,
    )
    print(
        f"  t=400: c={best_t400['c']:.6f}, "
        f"converged={best_t400['count']}/{len(points)} "
        f"({best_t400['fraction']:.4f})",
        flush=True,
    )
    print(f"  Saved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
