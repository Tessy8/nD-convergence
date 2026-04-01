"""
Sweep the coordinate threshold c in the professor's rule on [0, 2*pi)^3.

Model:
    dx_k is slow iff x_k > c  AND  x_1 + x_2 + x_3 < 2*pi
    dx_k is fast otherwise

Classification uses the actual predictor-corrector integrator with event
detection — no closed-form approximations.

Feasibility: slow region is non-empty only when c < 2*pi/3 ≈ 2.094.
The sweep covers c in [0, 0.95 * 2*pi/3] with a coarse pass followed by
a fine refinement around the best value found.

Outputs:
  - console summary
  - conv-basin/output/torus_0_2pi_coordinate_guard_sweep.json
"""

import json
import os
import numpy as np

PI     = np.pi
TWO_PI = 2.0 * PI

# ── configuration ──────────────────────────────────────────────────────────
DELTA         = 0.5
DT            = 0.05
T_TOL         = 1e-5
CONV_TOL      = 0.05
CONV_TIME     = 10.0
STANDARD_TMAX = 120.0
LONG_TMAX     = 400.0

GRID_N        = 9          # 9^3 = 729 points
GRID_MARGIN   = 0.35
SUM_GUARD     = TWO_PI

# Feasibility boundary: slow region empty for c >= 2*pi/3
C_FEASIBLE    = TWO_PI / 3.0   # ≈ 2.094
N_COARSE      = 20
N_FINE        = 15

OUTPUT_DIR    = "conv-basin/output"
OUTPUT_JSON   = os.path.join(OUTPUT_DIR, "torus_0_2pi_coordinate_guard_sweep.json")


# ── torus helpers ──────────────────────────────────────────────────────────
def wrap_theta(x):
    """Wrap coordinates into [0, 2*pi)."""
    return np.mod(np.asarray(x, dtype=float), TWO_PI)


def circular_mean_theta(x):
    x = np.asarray(x, dtype=float)
    return float(np.mod(np.arctan2(np.sin(x).mean(), np.cos(x).mean()), TWO_PI))


def torus_spread_theta(x):
    center = circular_mean_theta(x)
    residuals = ((np.asarray(x, dtype=float) - center + PI) % TWO_PI) - PI
    return float(np.abs(residuals).max())


# ── vector field ───────────────────────────────────────────────────────────
def vfield(x, c, delta):
    """dx_k = 1-delta if x_k > c AND sum(x) < 2*pi, else 1."""
    x = wrap_theta(x)
    slow = (x > c) & (x.sum() < SUM_GUARD)
    return np.where(slow, 1.0 - delta, 1.0)


# ── integrator ─────────────────────────────────────────────────────────────
def integrate(x0, c, delta, t_max, dt, t_tol, conv_tol, conv_time):
    """Predictor-corrector with bisection at every switching surface."""
    x   = wrap_theta(x0)
    h   = dt
    t   = 0.0
    dx0 = vfield(x, c, delta)
    conv_for  = 0.0
    converged = False

    while t < t_max:
        x_trial = x + h * dx0
        dx1     = vfield(x_trial, c, delta)

        if not np.allclose(dx1, dx0) and h > t_tol:
            h /= 2.0
            continue

        x   = wrap_theta(x + h * dx0)
        t  += h
        dx0 = vfield(x, c, delta)
        h   = min(h * 1.5, dt)

        if torus_spread_theta(x) < conv_tol:
            conv_for += h
            if conv_for >= conv_time:
                converged = True
                break
        else:
            conv_for = 0.0

    return converged


# ── grid classification ────────────────────────────────────────────────────
def classify(X, c, t_max, label):
    conv = np.zeros(len(X), dtype=bool)
    prog = max(1, len(X) // 8)
    print(f"  c={c:.5f}  t_max={t_max:.0f}  label={label}", flush=True)
    for i, x0 in enumerate(X):
        conv[i] = integrate(x0, c, DELTA, t_max, DT, T_TOL, CONV_TOL, CONV_TIME)
        if (i + 1) % prog == 0:
            print(f"    {i+1}/{len(X)}", flush=True)
    return conv


# ── build grid ─────────────────────────────────────────────────────────────
vals = np.linspace(GRID_MARGIN, TWO_PI - GRID_MARGIN, GRID_N)
g    = np.meshgrid(vals, vals, vals, indexing="ij")
X    = np.stack([g[0].ravel(), g[1].ravel(), g[2].ravel()], axis=1)
N    = len(X)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Grid: {GRID_N}^3 = {N} points")
print(f"Feasibility boundary: c < 2pi/3 = {C_FEASIBLE:.5f}")
print(f"Sweep range: [0, {0.95*C_FEASIBLE:.5f}]\n")

# ── coarse sweep ───────────────────────────────────────────────────────────
C_COARSE = np.linspace(0.0, 0.95 * C_FEASIBLE, N_COARSE)

print("=== Coarse sweep (t=120) ===")
coarse_results = []
for c in C_COARSE:
    inside = np.all(X > c, axis=1) & (X.sum(axis=1) < SUM_GUARD)
    conv   = classify(X, c, STANDARD_TMAX, "coarse_t120")
    n      = int(conv.sum())
    pct    = 100.0 * n / N
    print(f"  c={c:.5f}  slow_frac={inside.mean():.3f}  "
          f"converge={n}/{N} ({pct:.1f}%)", flush=True)
    coarse_results.append({
        "c": float(c),
        "n_converged_t120": n,
        "pct_converged_t120": pct,
        "slow_simplex_fraction": float(inside.mean()),
    })

best_coarse = max(coarse_results, key=lambda r: r["n_converged_t120"])
print(f"\nCoarse best: c={best_coarse['c']:.5f}  "
      f"{best_coarse['n_converged_t120']}/{N} ({best_coarse['pct_converged_t120']:.1f}%)")

# ── fine sweep around best ─────────────────────────────────────────────────
step   = (0.95 * C_FEASIBLE) / N_COARSE
c_lo   = max(0.0,            best_coarse["c"] - step)
c_hi   = min(0.95 * C_FEASIBLE, best_coarse["c"] + step)
C_FINE = np.linspace(c_lo, c_hi, N_FINE)

print(f"\n=== Fine sweep (t=120) around [{c_lo:.5f}, {c_hi:.5f}] ===")
fine_results = []
for c in C_FINE:
    inside = np.all(X > c, axis=1) & (X.sum(axis=1) < SUM_GUARD)
    conv   = classify(X, c, STANDARD_TMAX, "fine_t120")
    n      = int(conv.sum())
    pct    = 100.0 * n / N
    print(f"  c={c:.5f}  converge={n}/{N} ({pct:.1f}%)", flush=True)
    fine_results.append({
        "c": float(c),
        "n_converged_t120": n,
        "pct_converged_t120": pct,
        "slow_simplex_fraction": float(inside.mean()),
    })

best_fine = max(fine_results, key=lambda r: r["n_converged_t120"])
print(f"\nFine best: c={best_fine['c']:.5f}  "
      f"{best_fine['n_converged_t120']}/{N} ({best_fine['pct_converged_t120']:.1f}%)")

# ── run best c at long horizon ─────────────────────────────────────────────
c_opt = best_fine["c"]
print(f"\n=== Long horizon (t=400) at optimal c={c_opt:.5f} and c=0 ===")

conv_opt_400 = classify(X, c_opt, LONG_TMAX, "optimal_t400")
conv_c0_400  = classify(X, 0.0,   LONG_TMAX, "c0_t400")
conv_opt_120 = np.array([r["n_converged_t120"] for r in fine_results
                          if abs(r["c"] - c_opt) < 1e-9], dtype=bool)

# re-classify c=0 at t=120 for comparison
conv_c0_120 = classify(X, 0.0, STANDARD_TMAX, "c0_t120")

print(f"  c=0      t=120: {int(conv_c0_120.sum())}/{N}")
print(f"  c=0      t=400: {int(conv_c0_400.sum())}/{N}")
print(f"  c={c_opt:.4f} t=120: {best_fine['n_converged_t120']}/{N}")
print(f"  c={c_opt:.4f} t=400: {int(conv_opt_400.sum())}/{N}")

# ── save ───────────────────────────────────────────────────────────────────
output = {
    "model": {
        "torus_domain":  [0.0, TWO_PI],
        "dimension":     3,
        "delta":         DELTA,
        "sum_guard":     SUM_GUARD,
        "rule":          "dx_k slow iff x_k > c AND sum(x) < 2*pi",
        "feasibility_boundary_c": float(C_FEASIBLE),
        "note": (
            "Classification uses the predictor-corrector integrator with "
            "bisection at switching surfaces. No closed-form approximations."
        ),
    },
    "grid": {
        "n_per_axis":   GRID_N,
        "margin":       GRID_MARGIN,
        "total_points": N,
    },
    "coarse_sweep": coarse_results,
    "fine_sweep":   fine_results,
    "optimal": {
        "c":                  float(c_opt),
        "n_converged_t120":   best_fine["n_converged_t120"],
        "pct_converged_t120": best_fine["pct_converged_t120"],
        "n_converged_t400":   int(conv_opt_400.sum()),
        "pct_converged_t400": float(100.0 * conv_opt_400.sum() / N),
    },
    "comparison_with_c0": {
        "c0_t120": int(conv_c0_120.sum()),
        "c0_t400": int(conv_c0_400.sum()),
        "optimal_t120": best_fine["n_converged_t120"],
        "optimal_t400": int(conv_opt_400.sum()),
    },
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved {OUTPUT_JSON}")