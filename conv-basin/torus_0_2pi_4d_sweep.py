"""
4D coordinate-guard sweep on [0, 2*pi)^4.

Model (professor's rule, D=4):
    dx_k is slow iff x_k > c  AND  sum_k x_k < 2*pi
    dx_k is fast otherwise

Feasibility: slow region is non-empty only when c < 2*pi/4 = pi/2 ≈ 1.5708.
The sweep covers c in [0, 0.95 * pi/2].

Hypothesis being tested:
    Optimal c = 2*pi / (2*D) = pi/4 ≈ 0.7854  (D=4)
    (generalising the D=3 result c_opt ≈ pi/3 = 2*pi/(2*3))

Grid: 7^4 = 2401 points  (reduced from 9^4=6561 for speed; increase if time permits)

Outputs:
  - console summary with the convergence table
  - torus_0_2pi_4d_sweep.json  (same schema as the 3D sweep)
"""

import json
import os
import numpy as np

PI     = np.pi
TWO_PI = 2.0 * PI

# ── configuration ──────────────────────────────────────────────────────────
DIM           = 4
DELTA         = 0.5
DT            = 0.05
T_TOL         = 1e-5
CONV_TOL      = 0.05
CONV_TIME     = 10.0
STANDARD_TMAX = 120.0
LONG_TMAX     = 400.0

# 7^4 = 2401; raise to 8 (4096) or 9 (6561) for finer resolution if time permits
GRID_N        = 7
GRID_MARGIN   = 0.40
SUM_GUARD     = TWO_PI

# Feasibility boundary: slow region empty for c >= 2*pi/D
C_FEASIBLE    = TWO_PI / DIM        # pi/2 ≈ 1.5708
N_COARSE      = 16
N_FINE        = 12

OUTPUT_JSON   = "torus_0_2pi_4d_sweep.json"

# Hypothesis: optimal c = 2*pi/(2*D)
C_HYPOTHESIS  = TWO_PI / (2 * DIM)  # pi/4 ≈ 0.7854


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
    prog = max(1, len(X) // 10)
    print(f"  c={c:.5f}  t_max={t_max:.0f}  [{label}]", flush=True)
    for i, x0 in enumerate(X):
        conv[i] = integrate(x0, c, DELTA, t_max, DT, T_TOL, CONV_TOL, CONV_TIME)
        if (i + 1) % prog == 0:
            print(f"    {i+1}/{len(X)}", flush=True)
    return conv


# ── build 4D grid ──────────────────────────────────────────────────────────
vals   = np.linspace(GRID_MARGIN, TWO_PI - GRID_MARGIN, GRID_N)
grids  = np.meshgrid(*([vals] * DIM), indexing="ij")
X      = np.stack([g.ravel() for g in grids], axis=1)
N      = len(X)

print(f"Dimension: {DIM}")
print(f"Grid: {GRID_N}^{DIM} = {N} points")
print(f"Feasibility boundary: c < 2pi/{DIM} = {C_FEASIBLE:.5f}")
print(f"Hypothesis: c_opt = 2pi/(2*{DIM}) = {C_HYPOTHESIS:.5f}")
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
    print(f"  c={c:.5f}  slow_frac={inside.mean():.4f}  "
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
print(f"Hypothesis c={C_HYPOTHESIS:.5f} {'CONFIRMED' if abs(best_coarse['c'] - C_HYPOTHESIS) < 0.15 else 'DIFFERS'}")

# ── fine sweep around best ─────────────────────────────────────────────────
step   = (0.95 * C_FEASIBLE) / N_COARSE
c_lo   = max(0.0,             best_coarse["c"] - step)
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
conv_c0_120  = classify(X, 0.0, STANDARD_TMAX, "c0_t120")
conv_c0_400  = classify(X, 0.0, LONG_TMAX,     "c0_t400")

print(f"  c=0        t=120: {int(conv_c0_120.sum())}/{N}")
print(f"  c=0        t=400: {int(conv_c0_400.sum())}/{N}")
print(f"  c={c_opt:.4f}  t=120: {best_fine['n_converged_t120']}/{N}")
print(f"  c={c_opt:.4f}  t=400: {int(conv_opt_400.sum())}/{N}")

# ── hypothesis check ───────────────────────────────────────────────────────
print(f"\n=== Hypothesis check ===")
print(f"Predicted optimal c = 2pi/(2*D) = {C_HYPOTHESIS:.6f}")
print(f"Observed  optimal c =             {c_opt:.6f}")
err = abs(c_opt - C_HYPOTHESIS) / C_HYPOTHESIS
print(f"Relative error: {err:.2%}  →  {'SUPPORTED' if err < 0.15 else 'NOT SUPPORTED'}")

# ── save ───────────────────────────────────────────────────────────────────
output = {
    "model": {
        "torus_domain":  [0.0, TWO_PI],
        "dimension":     DIM,
        "delta":         DELTA,
        "sum_guard":     SUM_GUARD,
        "rule":          "dx_k slow iff x_k > c AND sum(x) < 2*pi",
        "feasibility_boundary_c": float(C_FEASIBLE),
        "hypothesis_c":  float(C_HYPOTHESIS),
        "note": (
            "Classification uses the predictor-corrector integrator with "
            "bisection at switching surfaces. Hypothesis: c_opt = 2pi/(2D)."
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
    "hypothesis": {
        "predicted_c_opt": float(C_HYPOTHESIS),
        "observed_c_opt":  float(c_opt),
        "relative_error":  float(err),
        "supported":       bool(err < 0.15),
    },
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved {OUTPUT_JSON}")
