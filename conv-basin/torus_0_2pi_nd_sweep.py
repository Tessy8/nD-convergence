"""
Sweep the coordinate threshold c in the professor's rule on [0, 2*pi)^D.

Model:
    dx_k is slow iff x_k > c  AND  sum(x) < 2*pi
    dx_k is fast otherwise

Works for any D. Feasibility: slow region non-empty only when c < 2*pi/D.

Usage:
    python3 torus_sweep_nD.py          # runs D=8 by default
    python3 torus_sweep_nD.py --dim 3  # run D=3
    python3 torus_sweep_nD.py --dim 4  # run D=4

Outputs:
    conv-basin/output/torus_sweep_D{D}.json
"""

import argparse
import json
import os
import numpy as np

PI     = np.pi
TWO_PI = 2.0 * PI

# ── defaults ───────────────────────────────────────────────────────────────
DELTA         = 0.5
DT            = 0.05
T_TOL         = 1e-5
CONV_TOL      = 0.05
CONV_TIME     = 10.0
STANDARD_TMAX = 120.0
LONG_TMAX     = 400.0
SUM_GUARD     = TWO_PI

# Grid points per axis — reduce for high D to keep runtime manageable
# D=3: 9^3=729, D=4: 7^4=2401, D=8: 4^8=65536 (too slow) -> use 3^8=6561
GRID_N_BY_DIM = {3: 9, 4: 7, 5: 5, 6: 4, 7: 3, 8: 3}
GRID_MARGIN   = 0.35

N_COARSE = 20
N_FINE   = 15

OUTPUT_DIR = "conv-basin/output"


# ── torus helpers ──────────────────────────────────────────────────────────
def wrap_theta(x):
    return np.mod(np.asarray(x, dtype=float), TWO_PI)


def circular_mean_theta(x):
    x = np.asarray(x, dtype=float)
    return float(np.mod(np.arctan2(np.sin(x).mean(), np.cos(x).mean()), TWO_PI))


def torus_spread_theta(x):
    center   = circular_mean_theta(x)
    residuals = ((np.asarray(x, dtype=float) - center + PI) % TWO_PI) - PI
    return float(np.abs(residuals).max())


# ── vector field ───────────────────────────────────────────────────────────
def vfield(x, c, delta):
    x    = wrap_theta(x)
    slow = (x > c) & (x.sum() < SUM_GUARD)
    return np.where(slow, 1.0 - delta, 1.0)


# ── integrator ─────────────────────────────────────────────────────────────
def integrate(x0, c, delta, t_max, dt, t_tol, conv_tol, conv_time):
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


# ── grid ───────────────────────────────────────────────────────────────────
def build_grid(D, grid_n, margin):
    vals   = np.linspace(margin, TWO_PI - margin, grid_n)
    grids  = np.meshgrid(*([vals] * D), indexing="ij")
    X      = np.stack([g.ravel() for g in grids], axis=1)
    return vals, X


# ── classify ───────────────────────────────────────────────────────────────
def classify(X, c, t_max, D, label=""):
    conv = np.zeros(len(X), dtype=bool)
    prog = max(1, len(X) // 10)
    print(f"  c={c:.5f}  D={D}  t_max={t_max:.0f}  n={len(X)}  {label}",
          flush=True)
    for i, x0 in enumerate(X):
        conv[i] = integrate(x0, c, DELTA, t_max, DT, T_TOL, CONV_TOL, CONV_TIME)
        if (i + 1) % prog == 0:
            pct = 100.0 * conv[:i+1].sum() / (i + 1)
            print(f"    {i+1}/{len(X)}  running conv={pct:.1f}%", flush=True)
    return conv


# ── main ───────────────────────────────────────────────────────────────────
def run(D):
    C_FEASIBLE = TWO_PI / D      # slow region empty for c >= this
    GRID_N     = GRID_N_BY_DIM.get(D, 3)
    OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"torus_sweep_D{D}.json")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vals, X = build_grid(D, GRID_N, GRID_MARGIN)
    N = len(X)

    print(f"\n{'='*60}")
    print(f"D={D}  grid={GRID_N}^{D}={N} points")
    print(f"Feasibility boundary: c < 2pi/{D} = {C_FEASIBLE:.5f}")
    print(f"Sweep range: [0, {0.95*C_FEASIBLE:.5f}]")
    print(f"Theoretical optimal (conjecture): c = 2pi/(2D) = "
          f"{TWO_PI/(2*D):.5f}  (=pi/{D})")
    print(f"{'='*60}\n")

    # ── coarse sweep at t=120 ──────────────────────────────────────────────
    C_COARSE = np.linspace(0.0, 0.95 * C_FEASIBLE, N_COARSE)
    print("=== Coarse sweep (t=120) ===")
    coarse_results = []
    for c in C_COARSE:
        inside = np.all(X > c, axis=1) & (X.sum(axis=1) < SUM_GUARD)
        conv   = classify(X, c, STANDARD_TMAX, D, "coarse")
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

    best_c = max(coarse_results, key=lambda r: r["n_converged_t120"])
    print(f"\nCoarse best: c={best_c['c']:.5f}  "
          f"{best_c['n_converged_t120']}/{N} ({best_c['pct_converged_t120']:.1f}%)")

    # ── fine sweep ─────────────────────────────────────────────────────────
    step   = (0.95 * C_FEASIBLE) / N_COARSE
    c_lo   = max(0.0,             best_c["c"] - step)
    c_hi   = min(0.95*C_FEASIBLE, best_c["c"] + step)
    C_FINE = np.linspace(c_lo, c_hi, N_FINE)

    print(f"\n=== Fine sweep (t=120) around [{c_lo:.5f}, {c_hi:.5f}] ===")
    fine_results = []
    for c in C_FINE:
        inside = np.all(X > c, axis=1) & (X.sum(axis=1) < SUM_GUARD)
        conv   = classify(X, c, STANDARD_TMAX, D, "fine")
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
    c_opt = best_fine["c"]
    print(f"\nFine best: c={c_opt:.5f}  "
          f"{best_fine['n_converged_t120']}/{N} ({best_fine['pct_converged_t120']:.1f}%)")

    # ── long horizon at optimal c and c=0 ─────────────────────────────────
    print(f"\n=== Long horizon (t=400) at c_opt={c_opt:.5f} and c=0 ===")
    conv_opt_400 = classify(X, c_opt, LONG_TMAX, D, "opt_t400")
    conv_c0_120  = classify(X, 0.0,   STANDARD_TMAX, D, "c0_t120")
    conv_c0_400  = classify(X, 0.0,   LONG_TMAX, D, "c0_t400")

    print(f"\n  c=0       t=120: {int(conv_c0_120.sum())}/{N} "
          f"({100*conv_c0_120.mean():.1f}%)")
    print(f"  c=0       t=400: {int(conv_c0_400.sum())}/{N} "
          f"({100*conv_c0_400.mean():.1f}%)")
    print(f"  c={c_opt:.4f} t=120: {best_fine['n_converged_t120']}/{N} "
          f"({best_fine['pct_converged_t120']:.1f}%)")
    print(f"  c={c_opt:.4f} t=400: {int(conv_opt_400.sum())}/{N} "
          f"({100*conv_opt_400.mean():.1f}%)")

    # ── save ───────────────────────────────────────────────────────────────
    output = {
        "model": {
            "dimension": D,
            "torus_domain": [0.0, TWO_PI],
            "delta": DELTA,
            "sum_guard": SUM_GUARD,
            "rule": "dx_k slow iff x_k > c AND sum(x) < 2*pi",
            "feasibility_boundary_c": float(C_FEASIBLE),
            "conjectured_optimal_c": float(TWO_PI / (2 * D)),
            "note": "Predictor-corrector integrator with bisection at switching surfaces.",
        },
        "grid": {
            "n_per_axis": GRID_N,
            "margin": GRID_MARGIN,
            "total_points": N,
        },
        "coarse_sweep": coarse_results,
        "fine_sweep":   fine_results,
        "optimal": {
            "c":                  float(c_opt),
            "n_converged_t120":   best_fine["n_converged_t120"],
            "pct_converged_t120": best_fine["pct_converged_t120"],
            "n_converged_t400":   int(conv_opt_400.sum()),
            "pct_converged_t400": float(100.0 * conv_opt_400.mean()),
        },
        "comparison_with_c0": {
            "c0_t120": int(conv_c0_120.sum()),
            "c0_t400": int(conv_c0_400.sum()),
            "c0_pct_t120": float(100.0 * conv_c0_120.mean()),
            "c0_pct_t400": float(100.0 * conv_c0_400.mean()),
            "optimal_t120": best_fine["n_converged_t120"],
            "optimal_t400": int(conv_opt_400.sum()),
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=8,
                        help="Dimension D (default: 8)")
    args = parser.parse_args()
    run(args.dim)