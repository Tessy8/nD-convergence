"""
Fast nD coordinate-threshold sweep on [0, 2*pi)^D.

Two speedups over the original:
  1. Multiprocessing — the classify loop runs across all available CPU cores.
  2. Random sampling for D >= 6 — instead of a coarse structured grid
     (e.g. 3^8 = 6561 points with only 3 values per axis), draws N_RANDOM
     uniformly random points from [0, 2*pi)^D.  This gives far better
     coverage of the cube and a more reliable convergence fraction.

Usage:
    python3 torus_sweep_nD_fast.py            # D=8, random sampling
    python3 torus_sweep_nD_fast.py --dim 4    # D=4, structured grid
    python3 torus_sweep_nD_fast.py --dim 8 --n_random 10000
    python3 torus_sweep_nD_fast.py --dim 8 --grid   # force structured grid

For D <= 5 the code uses the same structured grid as before (fast enough).
For D >= 6 it defaults to random sampling unless --grid is passed.

Outputs:
    torus_sweep_D{D}_fast.json
"""

import argparse
import json
import os
import numpy as np
from multiprocessing import Pool, cpu_count

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
SUM_GUARD     = TWO_PI

# Structured grid: points per axis for each D (same as original)
GRID_N_BY_DIM = {3: 9, 4: 7, 5: 5, 6: 4, 7: 3, 8: 3}
GRID_MARGIN   = 0.35

# Random sampling: how many points for D >= 6 by default
N_RANDOM_DEFAULT = 5000

N_COARSE = 20
N_FINE   = 15

OUTPUT_DIR = "."   # save in current directory


# ── torus helpers ──────────────────────────────────────────────────────────
def wrap_theta(x):
    return np.mod(x, TWO_PI)

def circular_mean_theta(x):
    return np.mod(np.arctan2(np.sin(x).mean(), np.cos(x).mean()), TWO_PI)

def torus_spread_theta(x):
    center = circular_mean_theta(x)
    return float(np.abs(((x - center + PI) % TWO_PI) - PI).max())

def vfield(x, c, delta):
    slow = (x > c) & (x.sum() < SUM_GUARD)
    return np.where(slow, 1.0 - delta, 1.0)


# ── integrator ─────────────────────────────────────────────────────────────
def integrate(x0, c, delta, t_max, dt, t_tol, conv_tol, conv_time):
    x   = wrap_theta(x0)
    h   = dt
    t   = 0.0
    dx0 = vfield(x, c, delta)
    conv_for  = 0.0

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
                return True
        else:
            conv_for = 0.0
    return False


# ── worker for multiprocessing ─────────────────────────────────────────────
# Parameters are passed as a tuple alongside x0 — works correctly on Windows
# (spawn) and Linux/Mac (fork) because nothing relies on shared global state.

def _worker(args):
    x0, c, delta, t_max, dt, t_tol, conv_tol, conv_time = args
    return integrate(x0, c, delta, t_max, dt, t_tol, conv_tol, conv_time)

def classify_parallel(X, c, t_max, n_workers, label=""):
    print(f"  c={c:.5f}  n={len(X)}  workers={n_workers}  [{label}]",
          flush=True)
    tasks = [(x0, c, DELTA, t_max, DT, T_TOL, CONV_TOL, CONV_TIME) for x0 in X]
    with Pool(n_workers) as pool:
        results = pool.map(_worker, tasks, chunksize=max(1, len(X)//(n_workers*4)))
    return np.array(results, dtype=bool)


# ── grid builders ──────────────────────────────────────────────────────────
def build_structured_grid(D, grid_n, margin):
    vals  = np.linspace(margin, TWO_PI - margin, grid_n)
    grids = np.meshgrid(*([vals] * D), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)

def build_random_grid(D, n, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, TWO_PI, size=(n, D))


# ── main ───────────────────────────────────────────────────────────────────
def run(D, use_random, n_random, n_workers):
    C_FEASIBLE  = TWO_PI / D
    C_CONJ      = TWO_PI / (2 * D)   # conjecture: c_opt = pi/D
    OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"torus_sweep_D{D}_fast.json")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if use_random:
        X = build_random_grid(D, n_random)
        grid_desc = f"random uniform, n={n_random}"
    else:
        grid_n = GRID_N_BY_DIM.get(D, 3)
        X = build_structured_grid(D, grid_n, GRID_MARGIN)
        grid_desc = f"structured {grid_n}^{D}"

    N = len(X)

    print(f"\n{'='*60}")
    print(f"D={D}  {grid_desc}  total points={N}")
    print(f"Workers: {n_workers}")
    print(f"Feasibility boundary: c < 2pi/{D} = {C_FEASIBLE:.5f}")
    print(f"Conjecture c_opt = pi/{D} = {C_CONJ:.5f}")
    print(f"{'='*60}\n")

    # ── coarse sweep ───────────────────────────────────────────────────────
    C_COARSE = np.linspace(0.0, 0.95 * C_FEASIBLE, N_COARSE)
    print("=== Coarse sweep (t=120) ===")
    coarse_results = []
    for c in C_COARSE:
        inside = np.all(X > c, axis=1) & (X.sum(axis=1) < SUM_GUARD)
        conv   = classify_parallel(X, c, STANDARD_TMAX, n_workers, "coarse")
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
    c_lo   = max(0.0, best_c["c"] - step)
    c_hi   = min(0.95 * C_FEASIBLE, best_c["c"] + step)
    C_FINE = np.linspace(c_lo, c_hi, N_FINE)

    print(f"\n=== Fine sweep (t=120) around [{c_lo:.5f}, {c_hi:.5f}] ===")
    fine_results = []
    for c in C_FINE:
        inside = np.all(X > c, axis=1) & (X.sum(axis=1) < SUM_GUARD)
        conv   = classify_parallel(X, c, STANDARD_TMAX, n_workers, "fine")
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

    # ── long horizon ───────────────────────────────────────────────────────
    print(f"\n=== Long horizon (t=400) at c_opt={c_opt:.5f} and c=0 ===")
    conv_opt_400 = classify_parallel(X, c_opt, LONG_TMAX, n_workers, "opt_t400")
    conv_c0_120  = classify_parallel(X, 0.0, STANDARD_TMAX, n_workers, "c0_t120")
    conv_c0_400  = classify_parallel(X, 0.0, LONG_TMAX,     n_workers, "c0_t400")

    err = abs(c_opt - C_CONJ) / C_CONJ
    print(f"\nConjecture check: pi/{D} = {C_CONJ:.5f}, observed = {c_opt:.5f}, "
          f"error = {err:.1%}  ({'supported' if err < 0.20 else 'not supported'})")

    output = {
        "model": {
            "dimension": D,
            "torus_domain": [0.0, TWO_PI],
            "delta": DELTA,
            "sum_guard": SUM_GUARD,
            "rule": "dx_k slow iff x_k > c AND sum(x) < 2*pi",
            "feasibility_boundary_c": float(C_FEASIBLE),
            "conjectured_optimal_c": float(C_CONJ),
            "grid_description": grid_desc,
        },
        "grid": {"total_points": N, "description": grid_desc},
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
        "conjecture": {
            "predicted_c_opt": float(C_CONJ),
            "observed_c_opt":  float(c_opt),
            "relative_error":  float(err),
            "supported":       bool(err < 0.20),
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim",      type=int,  default=8)
    parser.add_argument("--n_random", type=int,  default=N_RANDOM_DEFAULT,
                        help="Number of random points for D>=6 (default 5000)")
    parser.add_argument("--grid",     action="store_true",
                        help="Force structured grid even for high D")
    parser.add_argument("--workers",  type=int,  default=0,
                        help="CPU workers (0 = all available)")
    args = parser.parse_args()

    use_random = (args.dim >= 6) and not args.grid
    n_workers  = args.workers if args.workers > 0 else cpu_count()

    run(args.dim, use_random, args.n_random, n_workers)