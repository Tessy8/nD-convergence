"""
Characterise the 18 non-converging points at optimal c ≈ π/3 on [0, 2π)³.

For each non-converging point this script:
  1. Integrates for a long time (t=1000) and records whether the point ever
     converges at extended horizon.
  2. Tracks the minimum torus-spread reached and when.
  3. Records whether the trajectory ever enters the slow simplex
     {x_k > c, sum x_k < 2π} and how many times.
  4. Saves the full trajectory (downsampled) for inspection.
  5. Prints a compact diagnostic table to stdout.

Outputs:
  - nonconverging_diagnostics.json    (full per-point data)
  - nonconverging_trajectories.npz    (trajectories, keyed by point index)
"""

import json
import numpy as np

PI     = np.pi
TWO_PI = 2.0 * PI

# ── same parameters as the 3D sweep ───────────────────────────────────────
C_OPT      = PI / 3.0      # ≈ 1.0472  (optimal from the sweep)
DELTA      = 0.5
DT         = 0.05
T_TOL      = 1e-5
CONV_TOL   = 0.05
CONV_TIME  = 10.0
SUM_GUARD  = TWO_PI

GRID_N      = 9
GRID_MARGIN = 0.35

STANDARD_TMAX = 120.0
LONG_TMAX     = 1000.0   # extended to see if these ever converge
STORE_EVERY   = 10       # save 1 in STORE_EVERY steps for the trajectory

OUTPUT_JSON = "nonconverging_diagnostics.json"
OUTPUT_NPZ  = "nonconverging_trajectories.npz"


# ── torus helpers ──────────────────────────────────────────────────────────
def wrap_theta(x):
    return np.mod(np.asarray(x, dtype=float), TWO_PI)

def circular_mean_theta(x):
    x = np.asarray(x, dtype=float)
    return float(np.mod(np.arctan2(np.sin(x).mean(), np.cos(x).mean()), TWO_PI))

def torus_spread_theta(x):
    center = circular_mean_theta(x)
    residuals = ((np.asarray(x, dtype=float) - center + PI) % TWO_PI) - PI
    return float(np.abs(residuals).max())

def in_slow_simplex(x, c):
    x = wrap_theta(x)
    return bool(np.all(x > c) and x.sum() < SUM_GUARD)

def vfield(x, c, delta):
    x = wrap_theta(x)
    slow = (x > c) & (x.sum() < SUM_GUARD)
    return np.where(slow, 1.0 - delta, 1.0)


# ── integrator with trajectory recording ───────────────────────────────────
def integrate_full(x0, c, delta, t_max, dt, t_tol, conv_tol, conv_time,
                   store_every=STORE_EVERY):
    x   = wrap_theta(x0)
    h   = dt
    t   = 0.0
    dx0 = vfield(x, c, delta)
    conv_for  = 0.0
    converged = False
    step = 0

    traj  = [x.copy()]
    times = [0.0]

    min_spread       = torus_spread_theta(x)
    t_min_spread     = 0.0
    simplex_entries  = 0
    in_simplex_prev  = in_slow_simplex(x, c)

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
        step += 1

        sp = torus_spread_theta(x)
        if sp < min_spread:
            min_spread   = sp
            t_min_spread = t

        in_s = in_slow_simplex(x, c)
        if in_s and not in_simplex_prev:
            simplex_entries += 1
        in_simplex_prev = in_s

        if step % store_every == 0:
            traj.append(x.copy())
            times.append(t)

        if sp < conv_tol:
            conv_for += h
            if conv_for >= conv_time:
                converged = True
                break
        else:
            conv_for = 0.0

    return {
        "converged":       converged,
        "t_final":         t,
        "min_spread":      min_spread,
        "t_min_spread":    t_min_spread,
        "simplex_entries": simplex_entries,
        "final_x":         x.tolist(),
        "traj":            np.array(traj),
        "times":           np.array(times),
    }


# ── build the same 3D grid ─────────────────────────────────────────────────
vals = np.linspace(GRID_MARGIN, TWO_PI - GRID_MARGIN, GRID_N)
g    = np.meshgrid(vals, vals, vals, indexing="ij")
X    = np.stack([g[0].ravel(), g[1].ravel(), g[2].ravel()], axis=1)
N    = len(X)

# ── quick pass at t=120 to identify non-converging points ─────────────────
print(f"Step 1: classify all {N} points at t={STANDARD_TMAX}, c={C_OPT:.6f} (π/3)")
def classify_quick(X, c):
    out = np.zeros(len(X), dtype=bool)
    prog = max(1, len(X) // 8)
    for i, x0 in enumerate(X):
        # lightweight integration
        x   = wrap_theta(x0)
        h   = DT
        t   = 0.0
        dx0 = vfield(x, c, DELTA)
        conv_for = 0.0
        while t < STANDARD_TMAX:
            x_trial = x + h * dx0
            dx1 = vfield(x_trial, c, DELTA)
            if not np.allclose(dx1, dx0) and h > T_TOL:
                h /= 2.0; continue
            x   = wrap_theta(x + h * dx0)
            t  += h
            dx0 = vfield(x, c, DELTA)
            h   = min(h * 1.5, DT)
            if torus_spread_theta(x) < CONV_TOL:
                conv_for += h
                if conv_for >= CONV_TIME:
                    out[i] = True; break
            else:
                conv_for = 0.0
        if (i + 1) % prog == 0:
            print(f"  {i+1}/{len(X)}", flush=True)
    return out

conv_t120 = classify_quick(X, C_OPT)
nc_idx    = np.where(~conv_t120)[0]
print(f"  Converged at t=120: {conv_t120.sum()}/{N}")
print(f"  Non-converging:     {len(nc_idx)}\n")

# ── characterise each non-converging point ─────────────────────────────────
print(f"Step 2: deep integration (t={LONG_TMAX}) of {len(nc_idx)} non-converging points\n")

diagnostics = []
traj_store  = {}

# header
print(f"{'idx':>5}  {'x0':^30}  {'conv?':^6}  "
      f"{'min_spread':^10}  {'t@min':^7}  {'entries':^7}  {'final_x':^30}")
print("-" * 105)

for rank, idx in enumerate(nc_idx):
    x0   = X[idx]
    res  = integrate_full(x0, C_OPT, DELTA, LONG_TMAX, DT, T_TOL,
                          CONV_TOL, CONV_TIME)

    # format
    x0_str = "[" + ", ".join(f"{v:.3f}" for v in x0) + "]"
    fx_str = "[" + ", ".join(f"{v:.3f}" for v in res["final_x"]) + "]"
    print(f"{idx:>5}  {x0_str:^30}  {'YES' if res['converged'] else 'NO':^6}  "
          f"{res['min_spread']:^10.4f}  {res['t_min_spread']:^7.1f}  "
          f"{res['simplex_entries']:^7}  {fx_str:^30}")

    # symmetry: is x0 close to a coordinate permutation of a converging structure?
    diffs = np.sort(x0) - np.sort(x0).mean()  # deviation from uniform after sort
    diagnostics.append({
        "grid_index":      int(idx),
        "x0":              x0.tolist(),
        "x0_sorted":       np.sort(x0).tolist(),
        "x0_spread":       float(torus_spread_theta(x0)),
        "x0_sum":          float(x0.sum()),
        "x0_in_simplex":   bool(in_slow_simplex(x0, C_OPT)),
        "converged_t1000": bool(res["converged"]),
        "t_final":         float(res["t_final"]),
        "min_spread":      float(res["min_spread"]),
        "t_min_spread":    float(res["t_min_spread"]),
        "simplex_entries": int(res["simplex_entries"]),
        "final_x":         res["final_x"],
    })
    traj_store[f"traj_{idx}"]  = res["traj"]
    traj_store[f"times_{idx}"] = res["times"]

# ── summary ────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
n_conv_1000  = sum(d["converged_t1000"]  for d in diagnostics)
n_ever_simp  = sum(d["simplex_entries"] > 0 for d in diagnostics)
spreads      = [d["min_spread"] for d in diagnostics]
print(f"  Converged by t=1000 :  {n_conv_1000}/{len(diagnostics)}")
print(f"  Ever entered simplex:  {n_ever_simp}/{len(diagnostics)}")
print(f"  Min spread achieved  :  min={min(spreads):.4f}  max={max(spreads):.4f}")

# symmetry check: are all 18 points related by permutation?
x0s = np.array([d["x0"] for d in diagnostics])
x0s_sorted = np.sort(x0s, axis=1)
unique_sorted = np.unique(x0s_sorted, axis=0)
print(f"  Unique sorted x0 vectors: {len(unique_sorted)} "
      f"({'all permutation-related' if len(unique_sorted)==1 else 'distinct orbits'})")
if len(unique_sorted) <= 6:
    for v in unique_sorted:
        print(f"    {v}")

# coordinate values present
print(f"\n  Grid values used: {np.round(vals, 4)}")
print(f"  Non-converging x0 coordinates (unique values):")
uc = np.unique(np.round(x0s, 6))
print(f"    {uc}")

# ── save ───────────────────────────────────────────────────────────────────
with open(OUTPUT_JSON, "w") as f:
    json.dump({
        "parameters": {
            "c_opt": C_OPT,
            "c_opt_label": "pi/3",
            "delta": DELTA,
            "sum_guard": SUM_GUARD,
            "t_standard": STANDARD_TMAX,
            "t_extended": LONG_TMAX,
            "grid_n": GRID_N,
            "grid_margin": GRID_MARGIN,
        },
        "n_nonconverging_t120": len(nc_idx),
        "n_converged_t1000": n_conv_1000,
        "points": diagnostics,
    }, f, indent=2)
print(f"\nSaved {OUTPUT_JSON}")

np.savez_compressed(OUTPUT_NPZ, **traj_store)
print(f"Saved {OUTPUT_NPZ}")
print("\nDone.")
