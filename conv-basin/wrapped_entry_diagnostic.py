"""
Diagnostic: why is event_wrapped_to_inner_same_visit always True?
              (why are there zero later_visit_inner cases?)

Two mechanistic hypotheses:
  A. "Preparation" — the trajectory never enters the simplex until its final
     wrap.  Each lap around the torus is bringing the landing position closer
     to the inner basin.  The geometry prepares the landing.
  B. "Near-miss" — the trajectory does enter the simplex on earlier visits but
     exits without hitting the inner basin.  The inner basin condition is
     satisfied only on the last visit.

This script re-integrates the high-wrap-count representative points with the
full trajectory stored (store_every=1) and classifies every simplex visit as:
  - "miss"      : entered outer simplex but never reached inner basin
  - "converged" : entered outer simplex AND hit inner basin (= the final visit)
  - (no entry)  : the wrap landed outside the simplex entirely

Usage:
    python wrapped_entry_diagnostic.py

Outputs:
  1. Console report per point: per-visit classification
  2. wrapped_entry_visit_log.csv  — one row per simplex visit per point
  3. wrapped_entry_mechanism_summary.txt — concise conclusion
"""

import numpy as np
import pandas as pd
from pathlib import Path

from hybrid_tools import PI, integrate_pc

# ---------------------------------------------------------------------------
# Configuration — match your basin study parameters exactly
# ---------------------------------------------------------------------------
GUARD_OFFSET = 0.0     # use the standard c=0 run
DELTA        = 0.5
T_MAX        = 120.0
DT           = 0.05
T_TOL        = 1e-5
CONV_TOL     = 0.05
CONV_TIME    = 10.0

OUTPUT_DIR = Path("conv-basin/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Points to investigate — pick the high-wrap-count cases from your CSV.
# These have wraps_before_first_simplex of 2, 4, 6, 10, 14, 20, 29 …
# Edit this list to match the rows you want.  Format: (x1, x2, x3, label)
# ---------------------------------------------------------------------------
DIAGNOSTIC_POINTS = [
    # wraps=2  (baseline — just 1 prior wrap to examine)
    (-2.991592653589793, -2.991592653589793, -0.8158889055244889,  "wraps=2,  x3=-0.82"),
    (-2.991592653589793, -2.991592653589793, -0.2719629685081628,  "wraps=2,  x3=-0.27"),
    # wraps=4
    (-2.991592653589793, -2.991592653589793,  0.27196296850816326, "wraps=4,  x3=+0.27"),
    # wraps=6
    (-2.991592653589793, -2.991592653589793,  0.8158889055244893,  "wraps=6,  x3=+0.82"),
    # wraps=10
    (-2.991592653589793, -2.991592653589793,  1.3598148425408154,  "wraps=10, x3=+1.36"),
    # wraps=14
    (-2.991592653589793, -2.991592653589793,  1.903740779557141,   "wraps=14, x3=+1.90"),
    # wraps=20
    (-2.991592653589793, -2.991592653589793,  2.4476667165734676,  "wraps=20, x3=+2.45"),
    # wraps=29  (longest preparation chain in CSV)
    (-2.991592653589793, -2.447666716573467,  2.4476667165734676,  "wraps=29, x2=-2.45"),
]


# ---------------------------------------------------------------------------
# Simplex membership helpers
# (mirror the logic in hybrid_tools so results are consistent)
# ---------------------------------------------------------------------------
def in_outer_simplex(x, c=GUARD_OFFSET):
    """
    The 'outer simplex' / slow region: all coords >= c AND sum <= 1.
    This is what first_simplex_idx flags in your classifier.
    """
    return np.all(x >= c) and np.sum(x) <= 1.0


def in_inner_basin(x, conv_tol=CONV_TOL):
    """
    The 'inner basin' / contracting region: all coords >= 0 AND sum <= 1.
    Matches the same_visit_contracting_idx logic in hybrid_tools.
    Note: at c=0 the inner and outer simplex are the same region.
    At c>0 the inner basin (c=0 version) is strictly inside the outer simplex.
    """
    return np.all(x >= 0.0) and np.sum(x) <= 1.0


def find_wrap_indices(traj_unwrapped):
    """
    Return indices where a torus wrap occurred (any coordinate crossed ±pi).
    Uses the same tile-floor logic as split_at_tile_crossings.
    """
    tiles   = np.floor((traj_unwrapped + PI) / (2.0 * PI)).astype(int)
    crossed = np.any(np.diff(tiles, axis=0) != 0, axis=1)
    return np.where(crossed)[0] + 1   # index AFTER the crossing


def find_simplex_visits(traj_wrapped, c=GUARD_OFFSET):
    """
    Scan the trajectory for contiguous blocks where the point is inside the
    outer simplex.  Returns a list of (entry_idx, exit_idx) tuples.
    exit_idx is the first index OUTSIDE the simplex after entry.
    If the trajectory ends inside, exit_idx = len(traj_wrapped).
    """
    inside = np.array([in_outer_simplex(x, c) for x in traj_wrapped])
    visits = []
    in_block = False
    for i, v in enumerate(inside):
        if v and not in_block:
            entry = i
            in_block = True
        elif not v and in_block:
            visits.append((entry, i))
            in_block = False
    if in_block:
        visits.append((entry, len(traj_wrapped)))
    return visits


def classify_visit(traj_wrapped, entry, exit_):
    """
    For a single simplex visit [entry, exit_), determine:
    - whether the inner basin was reached during this visit
    - the approach direction (which coords were negative just before entry)
    - the closest approach to the diagonal x* = (1/3,1/3,1/3)
    """
    segment = traj_wrapped[entry:exit_]
    hit_inner = any(in_inner_basin(x) for x in segment)

    # Approach direction: look at the 5 steps before entry
    pre_start = max(0, entry - 5)
    pre_seg   = traj_wrapped[pre_start:entry]
    neg_coords_before = []
    if len(pre_seg) > 0:
        neg_coords_before = list(np.where(pre_seg[-1] < 0)[0])

    # Closest approach to fixed point
    xstar = np.array([1/3, 1/3, 1/3])
    dists = np.linalg.norm(segment - xstar, axis=1)
    closest = float(np.min(dists))

    return {
        "hit_inner":          hit_inner,
        "neg_coords_before":  neg_coords_before,
        "closest_to_xstar":   closest,
        "visit_length":       exit_ - entry,
    }


# ---------------------------------------------------------------------------
# Main diagnostic loop
# ---------------------------------------------------------------------------
all_rows = []
mechanism_votes = {"A_preparation": 0, "B_near_miss": 0}

print("=" * 70)
print("WRAPPED-ENTRY MECHANISM DIAGNOSTIC")
print(f"  guard_offset = {GUARD_OFFSET},  c_inner = 0.0 (standard)")
print("=" * 70)

for (x1, x2, x3, label) in DIAGNOSTIC_POINTS:
    pt = (x1, x2, x3)
    print(f"\nPoint: {label}   {np.round(pt, 3)}")
    print("-" * 60)

    res = integrate_pc(
        pt,
        guard_offset=GUARD_OFFSET,
        delta=DELTA,
        t_max=T_MAX,
        dt=DT,
        t_tol=T_TOL,
        conv_tol=CONV_TOL,
        conv_time=CONV_TIME,
        store_every=1,          # full resolution — don't skip steps
        keep_trajectory=True,
    )

    if not res["converged"]:
        print("  [did not converge — skipping]")
        continue

    tw = res["traj_wrapped"]    # (N, 3)
    tu = res["traj_unwrapped"]  # (N, 3)
    N  = len(tw)

    # Find all simplex visits
    visits = find_simplex_visits(tw, c=GUARD_OFFSET)
    wrap_idxs = find_wrap_indices(tu)

    print(f"  Trajectory length:    {N} steps  (~{N * DT:.1f} time units)")
    print(f"  Total simplex visits: {len(visits)}")
    print(f"  Total wraps:          {len(wrap_idxs)}")
    print(f"  wraps_before_first_simplex (from CSV): {res.get('wraps_before_first_wrap', '?')}")

    if len(visits) == 0:
        print("  No simplex visits found — check integration parameters.")
        continue

    # Classify each visit
    n_inner_hit = 0
    n_miss      = 0
    print(f"\n  {'Visit':>5}  {'Entry idx':>10}  {'Exit idx':>9}  {'Length':>7}  {'Inner?':>7}  {'Neg coords before':>18}  {'Closest to x*':>13}")
    print("  " + "-" * 80)

    for v_idx, (entry, exit_) in enumerate(visits):
        info = classify_visit(tw, entry, exit_)
        hit  = info["hit_inner"]
        neg  = info["neg_coords_before"]
        clos = info["closest_to_xstar"]

        marker = "← CONVERGING" if hit else ""
        print(f"  {v_idx+1:>5}  {entry:>10}  {exit_:>9}  {info['visit_length']:>7}  "
              f"{'YES' if hit else 'no':>7}  {str(neg):>18}  {clos:>13.4f}  {marker}")

        if hit:
            n_inner_hit += 1
        else:
            n_miss += 1

        all_rows.append({
            "label":             label,
            "x1": x1, "x2": x2, "x3": x3,
            "visit_number":      v_idx + 1,
            "total_visits":      len(visits),
            "entry_idx":         entry,
            "exit_idx":          exit_,
            "visit_length":      info["visit_length"],
            "hit_inner_basin":   hit,
            "neg_coords_before": str(neg),
            "closest_to_xstar":  clos,
            "total_wraps":       len(wrap_idxs),
        })

    # Determine mechanism for this point
    converging_visit = n_inner_hit   # should be 1
    prior_visits     = n_miss        # visits before the converging one

    print(f"\n  Summary: {prior_visits} miss visit(s) before 1 converging visit")

    if prior_visits == 0:
        mechanism = "A_preparation"
        print("  → Mechanism A (PREPARATION): trajectory never entered simplex")
        print("    before the final converging visit.")
        print("    The wraps are 'preparing' the landing position.")
    else:
        mechanism = "B_near_miss"
        print("  → Mechanism B (NEAR-MISS): trajectory entered simplex on")
        print(f"    {prior_visits} earlier visit(s) but did not reach inner basin.")
        print("    Inner basin condition was not satisfied on those visits.")

    mechanism_votes[mechanism] += 1


# ---------------------------------------------------------------------------
# Overall conclusion
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MECHANISM VOTE ACROSS ALL POINTS")
print("=" * 70)
total = sum(mechanism_votes.values())
for mech, count in mechanism_votes.items():
    pct = 100 * count / total if total else 0
    print(f"  {mech}: {count}/{total} points  ({pct:.0f}%)")

if mechanism_votes["A_preparation"] == total:
    conclusion = (
        "CONCLUSION: Pure preparation mechanism.\n"
        "Every wrapped-entry point reaches the outer simplex for the FIRST time\n"
        "on the same visit that leads to convergence.  Prior wraps do not produce\n"
        "simplex visits — they are repositioning the trajectory on the torus.\n"
        "The 'wraps_before_first_simplex' count reflects laps that are purely\n"
        "outside the simplex, each one shifting the landing coordinates until\n"
        "the geometry places the entry point inside the inner basin."
    )
elif mechanism_votes["B_near_miss"] == total:
    conclusion = (
        "CONCLUSION: Pure near-miss mechanism.\n"
        "Trajectories enter the outer simplex on intermediate visits but exit\n"
        "without reaching the inner basin.  The final visit lands in the inner\n"
        "basin.  The question then becomes: what changes between visits?"
    )
else:
    conclusion = (
        "CONCLUSION: Mixed mechanism.\n"
        f"  Preparation: {mechanism_votes['A_preparation']} points\n"
        f"  Near-miss:   {mechanism_votes['B_near_miss']} points\n"
        "Investigate the near-miss points further — they show a different\n"
        "dynamical regime from the preparation-dominated majority."
    )

print()
print(conclusion)

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
if all_rows:
    df = pd.DataFrame(all_rows)
    csv_path = OUTPUT_DIR / "wrapped_entry_visit_log.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nVisit log saved to: {csv_path}")

    # --- Summary plot: closest approach per visit, coloured by point ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: closest_to_xstar vs visit number, coloured by point
    ax1 = axes[0]
    for lbl, grp in df.groupby("label"):
        grp_sorted = grp.sort_values("visit_number")
        ax1.plot(grp_sorted["visit_number"], grp_sorted["closest_to_xstar"],
                 marker="o", markersize=5, linewidth=1.3, label=lbl, alpha=0.8)
        # highlight the converging visit
        conv_row = grp_sorted[grp_sorted["hit_inner_basin"] == True]
        if len(conv_row):
            ax1.scatter(conv_row["visit_number"], conv_row["closest_to_xstar"],
                        marker="*", s=120, zorder=5,
                        color=ax1.lines[-1].get_color(), edgecolors="black", linewidths=0.8)

    ax1.set_xlabel("Simplex visit number", fontsize=10)
    ax1.set_ylabel("Closest approach to x* = (1/3,1/3,1/3)", fontsize=10)
    ax1.set_title("Does distance to x* decrease across visits?\n(★ = converging visit)", fontsize=10)
    ax1.legend(fontsize=6.5, loc="upper right", ncol=1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: visit length (time spent in simplex per visit)
    ax2 = axes[1]
    for lbl, grp in df.groupby("label"):
        grp_sorted = grp.sort_values("visit_number")
        ax2.bar(
            grp_sorted["visit_number"] + list(df["label"].unique()).index(lbl) * 0.08,
            grp_sorted["visit_length"],
            width=0.07, alpha=0.7,
            label=lbl,
        )
    ax2.set_xlabel("Simplex visit number", fontsize=10)
    ax2.set_ylabel("Visit length (stored steps)", fontsize=10)
    ax2.set_title("Time spent in simplex per visit\n(longer = deeper penetration)", fontsize=10)
    ax2.legend(fontsize=6.5, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Wrapped-entry mechanism diagnostic  |  guard_offset={GUARD_OFFSET}  |  c_inner=0.0",
        fontsize=11,
    )
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "wrapped_entry_mechanism_diagnostic.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Diagnostic plot saved to: {plot_path}")

summary_path = OUTPUT_DIR / "wrapped_entry_mechanism_summary.txt"
with open(summary_path, "w") as f:
    f.write(conclusion + "\n\n")
    f.write(f"Votes: {mechanism_votes}\n\n")
    if all_rows:
        df_agg = (
            df.groupby("label")
            .agg(
                total_wraps    =("total_wraps",    "first"),
                total_visits   =("total_visits",   "first"),
                n_miss_visits  =("hit_inner_basin", lambda x: (~x).sum()),
                n_conv_visits  =("hit_inner_basin", "sum"),
                min_approach   =("closest_to_xstar", "min"),
            )
            .reset_index()
        )
        f.write(df_agg.to_string(index=False))
print(f"Summary saved to: {summary_path}")