"""
Strict wrapped-entry basin study — three guard offsets compared.
Produces one figure per guard offset: c=0.00, c=0.15, c=0.25.
"""

import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from hybrid_tools import PI, classify_points_with_diagnostics

DELTA     = 0.5
T_MAX     = 120.0
DT        = 0.05
T_TOL     = 1e-5
CONV_TOL  = 0.05
CONV_TIME = 10.0
N         = 12
MARGIN    = 0.15

# GUARD_OFFSETS = [0.0, 0.15, 0.25]
# GUARD_LABELS  = ["c = 0.00 (standard)", "c = 0.15", "c = 0.25"]

GUARD_OFFSETS = [0.3, -0.15, -0.25]
GUARD_LABELS  = ["c = 0.30", "c = -0.15", "c = -0.25"]

vals = np.linspace(-PI + MARGIN, PI - MARGIN, N)
grid = np.meshgrid(vals, vals, vals, indexing="ij")
X    = np.stack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()], axis=1)

os.makedirs("conv-basin/output", exist_ok=True)

all_summaries = {}

for GUARD_OFFSET, GUARD_LABEL in zip(GUARD_OFFSETS, GUARD_LABELS):
    tag = f"c{GUARD_OFFSET:.2f}".replace(".", "p")
    print(f"\n=== {GUARD_LABEL} ===")

    diag = classify_points_with_diagnostics(
        X,
        guard_offset=GUARD_OFFSET,
        delta=DELTA,
        t_max=T_MAX,
        dt=DT,
        t_tol=T_TOL,
        conv_tol=CONV_TOL,
        conv_time=CONV_TIME,
        progress_every=max(1, len(X) // 8),
    )

    converged             = diag["converged"]
    started_inside        = diag["initial_in_simplex"]
    inside_start_converged = converged & started_inside

    strict_wrapped_entry = (
        converged & (~started_inside) & diag["first_simplex_after_wrap"]
    )
    event_wrapped_to_inner_same_visit = (
        strict_wrapped_entry & diag["wrapped_entry_to_inner_same_visit"]
    )
    strict_wrapped_eventual_inner = (
        strict_wrapped_entry & diag["ever_enters_contracting"]
    )
    later_visit_inner = (
        strict_wrapped_eventual_inner & (~event_wrapped_to_inner_same_visit)
    )
    strict_wrapped_never_inner = (
        strict_wrapped_entry & (~diag["ever_enters_contracting"])
    )
    late_simplex_but_not_wrapped = (
        converged & (~started_inside)
        & diag["enters_simplex_later"]
        & (~diag["first_simplex_after_wrap"])
    )

    assert np.array_equal(
        strict_wrapped_entry,
        event_wrapped_to_inner_same_visit | later_visit_inner | strict_wrapped_never_inner,
    ), "Partition check failed"

    counts = {
        "total_points": int(len(X)),
        "converged": int(converged.sum()),
        "inside_start_converged": int(inside_start_converged.sum()),
        "late_simplex_no_prior_wrap": int(late_simplex_but_not_wrapped.sum()),
        "strict_wrapped_entry": int(strict_wrapped_entry.sum()),
        "later_visit_inner": int(later_visit_inner.sum()),
        "strict_wrapped_never_inner": int(strict_wrapped_never_inner.sum()),
        "event_wrapped_to_inner_same_visit": int(event_wrapped_to_inner_same_visit.sum()),
    }
    for k, v in counts.items():
        if k != "total_points":
            pct = 100.0 * v / len(X)
            print(f"  {k:45s}: {v:4d}  ({pct:.2f}%)")

    all_summaries[GUARD_LABEL] = counts

    # ── Figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)

    nonconv = X[~converged]
    if len(nonconv):
        ax1.scatter(nonconv[:, 0], nonconv[:, 1], nonconv[:, 2],
                    c="lightgray", s=5, alpha=0.06, label="does not converge")
    if len(X[inside_start_converged]):
        ax1.scatter(*X[inside_start_converged].T,
                    c="#2ca02c", s=14, alpha=0.55, label="starts in simplex, converges")
    if len(X[late_simplex_but_not_wrapped]):
        ax1.scatter(*X[late_simplex_but_not_wrapped].T,
                    c="#ffbf00", s=18, alpha=0.85, label="late simplex entry, no wrap")
    if len(X[later_visit_inner]):
        ax1.scatter(*X[later_visit_inner].T,
                    c="#ff7f0e", s=22, alpha=0.80, label="wrapped-entry -> inner (later visit)")
    if len(X[strict_wrapped_never_inner]):
        ax1.scatter(*X[strict_wrapped_never_inner].T,
                    c="#8c564b", s=22, alpha=0.85, label="wrapped-entry, never inner")
    if len(X[event_wrapped_to_inner_same_visit]):
        ax1.scatter(*X[event_wrapped_to_inner_same_visit].T,
                    c="#d62728", s=26, alpha=0.95, label="wrapped-entry -> inner (same visit)")

    t_d = np.linspace(-PI, PI / 3 + 0.1, 80)
    ax1.plot(t_d, t_d, t_d, "k-", lw=2.3, label="Diagonal")
    ax1.scatter([1/3], [1/3], [1/3], color="black", s=50)
    ax1.set_xlim(-PI, PI); ax1.set_ylim(-PI, PI); ax1.set_zlim(-PI, PI)
    ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$"); ax1.set_zlabel("$x_3$")
    ax1.set_title(f"Wrapped-entry categories in 3D\n{GUARD_LABEL}")
    ax1.legend(fontsize=7, loc="upper left")

    # 2D slice
    slice_idx = N // 2
    x3_val = vals[slice_idx]
    cat = np.zeros(len(X), dtype=int)
    cat[inside_start_converged]          = 1
    cat[late_simplex_but_not_wrapped]    = 2
    cat[later_visit_inner]               = 3
    cat[strict_wrapped_never_inner]      = 4
    cat[event_wrapped_to_inner_same_visit] = 5
    sl = cat.reshape(N, N, N)[:, :, slice_idx]

    cmap = ListedColormap(["#d9d9d9","#9fd39f","#ffe08a","#ffb366","#8c564b","#e57373"])
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N)
    ax2.imshow(sl.T, origin="lower", extent=[-PI, PI, -PI, PI],
               cmap=cmap, norm=norm, aspect="auto")
    ax2.set_xlim(-PI, PI); ax2.set_ylim(-PI, PI)
    ax2.set_aspect("equal")
    ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
    ax2.set_title(f"Slice at $x_3 = {x3_val:.2f}$ | {GUARD_LABEL}")
    ax2.grid(True, alpha=0.25)
    ax2.plot([-PI, PI], [-PI, PI], "k-", lw=1.0, alpha=0.35)
    ax2.scatter([x3_val], [x3_val], color="black", s=35, zorder=5)

    legend_handles = [
        Patch(facecolor="#d9d9d9", label="other / nonconverging"),
        Patch(facecolor="#9fd39f", label="starts in simplex, converges"),
        Patch(facecolor="#ffe08a", label="late simplex entry, no wrap"),
        Patch(facecolor="#ffb366", label="wrapped-entry -> inner (later visit)"),
        Patch(facecolor="#8c564b", label="wrapped-entry, never inner"),
        Patch(facecolor="#e57373", label="wrapped-entry -> inner (same visit)"),
    ]
    ax2.legend(handles=legend_handles, fontsize=7, loc="upper left")

    pct_conv = 100.0 * counts["converged"] / len(X)
    plt.suptitle(
        f"Strict wrapped-entry basin study | {GUARD_LABEL} | "
        f"$\\delta={DELTA}$ | N={N} | {counts['converged']}/1728 converge ({pct_conv:.1f}%)",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = f"conv-basin/output/wrapped_entry_basin_{tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close()

# ── Summary comparison table ───────────────────────────────────────────────
print("\n\nSummary comparison across guard offsets:")
print(f"{'Class':45s}", end="")
for lbl in GUARD_LABELS:
    print(f"  {lbl:>20s}", end="")
print()
print("-" * (45 + 22 * len(GUARD_LABELS)))

keys = ["converged", "inside_start_converged", "late_simplex_no_prior_wrap",
        "strict_wrapped_entry", "event_wrapped_to_inner_same_visit"]
for k in keys:
    print(f"{k:45s}", end="")
    for lbl in GUARD_LABELS:
        v = all_summaries[lbl][k]
        pct = 100.0 * v / 1728
        print(f"  {v:5d} ({pct:5.1f}%)", end="")
    print()

with open("conv-basin/output/wrapped_entry_comparison.json", "w") as f:
    json.dump(all_summaries, f, indent=2)
print("\nSaved conv-basin/output/wrapped_entry_comparison.json")