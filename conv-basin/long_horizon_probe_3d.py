"""
Long-horizon 3D probe for points that do not appear to converge at c=0.

Workflow:
  1. Sample a coarse grid in [-pi, pi)^3.
  2. Keep points that do NOT converge by the usual horizon at c=0.
  3. Pick a diverse subset of those points.
  4. Animate that same subset for a much longer time horizon at:
       - c = 0.00
       - c = 0.25

The goal is to visually test whether points that look non-convergent on the
standard horizon might still settle down when c=0.25 is given much more time.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hybrid_tools import PI, integrate_pc, torus_spread_x

# ── configuration ─────────────────────────────────────────────────────────────
DELTA          = 0.5
DT             = 0.05
T_TOL          = 1e-5
CONV_TOL       = 0.05
CONV_TIME      = 10.0
STORE_EVERY    = 4
TAIL           = 16
SPEED          = 2
FPS            = 24

SELECT_T_MAX   = 120.0
LONG_T_MAX     = 400.0
SELECT_GRID_N  = 9
SELECT_MARGIN  = 0.35
NUM_POINTS     = 18

GUARD_OFFSETS  = [0.00, 0.25]
GUARD_LABELS   = ["c = 0.00", "c = 0.25"]
OUTPUT_DIR     = "conv-basin/output"
JSON_PATH      = os.path.join(OUTPUT_DIR, "long_horizon_probe_analysis.json")

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939",
    "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#31a354", "#756bb1",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def break_trail_3d(tw, tu):
    """Insert NaN breaks wherever the unwrapped trajectory changes torus tile."""
    if len(tw) < 2:
        return tw.copy()
    tiles = np.floor((tu + PI) / (2.0 * PI)).astype(int)
    jumps = np.where(np.any(np.diff(tiles, axis=0) != 0, axis=1))[0]
    if len(jumps) == 0:
        return tw.copy()
    out = tw.tolist()
    for j in jumps[::-1]:
        out.insert(j + 1, [np.nan, np.nan, np.nan])
    return np.asarray(out, dtype=float)


def draw_simplex_face(ax):
    verts = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)]
    tri = Poly3DCollection(verts, alpha=0.18)
    tri.set_facecolor("royalblue")
    tri.set_edgecolor("royalblue")
    tri.set_linewidth(1.2)
    ax.add_collection3d(tri)


def draw_guard_planes(ax, c):
    lo, hi = -PI, PI
    specs = [
        ("darkorange", [c, c, c, c], [lo, lo, hi, hi], [lo, hi, hi, lo]),
        ("purple", [lo, lo, hi, hi], [c, c, c, c], [lo, hi, hi, lo]),
        ("saddlebrown", [lo, lo, hi, hi], [lo, hi, hi, lo], [c, c, c, c]),
    ]
    for col, xs, ys, zs in specs:
        sq = Poly3DCollection([list(zip(xs, ys, zs))], alpha=0.07)
        sq.set_facecolor(col)
        sq.set_edgecolor(col)
        sq.set_linewidth(0.6)
        ax.add_collection3d(sq)


def draw_guard_simplex_intersection(ax, c):
    if c > 1.0 / 3.0 + 1e-6:
        return
    v = 1.0 - 2.0 * c
    if v < c:
        return
    verts = [np.array([[c, c, v], [c, v, c], [v, c, c]], dtype=float)]
    tri = Poly3DCollection(verts, alpha=0.45)
    tri.set_facecolor("dodgerblue")
    tri.set_edgecolor("blue")
    tri.set_linewidth(2.0)
    ax.add_collection3d(tri)


def draw_diagonal(ax):
    t = np.linspace(-PI, 1.0 / 3.0 + 0.05, 120)
    ax.plot(t, t, t, "k-", lw=2.2, alpha=0.9, zorder=10)
    ax.scatter([1/3], [1/3], [1/3], color="black", s=80, marker="*", zorder=15)


def setup_ax(ax, title, c):
    ax.set_xlim(-PI, PI)
    ax.set_ylim(-PI, PI)
    ax.set_zlim(-PI, PI)
    ax.set_xlabel("$x_1$", fontsize=10)
    ax.set_ylabel("$x_2$", fontsize=10)
    ax.set_zlabel("$x_3$", fontsize=10)
    ax.set_title(title, fontsize=11)
    draw_simplex_face(ax)
    draw_guard_planes(ax, c)
    draw_guard_simplex_intersection(ax, c)
    draw_diagonal(ax)


def make_selection_grid():
    vals = np.linspace(-PI + SELECT_MARGIN, PI - SELECT_MARGIN, SELECT_GRID_N)
    grid = np.meshgrid(vals, vals, vals, indexing="ij")
    X = np.stack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()], axis=1)
    return vals, X


def classify_grid(points, guard_offset, t_max, label):
    conv = np.zeros(len(points), dtype=bool)
    print(
        f"Classifying {len(points)} grid points for {label} "
        f"(c={guard_offset:.2f}, t={t_max:.0f})"
    )
    for i, x0 in enumerate(points):
        conv[i] = integrate_pc(
            x0,
            guard_offset=guard_offset,
            delta=DELTA,
            t_max=t_max,
            dt=DT,
            t_tol=T_TOL,
            conv_tol=CONV_TOL,
            conv_time=CONV_TIME,
            keep_trajectory=False,
        )["converged"]
        if (i + 1) % max(1, len(points) // 8) == 0:
            print(f"  diagnosed {i + 1}/{len(points)} points")
    return conv


def mask_points_to_list(points, mask, limit=20):
    return [np.round(pt, 6).tolist() for pt in points[mask][:limit]]


def summarize_pair(points, conv_a, conv_b, label_a, label_b):
    conv_a = np.asarray(conv_a, dtype=bool)
    conv_b = np.asarray(conv_b, dtype=bool)
    total = int(len(points))

    both = conv_a & conv_b
    neither = (~conv_a) & (~conv_b)
    gained_b = (~conv_a) & conv_b
    lost_b = conv_a & (~conv_b)

    nonconv_a = ~conv_a
    nonconv_b = ~conv_b

    return {
        "labels": {
            "a": label_a,
            "b": label_b,
        },
        "counts": {
            f"converged_{label_a}": int(conv_a.sum()),
            f"converged_{label_b}": int(conv_b.sum()),
            "converged_both": int(both.sum()),
            "converged_neither": int(neither.sum()),
            f"nonconverged_{label_a}_that_converge_{label_b}": int(gained_b.sum()),
            f"nonconverged_{label_b}_that_converge_{label_a}": int(lost_b.sum()),
        },
        "fractions": {
            f"fraction_nonconverged_{label_a}_that_converge_{label_b}": (
                float(gained_b.sum() / nonconv_a.sum()) if nonconv_a.sum() else 0.0
            ),
            f"fraction_nonconverged_{label_b}_that_converge_{label_a}": (
                float(lost_b.sum() / nonconv_b.sum()) if nonconv_b.sum() else 0.0
            ),
            f"fraction_converged_{label_a}": float(conv_a.sum() / total),
            f"fraction_converged_{label_b}": float(conv_b.sum() / total),
        },
        "sample_points": {
            f"nonconverged_{label_a}_that_converge_{label_b}": mask_points_to_list(points, gained_b),
            f"nonconverged_{label_b}_that_converge_{label_a}": mask_points_to_list(points, lost_b),
            "converged_both": mask_points_to_list(points, both),
            "converged_neither": mask_points_to_list(points, neither),
        },
    }


def run_grid_analysis():
    vals, X = make_selection_grid()
    short_c0 = classify_grid(X, 0.00, SELECT_T_MAX, "short_horizon")
    short_c025 = classify_grid(X, 0.25, SELECT_T_MAX, "short_horizon")
    long_c0 = classify_grid(X, 0.00, LONG_T_MAX, "long_horizon")
    long_c025 = classify_grid(X, 0.25, LONG_T_MAX, "long_horizon")

    summary = {
        "grid": {
            "n_per_axis": SELECT_GRID_N,
            "margin": SELECT_MARGIN,
            "total_points": int(len(X)),
            "axis_values": np.round(vals, 6).tolist(),
        },
        "standard_horizon_comparison": summarize_pair(
            X, short_c0, short_c025, "c0_t120", "c025_t120"
        ),
        "long_horizon_comparison": summarize_pair(
            X, long_c0, long_c025, "c0_t400", "c025_t400"
        ),
        "time_horizon_effect": {
            "c0": summarize_pair(X, short_c0, long_c0, "c0_t120", "c0_t400"),
            "c025": summarize_pair(X, short_c025, long_c025, "c025_t120", "c025_t400"),
        },
    }
    return X, short_c0, summary


def select_candidate_points(points, short_conv_c0):
    """
    Pick a diverse subset from the points that do not converge by SELECT_T_MAX
    when c=0. The selection is greedy farthest-point sampling seeded by the
    largest initial torus spread.
    """
    print(
        f"Selecting from {len(points)} grid points: keeping those non-convergent "
        f"by t={SELECT_T_MAX:.0f} at c=0.00"
    )
    pool = points[~short_conv_c0]
    if len(pool) == 0:
        raise RuntimeError("No non-converging points found in the selection grid.")

    spreads = np.array([torus_spread_x(x0) for x0 in pool], dtype=float)
    chosen_ids = [int(np.argmax(spreads))]

    while len(chosen_ids) < min(NUM_POINTS, len(pool)):
        chosen = pool[chosen_ids]
        best_idx = None
        best_score = -np.inf
        for idx in range(len(pool)):
            if idx in chosen_ids:
                continue
            d = np.linalg.norm(pool[idx] - chosen, axis=1)
            score = float(d.min()) + 0.2 * spreads[idx]
            if score > best_score:
                best_score = score
                best_idx = idx
        chosen_ids.append(best_idx)

    picked = pool[chosen_ids]
    print(f"Selected {len(picked)} representative points from {len(pool)} non-convergers.")
    return picked


def integrate_selected_points(points, guard_offset):
    results = []
    for i, pt in enumerate(points):
        res = integrate_pc(
            pt,
            guard_offset=guard_offset,
            delta=DELTA,
            t_max=LONG_T_MAX,
            dt=DT,
            t_tol=T_TOL,
            conv_tol=CONV_TOL,
            conv_time=CONV_TIME,
            store_every=STORE_EVERY,
            keep_trajectory=True,
            stop_on_convergence=False,
        )
        results.append({
            "id": i + 1,
            "point": np.asarray(pt, dtype=float),
            "traj_w": res["traj_wrapped"],
            "traj_u": res["traj_unwrapped"],
            "converged": bool(res["converged"]),
            "t_final": float(res["t_final"]),
        })
    return results


def build_info_lines(results, guard_label):
    conv_count = sum(item["converged"] for item in results)
    lines = [
        f"{guard_label}  |  long probe",
        f"Selected at c = 0.00 using t = {SELECT_T_MAX:.0f}",
        f"Animated to t = {LONG_T_MAX:.0f}",
        f"{conv_count}/{len(results)} converge by long horizon",
        "",
        "Same initial points in both movies",
        "They were chosen from points that do not",
        "converge by the standard c=0 horizon.",
        "",
        "Legend:",
        "triangle = initial condition",
        "star     = x*",
        "trail break = torus wrap",
        "",
        "Point status:",
    ]
    for item in results:
        status = "yes" if item["converged"] else "no"
        xyz = np.round(item["point"], 2)
        lines.append(f"P{item['id']:02d}  {status:>3s}  {xyz}")
    return lines


def save_animation(points, results, guard_offset, guard_label):
    fig = plt.figure(figsize=(16, 9))
    ax3d = fig.add_subplot(121, projection="3d")
    ax_info = fig.add_subplot(122)

    setup_ax(ax3d, f"Long-horizon probe  |  {guard_label}  |  δ=0.5", guard_offset)
    ax_info.axis("off")

    for item, color in zip(results, COLORS):
        pt = item["point"]
        ax3d.scatter([pt[0]], [pt[1]], [pt[2]],
                     color=color, marker="^", s=40, alpha=0.85, zorder=5)

    legend_handles = [
        Line2D([0], [0], color=COLORS[i], marker="o", lw=1.5, markersize=5,
               label=f"P{i+1:02d}")
        for i in range(len(results))
    ]
    ax3d.legend(handles=legend_handles, fontsize=7, loc="upper left", ncol=2)

    info_lines = build_info_lines(results, guard_label)
    ax_info.text(
        0.02, 0.98, "\n".join(info_lines),
        va="top", ha="left", fontsize=8.5, family="monospace",
        transform=ax_info.transAxes,
    )

    trail_lines = []
    head_dots = []
    for item, color in zip(results, COLORS):
        line, = ax3d.plot([], [], [], "-", color=color, alpha=0.78, lw=1.8, zorder=3)
        dot, = ax3d.plot([], [], [], "o", color=color, ms=6, zorder=6)
        trail_lines.append(line)
        head_dots.append(dot)

    max_len = max(len(item["traj_w"]) for item in results)
    n_frames = max_len // SPEED + 1
    time_annot = ax3d.text2D(
        0.02, 0.96, "t = 0.0",
        transform=ax3d.transAxes, fontsize=9, color="black", family="monospace",
    )

    def init():
        for ln, dot in zip(trail_lines, head_dots):
            ln.set_data([], [])
            ln.set_3d_properties([])
            dot.set_data([], [])
            dot.set_3d_properties([])
        return trail_lines + head_dots + [time_annot]

    def update(frame):
        f = frame * SPEED
        artists = [time_annot]
        t_approx = min(f * DT * STORE_EVERY, LONG_T_MAX)
        time_annot.set_text(f"t ≈ {t_approx:.1f}")

        for item, ln, dot in zip(results, trail_lines, head_dots):
            tw = item["traj_w"]
            tu = item["traj_u"]
            idx = min(f, len(tw) - 1)
            start = max(0, idx - TAIL)
            trail = break_trail_3d(tw[start:idx + 1], tu[start:idx + 1])
            ln.set_data(trail[:, 0], trail[:, 1])
            ln.set_3d_properties(trail[:, 2])

            hx, hy, hz = tw[idx]
            dot.set_data([hx], [hy])
            dot.set_3d_properties([hz])
            artists += [ln, dot]
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=1000 / FPS,
        blit=False,
        repeat=False,
    )

    tag = f"c{guard_offset:.2f}".replace(".", "p")
    out_path = os.path.join(OUTPUT_DIR, f"long_horizon_probe_3d_{tag}.mp4")
    plt.tight_layout()
    print(f"Saving {out_path} ...")
    ani.save(out_path, writer="ffmpeg", fps=FPS, dpi=150, bitrate=3000)
    print(f"Saved  {out_path}")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grid_points, short_conv_c0, grid_summary = run_grid_analysis()
    points = select_candidate_points(grid_points, short_conv_c0)
    analysis = {
        "config": {
            "delta": DELTA,
            "dt": DT,
            "t_tol": T_TOL,
            "conv_tol": CONV_TOL,
            "conv_time": CONV_TIME,
            "store_every": STORE_EVERY,
            "selection_t_max": SELECT_T_MAX,
            "long_t_max": LONG_T_MAX,
            "selection_grid_n": SELECT_GRID_N,
            "selection_margin": SELECT_MARGIN,
            "num_points_animated": NUM_POINTS,
            "guard_offsets": GUARD_OFFSETS,
        },
        "grid_analysis": grid_summary,
        "selected_points": [np.round(pt, 6).tolist() for pt in points],
        "selected_points_long_probe": {},
    }

    for guard_offset, guard_label in zip(GUARD_OFFSETS, GUARD_LABELS):
        print(f"\n=== Long-horizon integration for {guard_label} ===")
        results = integrate_selected_points(points, guard_offset)
        conv_count = sum(item["converged"] for item in results)
        print(f"  {conv_count}/{len(results)} converge by t={LONG_T_MAX:.0f}")
        tag = f"c{guard_offset:.2f}".replace(".", "p")
        analysis["selected_points_long_probe"][tag] = {
            "guard_label": guard_label,
            "converged_count": int(conv_count),
            "statuses": [
                {
                    "id": item["id"],
                    "point": np.round(item["point"], 6).tolist(),
                    "converged": bool(item["converged"]),
                    "t_final": float(item["t_final"]),
                }
                for item in results
            ],
        }
        # save_animation(points, results, guard_offset, guard_label)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved {JSON_PATH}")
    print("\nAll done.")


if __name__ == "__main__":
    main()
