"""
2D projection animation for the D=3 hybrid oscillator.
Produces one MP4 per guard value — much easier to read than 3D.

Each figure has three 2D panels showing projections of the 3D trajectory:
  Panel 1: x1 vs x2
  Panel 2: x1 vs x3
  Panel 3: x2 vs x3

Guard lines appear as simple dashed vertical/horizontal lines.
The simplex edge appears as a diagonal line.
Torus wraps are shown as trail breaks (same logic as the 4D code).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

from hybrid_tools import PI, integrate_pc

# ── parameters ────────────────────────────────────────────────────────────────
DELTA       = 0.5
T_MAX       = 120.0
DT          = 0.05
T_TOL       = 1e-5
CONV_TOL    = 0.05
CONV_TIME   = 10.0
STORE_EVERY = 4
TAIL        = 8     # trail length in stored steps
SPEED       = 2     # stored steps per animation frame
FPS         = 24
OUTPUT_DIR  = "conv-basin/output/"

# One MP4 will be saved per entry here
GUARD_OFFSETS = [0.00, 0.15, 0.25]
GUARD_LABELS  = ["c=0.00", "c=0.15", "c=0.25"]

POINT_SPECS = [
    ((1/3,  1/3,  1/3),   "diagonal x*"),
    ((0.28, 0.28, 0.28),  "deep in simplex"),
    ((0.18, 0.18, 0.55),  "straddles c=0.25"),
    ((0.05, 0.05, 0.70),  "straddles c=0.15"),
    ((0.08, 0.20, 0.63),  "layered transitions"),
    ((-0.20, 0.40, 0.40), "one coord negative"),
    ((-2.99, -2.99, 0.05),"wrap, x3 near c=0"),
    ((-2.99, -2.99, 0.20),"wrap, x3 near c=0.25"),
    ((-2.99, -2.99, 0.28),"wrap, x3 always slow"),
    ((2.80,  0.20, 0.20), "positive edge wrap"),
]

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Which coordinate pair each panel shows: (x-axis dim, y-axis dim)
PROJECTIONS = [(0, 1), (0, 2), (1, 2)]
PROJ_LABELS = [("$x_1$", "$x_2$"), ("$x_1$", "$x_3$"), ("$x_2$", "$x_3$")]


# ── helpers ───────────────────────────────────────────────────────────────────

def classify_label(result):
    if result["converged"] and result["initial_in_simplex"]:
        return "starts in simplex, converges"
    if result["converged"] and result["first_simplex_after_wrap"] \
            and result["wrapped_entry_to_inner_same_visit"]:
        return "wrapped-entry -> inner basin"
    if result["converged"] and result["enters_simplex_later"] \
            and not result["first_simplex_after_wrap"]:
        return "late simplex entry, no wrap"
    if result["converged"]:
        return "converges"
    return "does not converge"


def split_trail(wrapped_2d, unwrapped_2d):
    """
    Insert NaN breaks wherever the trajectory crosses a torus boundary.
    wrapped_2d, unwrapped_2d: (N, 2) arrays.
    Returns x, y arrays with NaNs at wrap points.
    """
    if len(wrapped_2d) < 2:
        return wrapped_2d[:, 0].copy(), wrapped_2d[:, 1].copy()

    tiles = np.floor((unwrapped_2d + PI) / (2.0 * PI)).astype(int)
    jumps = np.where(np.any(np.diff(tiles, axis=0) != 0, axis=1))[0]

    x = wrapped_2d[:, 0].copy()
    y = wrapped_2d[:, 1].copy()
    for j in jumps[::-1]:
        x = np.insert(x, j + 1, np.nan)
        y = np.insert(y, j + 1, np.nan)
    return x, y


def guard_cross_mask(traj, guard_offset):
    """True at step i if any coordinate or the sum guard was crossed."""
    if len(traj) <= 1:
        return np.zeros(len(traj), dtype=bool)
    coord_cross = np.any(
        (traj[:-1] < guard_offset) != (traj[1:] < guard_offset), axis=1)
    sum_cross = (traj[:-1].sum(axis=1) > 1.0) != (traj[1:].sum(axis=1) > 1.0)
    mask = np.zeros(len(traj), dtype=bool)
    mask[1:] = coord_cross | sum_cross
    return mask


def add_static_decorations(ax, guard_offset, xi, yi):
    """
    Draw guard lines and the simplex edge for one 2D projection panel.
    xi, yi: which dimensions are on the x and y axes (0=x1, 1=x2, 2=x3).
    """
    lo, hi = -PI, PI
    c = guard_offset

    ax.axvline(c, color="darkorange", lw=1.0, ls="--", alpha=0.6, zorder=1,
               label=f"$x_{{{xi+1}}}=c$")
    ax.axhline(c, color="purple",     lw=1.0, ls="--", alpha=0.6, zorder=1,
               label=f"$x_{{{yi+1}}}=c$")

    # Simplex edge: x_i + x_j = 1  =>  x_j = 1 - x_i
    t = np.linspace(lo, hi, 200)
    sy = 1.0 - t
    in_range = (sy >= lo) & (sy <= hi)
    ax.plot(t[in_range], sy[in_range],
            color="royalblue", lw=1.4, alpha=0.55, zorder=1,
            label="simplex edge")

    # Fixed point x* = (1/3, 1/3, 1/3)
    ax.scatter([1/3], [1/3], color="black", s=40, zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18)


# ── main loop: one animation per guard value ──────────────────────────────────

for guard_offset, guard_label in zip(GUARD_OFFSETS, GUARD_LABELS):

    print(f"\n=== Integrating for {guard_label} ===")
    results = []
    for (pt, short_name), color in zip(POINT_SPECS, COLORS):
        res = integrate_pc(
            pt,
            guard_offset=guard_offset,
            delta=DELTA,
            t_max=T_MAX,
            dt=DT,
            t_tol=T_TOL,
            conv_tol=CONV_TOL,
            conv_time=CONV_TIME,
            store_every=STORE_EVERY,
            keep_trajectory=True,
        )
        label = classify_label(res)
        results.append({
            "point":      np.asarray(pt, dtype=float),
            "short_name": short_name,
            "label":      label,
            "color":      color,
            "traj_w":     res["traj_wrapped"],    # (N, 3) in [-pi, pi]
            "traj_u":     res["traj_unwrapped"],  # (N, 3) unwrapped
            "cross_mask": guard_cross_mask(res["traj_wrapped"], guard_offset),
        })
        print(f"  {short_name:22s} -> {label}")

    # ── figure: 3 projection panels + 1 info panel ───────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"2D projections  |  {guard_label}  |  δ=0.5  |  10 representative points",
        fontsize=12
    )

    proj_axes = axes[:3]
    ax_info   = axes[3]

    for ax, (xi, yi), (xl, yl) in zip(proj_axes, PROJECTIONS, PROJ_LABELS):
        add_static_decorations(ax, guard_offset, xi, yi)
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.set_title(
            f"{xl.strip('$')}–{yl.strip('$')} projection", fontsize=11)

    # Trajectory legend in first panel
    legend_handles = [
        Line2D([0], [0], color=COLORS[i], marker="o", lw=1.5, markersize=5,
               label=f"P{i+1:02d} {POINT_SPECS[i][1]}")
        for i in range(len(POINT_SPECS))
    ]
    proj_axes[0].legend(handles=legend_handles, fontsize=6.5,
                        loc="upper left", ncol=1)

    # Static decoration legend in second panel
    proj_axes[1].legend(fontsize=7, loc="upper right")

    # Mark initial positions as triangles
    for item in results:
        pt = item["point"]
        for ax, (xi, yi) in zip(proj_axes, PROJECTIONS):
            ax.scatter([pt[xi]], [pt[yi]], color=item["color"],
                       marker="^", s=35, alpha=0.7, zorder=4)

    # Info panel text
    ax_info.axis("off")
    info_lines = [
        f"Guard offset  c = {guard_offset:.2f}",
        "δ = 0.5,  T_max = 120",
        "",
        "-- = guard thresholds (xi = c)",
        "Blue line  = simplex edge (xi+xj=1)",
        "Black dot  = fixed point x*",
        "Triangle   = initial condition",
        "Trail break= torus wrap event",
        "Ring       = guard crossing",
        "",
        "Point classifications:",
        "  ✓ converges   ✗ does not converge",
    ]
    ROUTE_SHORT = {
        "starts in simplex, converges":  "via simplex (started inside)",
        "late simplex entry, no wrap":   "via slow drift into simplex",
        "wrapped-entry -> inner basin":  "via torus wrap -> inner basin",
        "converges":                     "converges (other route)",
        "does not converge":             "does not converge",
    }
    for i, item in enumerate(results, 1):
        converged = item["label"] != "does not converge"
        sym = "✓" if converged else "✗"
        route = ROUTE_SHORT.get(item["label"], item["label"])
        info_lines.append(f"{sym} P{i:02d} {item['short_name'][:14]:14s}")
        info_lines.append(f"     {route}")
    ax_info.text(0.02, 0.98, "\n".join(info_lines),
                 va="top", ha="left", fontsize=8, family="monospace",
                 transform=ax_info.transAxes)

    # ── animated artists ──────────────────────────────────────────────────────
    # trails[k][p] = Line2D for point k in projection p
    trails = [
        [ax.plot([], [], "-", color=item["color"], alpha=0.75,
                 lw=1.6, zorder=3)[0]
         for ax in proj_axes]
        for item in results
    ]
    heads = [
        [ax.plot([], [], "o", color=item["color"], ms=5, zorder=5)[0]
         for ax in proj_axes]
        for item in results
    ]

    max_len  = max(len(item["traj_w"]) for item in results)
    n_frames = max_len // SPEED + 1

    def make_init(trails, heads):
        def init():
            art = []
            for k in range(len(results)):
                for p in range(3):
                    trails[k][p].set_data([], [])
                    heads[k][p].set_data([], [])
                    art += [trails[k][p], heads[k][p]]
            return art
        return init

    def make_update(results, trails, heads, proj_axes):
        def update(frame):
            f = frame * SPEED
            art = []
            for k, item in enumerate(results):
                tw = item["traj_w"]
                tu = item["traj_u"]
                idx   = min(f, len(tw) - 1)
                start = max(0, idx - TAIL)

                tw_tail = tw[start:idx + 1]
                tu_tail = tu[start:idx + 1]
                flashing = item["cross_mask"][idx]

                for p, (xi, yi) in enumerate(PROJECTIONS):
                    x, y = split_trail(
                        tw_tail[:, [xi, yi]],
                        tu_tail[:, [xi, yi]],
                    )
                    trails[k][p].set_data(x, y)
                    heads[k][p].set_data([tw[idx, xi]], [tw[idx, yi]])

                    if flashing:
                        heads[k][p].set_markeredgecolor("black")
                        heads[k][p].set_markeredgewidth(1.8)
                        heads[k][p].set_markersize(8)
                    else:
                        heads[k][p].set_markeredgecolor(item["color"])
                        heads[k][p].set_markeredgewidth(0)
                        heads[k][p].set_markersize(5)

                    art += [trails[k][p], heads[k][p]]
            return art
        return update

    ani = animation.FuncAnimation(
        fig,
        make_update(results, trails, heads, proj_axes),
        frames=n_frames,
        init_func=make_init(trails, heads),
        interval=1000 / FPS,
        blit=True,
        repeat=False,
    )

    safe_label = guard_label.replace("=", "").replace(".", "p").replace("-", "neg")
    save_path  = f"{OUTPUT_DIR}professor_animation_2d_{safe_label}.mp4"
    plt.tight_layout()
    print(f"Saving {save_path} ...")
    ani.save(save_path, writer="ffmpeg", fps=FPS, bitrate=2000)
    print(f"Saved  {save_path}")
    plt.close(fig)

print("\nAll done.")