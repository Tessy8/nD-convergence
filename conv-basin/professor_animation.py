"""
Professor-ready comparison animation for the D=3 hybrid oscillator.

This version is tailored to the professor's visual requests:
- 10 representative initial conditions in 3D
- the coordinate guard planes and simplex visible in each panel
- side-by-side comparison between centered guards and shifted guards
- short torus trails with breaks at wrap events
- visual highlighting when a trajectory crosses a guard surface
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hybrid_tools import PI, integrate_pc

DELTA = 0.5
T_MAX = 120.0
DT = 0.05
T_TOL = 1e-5
CONV_TOL = 0.05
CONV_TIME = 10.0
STORE_EVERY = 4
TAIL = 60
SPEED = 2
FPS = 24
SAVE_PATH = "conv-basin/output/professor_animation.mp4"

GUARD_OFFSETS = [0.0, 0.15, 0.25]
GUARD_LABELS  = ["c = 0.00 (standard)", "c = 0.15", "c = 0.25"]
# GUARD_OFFSETS = [-0.30, 0.0, 0.30]
# GUARD_LABELS = ["c = -0.30", "c = 0.00", "c = 0.30"]

POINT_SPECS = [
    # 1. Always converges — reference orbit
    ((1/3, 1/3, 1/3), "diagonal x*"),

    # 2. Inside simplex at all three c values (all coords > 0.25, sum < 1)
    # Converges at c=0, 0.15, 0.25 — stable baseline
    ((0.28, 0.28, 0.28), "deep in simplex"),

    # 3. Two coords at 0.18: inside simplex at c=0 and c=0.15 (0.18>0.15),
    # but fast at c=0.25 (0.18<0.25). Behaviour changes at the third panel.
    ((0.18, 0.18, 0.55), "straddles c=0.25"),

    # 4. Two coords at 0.05: inside simplex at c=0 (0.05>0),
    # but fast at c=0.15 and c=0.25 (0.05<0.15). Changes at second panel.
    ((0.05, 0.05, 0.70), "straddles c=0.15"),

    # 5. One coord at 0.08, one at 0.20, one at 0.63:
    # x1 transitions at c=0.15, x2 transitions at c=0.25. Layered effect.
    ((0.08, 0.20, 0.63), "layered transitions"),

    # 6. One slightly negative coord — fast at all c values because x1<0<c.
    # Shows mixed-speed dynamics clearly.
    ((-0.20, 0.40, 0.40), "one coord negative"),

    # 7. Two coords deep negative, one just above c=0 threshold.
    # Wraps around. x3=0.05 is slow at c=0 but fast at c=0.15, c=0.25.
    # Shows how the landing zone after wrap shrinks with c.
    ((-2.99, -2.99, 0.05), "wrap, x3 near c=0"),

    # 8. Two coords deep negative, one at 0.20.
    # x3=0.20 is slow at c=0 and c=0.15, fast at c=0.25.
    # Landing zone after wrap shrinks at third panel.
    ((-2.99, -2.99, 0.20), "wrap, x3 near c=0.25"),

    # 9. Two coords deep negative, one at 0.28.
    # x3=0.28 is slow at all c values. Does it converge at all three?
    # Contrasts with P07 and P08 above.
    ((-2.99, -2.99, 0.28), "wrap, x3 always slow"),

    # 10. One coord near positive edge (wraps to ~-pi), two positive.
    # Tests whether the positive-edge wrap helps or hurts at different c.
    ((2.80, 0.20, 0.20), "positive edge wrap"),
]

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def classify_label(result):
    if result["converged"] and result["initial_in_simplex"]:
        return "starts in simplex, converges"
    if result["converged"] and result["first_simplex_after_wrap"] and result["wrapped_entry_to_inner_same_visit"]:
        return "wrapped-entry -> inner basin"
    if result["converged"] and result["enters_simplex_later"] and not result["first_simplex_after_wrap"]:
        return "late simplex entry, no wrap"
    if result["converged"]:
        return "converges"
    return "does not converge"


def break_segments(raw_seg, unwrapped_seg):
    if len(raw_seg) < 2:
        return [raw_seg]
    tiles = np.floor((unwrapped_seg + PI) / (2.0 * PI)).astype(int)
    jumps = np.where(np.any(np.diff(tiles, axis=0) != 0, axis=1))[0]
    if len(jumps) == 0:
        return [raw_seg]
    cuts = np.concatenate([[0], jumps + 1, [len(raw_seg)]])
    return [raw_seg[cuts[k]:cuts[k + 1]] for k in range(len(cuts) - 1) if cuts[k + 1] > cuts[k]]


def guard_cross_mask(traj, guard_offset):
    if len(traj) <= 1:
        return np.zeros(len(traj), dtype=bool)
    coord_before = traj[:-1] < guard_offset
    coord_after = traj[1:] < guard_offset
    coord_cross = np.any(coord_before != coord_after, axis=1)

    sum_before = traj[:-1].sum(axis=1) > 1.0
    sum_after = traj[1:].sum(axis=1) > 1.0
    sum_cross = sum_before != sum_after

    mask = np.zeros(len(traj), dtype=bool)
    mask[1:] = coord_cross | sum_cross
    return mask


def add_guard_planes(ax, guard_offset):
    lo, hi, c = -PI, PI, guard_offset
    for col, plane in zip(
        ["darkorange", "purple", "saddlebrown"],
        [([c, c, c, c], [lo, lo, hi, hi], [lo, hi, hi, lo]),
         ([lo, lo, hi, hi], [c, c, c, c], [lo, hi, hi, lo]),
         ([lo, lo, hi, hi], [lo, hi, hi, lo], [c, c, c, c])],
    ):
        sq = Poly3DCollection([list(zip(*plane))], alpha=0.05)
        sq.set_facecolor(col)
        sq.set_edgecolor(col)
        ax.add_collection3d(sq)


def add_simplex(ax):
    tri = Poly3DCollection([[np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]], alpha=0.13)
    tri.set_facecolor("royalblue")
    tri.set_edgecolor("royalblue")
    ax.add_collection3d(tri)


def setup_axis(ax, title, guard_offset):
    ax.set_xlim(-PI, PI)
    ax.set_ylim(-PI, PI)
    ax.set_zlim(-PI, PI)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title(title, fontsize=12)
    add_simplex(ax)
    add_guard_planes(ax, guard_offset)
    t = np.linspace(-PI, PI / 3 + 0.1, 100)
    ax.plot(t, t, t, "k-", lw=2.3)
    ax.scatter([1 / 3], [1 / 3], [1 / 3], color="black", s=50, zorder=10)


print("Integrating representative trajectories for both guard settings...")
panel_results = []
for guard_offset, guard_label in zip(GUARD_OFFSETS, GUARD_LABELS):
    results = []
    print(f"  Guard {guard_label}")
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
        wrapped = res["traj_wrapped"]
        unwrapped = res["traj_unwrapped"]
        result_row = {
            "point": np.asarray(pt, dtype=float),
            "short_name": short_name,
            "detected_label": classify_label(res),
            "color": color,
            "traj_wrapped": wrapped,
            "traj_unwrapped": unwrapped,
            "cross_mask": guard_cross_mask(wrapped, guard_offset),
        }
        results.append(result_row)
        print(f"    {short_name:16s} {np.round(pt, 3)} -> {result_row['detected_label']}")
    panel_results.append(results)

fig = plt.figure(figsize=(22, 9))
ax_left = fig.add_subplot(141, projection="3d")
ax_mid = fig.add_subplot(142, projection="3d")
ax_right = fig.add_subplot(143, projection="3d")
ax_info = fig.add_subplot(144)

setup_axis(ax_left, f"Shifted guards ({GUARD_LABELS[0]})", GUARD_OFFSETS[0])
setup_axis(ax_mid, f"Centered guards ({GUARD_LABELS[1]})", GUARD_OFFSETS[1])
setup_axis(ax_right, f"Shifted guards ({GUARD_LABELS[2]})", GUARD_OFFSETS[2])

legend_handles = [
    Line2D([0], [0], color=COLORS[i], marker="o", lw=1.8, markersize=6, label=f"P{i+1:02d} {POINT_SPECS[i][1]}")
    for i in range(len(POINT_SPECS))
]
ax_left.legend(handles=legend_handles, fontsize=7, loc="upper left", ncol=2)

panel_axes = [ax_left, ax_mid, ax_right]
all_seg_lines = []
all_heads = []
for ax, results in zip(panel_axes, panel_results):
    seg_lines = []
    heads = []
    max_segs = 6
    for item in results:
        pt = item["point"]
        color = item["color"]
        ax.scatter([pt[0]], [pt[1]], [pt[2]], color=color, marker="^", s=42, alpha=0.75)
        segs = [ax.plot([], [], [], color=color, lw=1.5, alpha=0.8)[0] for _ in range(max_segs)]
        head, = ax.plot([], [], [], "o", color=color, ms=5.5, zorder=20)
        seg_lines.append(segs)
        heads.append(head)
    all_seg_lines.append(seg_lines)
    all_heads.append(heads)

ax_info.axis("off")
ax_info.set_title("What To Watch", fontsize=13, loc="left")
info_lines = [
    "Same 10 initial points shown in all three panels.",
    f"Left:   {GUARD_LABELS[0]}",
    f"Middle: {GUARD_LABELS[1]}",
    f"Right:  {GUARD_LABELS[2]}",
    "Trails break at torus wraps (clean lines).",
    "Black-outlined dot = trajectory at a guard crossing.",
    "Blue triangle = simplex face (sum guard).",
    "Orange/purple/brown planes = coordinate guards.",
    "",
    "Key question: do any non-converging points (at c=0)",
    "start converging when guards shift to c=0.25?",
]
for i, results in enumerate(panel_results):
    info_lines.append(f"{GUARD_LABELS[i]} examples:")
    for j, item in enumerate(results, start=1):
        info_lines.append(f"P{j:02d}  {item['short_name']}: {item['detected_label']}")
    info_lines.append("")
info_text = ax_info.text(0.02, 0.98, "\n".join(info_lines), va="top", ha="left", fontsize=9, family="monospace")

max_len = max(len(item["traj_wrapped"]) for results in panel_results for item in results)
n_frames = max_len // SPEED + 1


def update(frame):
    f = frame * SPEED
    artists = [info_text]
    for ax, results, seg_lines, heads, guard_label in zip(panel_axes, panel_results, all_seg_lines, all_heads, GUARD_LABELS):
        for item, segs, head in zip(results, seg_lines, heads):
            tr = item["traj_wrapped"]
            tu = item["traj_unwrapped"]
            idx = min(f, len(tr) - 1)
            start = max(0, idx - TAIL)
            head.set_data([tr[idx, 0]], [tr[idx, 1]])
            head.set_3d_properties([tr[idx, 2]])
            if item["cross_mask"][idx]:
                head.set_markeredgecolor("black")
                head.set_markeredgewidth(1.6)
                head.set_markersize(7.5)
            else:
                head.set_markeredgecolor(item["color"])
                head.set_markeredgewidth(0.0)
                head.set_markersize(5.5)
            parts = break_segments(tr[start:idx + 1], tu[start:idx + 1])
            for k, ln in enumerate(segs):
                if k < len(parts) and len(parts[k]) > 1:
                    p = parts[k]
                    ln.set_data(p[:, 0], p[:, 1])
                    ln.set_3d_properties(p[:, 2])
                else:
                    ln.set_data([], [])
                    ln.set_3d_properties([])
            artists.extend(segs)
            artists.append(head)
        ax.set_title(f"{guard_label} | wrap breaks and guard flashes", fontsize=11)
    return artists

# Quick pre-check: which points change classification between panels?
print("\nClassification changes across guard settings:")
for j, (pt, name) in enumerate(POINT_SPECS):
    labels = [panel_results[i][j]["detected_label"] for i in range(len(GUARD_OFFSETS))]
    if len(set(labels)) > 1:
        print(f"  P{j+1:02d} {name}: {labels}")
        
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS, blit=False, repeat=False)
plt.tight_layout()
print(f"Saving {SAVE_PATH} ...")
ani.save(SAVE_PATH, writer="ffmpeg", fps=FPS, dpi=150)
print(f"Saved {SAVE_PATH}")
