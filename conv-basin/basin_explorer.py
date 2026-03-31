"""
Basin of attraction explorer with event-accurate hybrid integration.

Tracked trajectories and optional background basin points now use the same
predictor-corrector integrator and the same torus-aware convergence test.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hybrid_tools import PI, classify_points_pc, integrate_pc

# CONFIG
GUARD_OFFSET = 0.0
DELTA        = 0.5
T_MAX        = 80.0
DT           = 0.05
T_TOL        = 1e-5
CONV_TOL     = 0.05
CONV_TIME    = 10.0

N_GRID       = 8
SHOW_SIMPLEX = True
SHOW_BASIN   = False
ANIMATE      = True
SAVE_MP4     = True
TAIL         = 120
SPEED        = 3

TRACKED_POINTS = [
    (1 / 3, 1 / 3, 1 / 3),
    (0.30, 0.35, 0.32),
    (0.80, 0.90, 0.40),
]

COLORS = [
    "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#2ca02c",
]


def classify_grid_exact(n, guard_offset, delta):
    vals = np.linspace(-PI + 0.4, PI - 0.4, n)
    grid = np.meshgrid(vals, vals, vals, indexing="ij")
    X = np.stack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()], axis=1)
    done = classify_points_pc(
        X,
        guard_offset=guard_offset,
        delta=delta,
        t_max=T_MAX,
        dt=DT,
        t_tol=T_TOL,
        conv_tol=CONV_TOL,
        conv_time=CONV_TIME,
        progress_every=max(1, len(X) // 6),
    )
    return X, done


print("Integrating tracked trajectories (predictor-corrector)...")
trajs_wrapped, trajs_unwrapped, conv_flags = [], [], []
for pt in TRACKED_POINTS:
    result = integrate_pc(
        pt,
        guard_offset=GUARD_OFFSET,
        delta=DELTA,
        t_max=T_MAX,
        dt=DT,
        t_tol=T_TOL,
        conv_tol=CONV_TOL,
        conv_time=CONV_TIME,
        store_every=4,
        keep_trajectory=True,
    )
    tw = result["traj_wrapped"]
    tu = result["traj_unwrapped"]
    conv = result["converged"]
    trajs_wrapped.append(tw)
    trajs_unwrapped.append(tu)
    conv_flags.append(conv)
    status = "CONVERGES" if conv else "DIVERGES"
    print(f"  {np.array(pt).round(2)}  {status:9s}  ({len(tw)} pts)")

if SHOW_BASIN and N_GRID > 0:
    print(f"Classifying {N_GRID**3} background points with event-accurate integration...")
    grid_pts, grid_conv = classify_grid_exact(N_GRID, GUARD_OFFSET, DELTA)
    print(f"  {grid_conv.sum()}/{len(grid_conv)} converge ({100 * grid_conv.sum() / len(grid_conv):.1f}%)")

fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-PI, PI)
ax.set_ylim(-PI, PI)
ax.set_zlim(-PI, PI)
ax.set_xlabel("$x_1$", fontsize=11)
ax.set_ylabel("$x_2$", fontsize=11)
ax.set_zlabel("$x_3$", fontsize=11)

if SHOW_BASIN and N_GRID > 0:
    cp, ncp = grid_pts[grid_conv], grid_pts[~grid_conv]
    if len(cp):
        ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], c="green", s=4, alpha=0.25)
    if len(ncp):
        ax.scatter(ncp[:, 0], ncp[:, 1], ncp[:, 2], c="red", s=4, alpha=0.08)

if SHOW_SIMPLEX:
    tri = Poly3DCollection([[np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]], alpha=0.15)
    tri.set_facecolor("royalblue")
    tri.set_edgecolor("royalblue")
    ax.add_collection3d(tri)

lo, hi, c = -PI, PI, GUARD_OFFSET
for col, plane in zip(
    ["darkorange", "purple", "saddlebrown"],
    [([c, c, c, c], [lo, lo, hi, hi], [lo, hi, hi, lo]),
     ([lo, lo, hi, hi], [c, c, c, c], [lo, hi, hi, lo]),
     ([lo, lo, hi, hi], [lo, hi, hi, lo], [c, c, c, c])],
):
    sq = Poly3DCollection([list(zip(*plane))], alpha=0.06)
    sq.set_facecolor(col)
    sq.set_edgecolor(col)
    ax.add_collection3d(sq)

t_d = np.linspace(-PI, PI / 3 + 0.1, 100)
ax.plot(t_d, t_d, t_d, "k-", lw=2.5, label="Diagonal")
ax.scatter([1 / 3], [1 / 3], [1 / 3], color="black", s=60, zorder=11)

for i, (pt, conv) in enumerate(zip(TRACKED_POINTS, conv_flags)):
    ax.scatter(
        *pt,
        color=COLORS[i % len(COLORS)],
        marker="^" if conv else "x",
        s=80,
        zorder=12,
        label=f"P{i + 1} {np.array(pt).round(1)} {'yes' if conv else 'no'}",
    )

if ANIMATE:
    max_segs = 8
    seg_lines = []
    heads = []
    for i in range(len(TRACKED_POINTS)):
        col = COLORS[i % len(COLORS)]
        lw = 2.0 if conv_flags[i] else 1.0
        segs = [ax.plot([], [], [], color=col, lw=lw, alpha=0.85)[0] for _ in range(max_segs)]
        seg_lines.append(segs)
        head, = ax.plot([], [], [], "o", color=col, ms=8, zorder=20)
        heads.append(head)

    max_raw = max(len(t) for t in trajs_wrapped)

    def get_segments(raw_seg, u_seg):
        if len(raw_seg) < 2:
            return [raw_seg]
        tiles = np.floor((u_seg + PI) / (2.0 * PI)).astype(int)
        jump_idx = np.where(np.any(np.diff(tiles, axis=0) != 0, axis=1))[0]
        if len(jump_idx) == 0:
            return [raw_seg]
        cuts = np.concatenate([[0], jump_idx + 1, [len(raw_seg)]])
        return [raw_seg[cuts[k]:cuts[k + 1]] for k in range(len(cuts) - 1) if cuts[k + 1] > cuts[k]]

    def update(frame):
        f = frame * SPEED
        artists = []
        for tr, tu, segs, head in zip(trajs_wrapped, trajs_unwrapped, seg_lines, heads):
            idx = min(f, len(tr) - 1)
            start = max(0, idx - TAIL)
            head.set_data([tr[idx, 0]], [tr[idx, 1]])
            head.set_3d_properties([tr[idx, 2]])
            parts = get_segments(tr[start:idx + 1], tu[start:idx + 1])
            for k, ln in enumerate(segs):
                if k < len(parts) and len(parts[k]) > 1:
                    p = parts[k]
                    ln.set_data(p[:, 0], p[:, 1])
                    ln.set_3d_properties(p[:, 2])
                else:
                    ln.set_data([], [])
                    ln.set_3d_properties([])
            artists += segs + [head]

        ax.set_title(
            f"Basin explorer | exact trajectories | $c={GUARD_OFFSET:.2f}$ | $\\delta={DELTA}$",
            fontsize=11,
        )
        return artists

    n_frames = max_raw // SPEED + 1
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=15, blit=False, repeat=False)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    plt.tight_layout()

    if SAVE_MP4:
        print("Saving...")
        ani.save("conv-basin/basin_animation.mp4", writer="ffmpeg", fps=30, dpi=150)
        print("Saved conv-basin/basin_animation.mp4")
    else:
        plt.show()
else:
    for i, tw in enumerate(trajs_wrapped):
        ax.plot(
            tw[:, 0], tw[:, 1], tw[:, 2],
            color=COLORS[i % len(COLORS)],
            lw=2.0 if conv_flags[i] else 1.0,
            alpha=0.8,
        )
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    plt.tight_layout()
    plt.show()
