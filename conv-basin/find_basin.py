"""
Numerically computes and visualises the basin of attraction for the
hybrid oscillator (D=3) on the torus [-pi, pi)^3.

This version uses the same event-accurate predictor-corrector integrator
for basin classification as the trajectory scripts, so the basin figures
and trajectory figures are on the same numerical footing.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from hybrid_tools import PI, classify_points_pc

# CONFIG
GUARD_OFFSET = 0.0
DELTA        = 0.5
T_MAX        = 120.0
DT           = 0.05
T_TOL        = 1e-5
CONV_TOL     = 0.05
CONV_TIME    = 10.0
N            = 16
MARGIN       = 0.15


def find_boundary(conv_3d):
    """
    A point is on the boundary if it converges but has at least one
    non-converging neighbour in the grid.
    """
    from scipy.ndimage import binary_dilation

    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[1, 0, 1] = struct[1, 2, 1] = True
    struct[0, 1, 1] = struct[2, 1, 1] = True
    struct[1, 1, 0] = struct[1, 1, 2] = True
    dilated = binary_dilation(~conv_3d, structure=struct)
    return conv_3d & dilated


vals = np.linspace(-PI + MARGIN, PI - MARGIN, N)
grid = np.meshgrid(vals, vals, vals, indexing="ij")
X = np.stack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()], axis=1)

print(f"Classifying {len(X)} points with event-accurate integration (N={N})...")
conv_flat = classify_points_pc(
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
conv_3d = conv_flat.reshape(N, N, N)

pct = 100.0 * conv_flat.sum() / len(conv_flat)
print(f"Converging: {conv_flat.sum()}/{len(conv_flat)} ({pct:.1f}%)")

try:
    boundary_3d = find_boundary(conv_3d)
    boundary_flat = boundary_3d.ravel()
    print(f"Boundary points: {boundary_flat.sum()}")
except ImportError:
    print("scipy not found; skipping boundary detection")
    boundary_flat = np.zeros(len(conv_flat), dtype=bool)

cp = X[conv_flat]
bnd = X[boundary_flat]
ncp = X[~conv_flat]

os.makedirs("conv-basin/output", exist_ok=True)

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection="3d")
if len(ncp):
    ax1.scatter(ncp[:, 0], ncp[:, 1], ncp[:, 2], c="red", s=3, alpha=0.04, label="Does not converge")
if len(cp):
    ax1.scatter(cp[:, 0], cp[:, 1], cp[:, 2], c="green", s=6, alpha=0.25, label=f"Converges ({pct:.0f}%)")
if len(bnd):
    ax1.scatter(bnd[:, 0], bnd[:, 1], bnd[:, 2], c="black", s=8, alpha=0.6, label="Basin boundary")

t = np.linspace(-PI, PI / 3 + 0.1, 80)
ax1.plot(t, t, t, "b-", lw=2.5, label="Diagonal")
ax1.scatter([1 / 3], [1 / 3], [1 / 3], color="blue", s=60)
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_zlabel("$x_3$")
ax1.set_xlim(-PI, PI)
ax1.set_ylim(-PI, PI)
ax1.set_zlim(-PI, PI)
ax1.set_title(
    f"Basin of attraction\nevent-accurate classifier, $c={GUARD_OFFSET:.2f}$, $\\delta={DELTA}$, N={N}",
    fontsize=11,
)
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(122)
slice_indices = [N // 8, N // 4, N // 2, 3 * N // 4]
colors_slice = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
labels = [f"$x_3={vals[s]:.2f}$" for s in slice_indices]

for s_idx, col, lbl in zip(slice_indices, colors_slice, labels):
    slice_conv = conv_3d[:, :, s_idx]
    ii, jj = np.where(slice_conv)
    ax2.scatter(vals[ii], vals[jj], s=9, color=col, alpha=0.55, label=lbl)

ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_xlim(-PI, PI)
ax2.set_ylim(-PI, PI)
ax2.set_aspect("equal")
ax2.set_title("2D basin slices at fixed $x_3$", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

for s_idx, col in zip(slice_indices, colors_slice):
    v = vals[s_idx]
    ax2.scatter([v], [v], s=60, color=col, marker="*", zorder=5)

plt.suptitle(
    f"Event-accurate basin classification | $c={GUARD_OFFSET:.2f}$ | $\\delta={DELTA}$ | N={N}",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("conv-basin/output/basin_computed_exact.png", dpi=140, bbox_inches="tight")
print("Saved conv-basin/output/basin_computed_exact.png")

n_slices = min(N, 16)
slice_ids = np.linspace(0, N - 1, n_slices, dtype=int)
cols_grid = 4
rows_grid = int(np.ceil(n_slices / cols_grid))

fig2, axs = plt.subplots(rows_grid, cols_grid, figsize=(cols_grid * 3, rows_grid * 3))
for k, s_idx in enumerate(slice_ids):
    ax = axs.flat[k]
    sl = conv_3d[:, :, s_idx]
    ax.imshow(
        sl.T,
        origin="lower",
        extent=[-PI, PI, -PI, PI],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_title(f"$x_3={vals[s_idx]:.2f}$", fontsize=8)
    ax.set_xlabel("$x_1$", fontsize=7)
    ax.set_ylabel("$x_2$", fontsize=7)
    ax.tick_params(labelsize=6)
    v = vals[s_idx]
    ax.scatter([v], [v], s=30, c="blue", marker="*", zorder=5)

for k in range(n_slices, len(axs.flat)):
    axs.flat[k].axis("off")

plt.suptitle(
    f"All basin slices | event-accurate classifier | $c={GUARD_OFFSET:.2f}$ | $\\delta={DELTA}$",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("conv-basin/output/basin_slices_exact.png", dpi=130, bbox_inches="tight")
print("Saved conv-basin/output/basin_slices_exact.png")
plt.show()
