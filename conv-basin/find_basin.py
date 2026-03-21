"""
Numerically computes and visualises the basin of attraction for the
hybrid oscillator (D=3) on the torus [-pi, pi)^3.

Method:
  - Classify every point on an N^3 grid as converging or not.
  - Visualise:
      (a) All converging points (cloud).
      (b) Basin boundary: converging points that have at least one
          non-converging neighbour on the grid.
      (c) 2D slices through the torus at fixed x3 values.

CONFIG: edit the block below.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
GUARD_OFFSET = 0    # coordinate guard c. Try 0.0, np.pi/3, 2*np.pi/3
DELTA        = 0.5
T_MAX        = 500.0
DT           = 5e-3
CONV_TOL     = 0.05
CONV_TIME    = 20.0
N            = 40     # grid resolution per axis (N^3 points total)
              # N=20 is fast (~10s), N=40 is detailed (~2 min)
# ══════════════════════════════════════════════════════════════════

PI = np.pi

def wrap(x):
    return ((x + PI) % (2*PI)) - PI

def classify(X, guard_offset):
    """Vectorised classification of N points. X shape (N,3)."""
    x        = X.copy().astype(float)
    conv_for = np.zeros(len(x))
    done     = np.zeros(len(x), dtype=bool)
    n_steps  = int(T_MAX / DT)
    for step in range(n_steps):
        s    = x.sum(axis=1)
        fast = (x < guard_offset) | (s[:, None] > 1.0)
        x   += DT * np.where(fast, 1.0, 1.0 - DELTA)
        x    = wrap(x)
        spread   = np.abs(x - x.mean(axis=1, keepdims=True)).max(axis=1)
        conv_for = np.where(spread < CONV_TOL, conv_for + DT, 0.0)
        done    |= (conv_for >= CONV_TIME)
    return done

def find_boundary(conv_3d):
    """
    A point is on the boundary if it converges but has at least one
    non-converging neighbour (6-connectivity).
    """
    from scipy.ndimage import binary_dilation
    struct = np.zeros((3,3,3), dtype=bool)
    struct[1,0,1] = struct[1,2,1] = True  # x neighbours
    struct[0,1,1] = struct[2,1,1] = True  # y neighbours
    struct[1,1,0] = struct[1,1,2] = True  # z neighbours
    dilated  = binary_dilation(~conv_3d, structure=struct)
    boundary = conv_3d & dilated
    return boundary

# ── Run ───────────────────────────────────────────────────────────
vals = np.linspace(-PI + 0.1, PI - 0.1, N)
g    = np.meshgrid(vals, vals, vals, indexing='ij')
X    = np.stack([g[0].ravel(), g[1].ravel(), g[2].ravel()], axis=1)

print(f"Classifying {len(X)} points (N={N})...")
conv_flat = classify(X, GUARD_OFFSET)
conv_3d   = conv_flat.reshape(N, N, N)

pct = 100 * conv_flat.sum() / len(conv_flat)
print(f"Converging: {conv_flat.sum()}/{len(conv_flat)}  ({pct:.1f}%)")

# Basin boundary
try:
    boundary_3d = find_boundary(conv_3d)
    boundary_flat = boundary_3d.ravel()
    print(f"Boundary points: {boundary_flat.sum()}")
except ImportError:
    print("scipy not found — skipping boundary detection")
    boundary_flat = np.zeros(len(conv_flat), dtype=bool)

# ── Reconstruct coordinate arrays ────────────────────────────────
cp   = X[conv_flat]
bnd  = X[boundary_flat]
ncp  = X[~conv_flat]

# ── Figure 1: 3D cloud + boundary ────────────────────────────────
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
if len(ncp):
    ax1.scatter(ncp[:,0], ncp[:,1], ncp[:,2],
                c='red', s=3, alpha=0.04, label='Does not converge')
if len(cp):
    ax1.scatter(cp[:,0], cp[:,1], cp[:,2],
                c='green', s=5, alpha=0.25, label=f'Converges ({pct:.0f}%)')
if len(bnd):
    ax1.scatter(bnd[:,0], bnd[:,1], bnd[:,2],
                c='black', s=8, alpha=0.6, label='Basin boundary')

# diagonal
t = np.linspace(-PI, PI/3+0.1, 80)
ax1.plot(t, t, t, 'b-', lw=2.5, label='Diagonal')
ax1.scatter([1/3],[1/3],[1/3], color='blue', s=60)

ax1.set_xlabel('$x_1$'); ax1.set_ylabel('$x_2$'); ax1.set_zlabel('$x_3$')
ax1.set_xlim(-PI,PI); ax1.set_ylim(-PI,PI); ax1.set_zlim(-PI,PI)
ax1.set_title(f'Basin of attraction\n$c={GUARD_OFFSET:.2f}$, '
              f'$\\delta={DELTA}$, N={N}', fontsize=11)
ax1.legend(fontsize=8)

# ── Figure 2: 2D slices ───────────────────────────────────────────
ax2 = fig.add_subplot(122)

# Pick 4 evenly spaced slices in x3
slice_indices = [N//8, N//4, N//2, 3*N//4]
colors_slice  = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
labels        = [f'$x_3={vals[s]:.2f}$' for s in slice_indices]

for s_idx, col, lbl in zip(slice_indices, colors_slice, labels):
    # conv_3d[i,j,k]: i=x1, j=x2, k=x3
    slice_conv = conv_3d[:, :, s_idx]   # shape (N,N) in x1,x2
    # find converging points in this slice
    ii, jj = np.where(slice_conv)
    ax2.scatter(vals[ii], vals[jj], s=6, color=col, alpha=0.5, label=lbl)

ax2.set_xlabel('$x_1$'); ax2.set_ylabel('$x_2$')
ax2.set_xlim(-PI, PI);   ax2.set_ylim(-PI, PI)
ax2.set_aspect('equal')
ax2.set_title('2D slices of basin\n(converging points at fixed $x_3$)', fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# mark diagonal intersection in each slice (x1=x2=x3=val[s])
for s_idx, col in zip(slice_indices, colors_slice):
    v = vals[s_idx]
    ax2.scatter([v], [v], s=60, color=col, marker='*', zorder=5)

plt.suptitle(f'Numerically computed basin  |  '
             f'$c={GUARD_OFFSET:.2f}$  |  $\\delta={DELTA}$  |  N={N}',
             fontsize=12)
plt.tight_layout()
plt.savefig('output/basin_computed.png', dpi=140, bbox_inches='tight')
print("Saved basin_computed.png")

# ── Figure 3: all slices as a grid ───────────────────────────────
n_slices = min(N, 16)
slice_ids = np.linspace(0, N-1, n_slices, dtype=int)
cols_grid = 4
rows_grid = int(np.ceil(n_slices / cols_grid))

fig2, axs = plt.subplots(rows_grid, cols_grid,
                          figsize=(cols_grid*3, rows_grid*3))
for k, s_idx in enumerate(slice_ids):
    ax = axs.flat[k]
    sl = conv_3d[:, :, s_idx]
    ax.imshow(sl.T, origin='lower',
              extent=[-PI, PI, -PI, PI],
              cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_title(f'$x_3={vals[s_idx]:.2f}$', fontsize=8)
    ax.set_xlabel('$x_1$', fontsize=7); ax.set_ylabel('$x_2$', fontsize=7)
    ax.tick_params(labelsize=6)
    # mark diagonal
    v = vals[s_idx]
    ax.scatter([v], [v], s=30, c='blue', marker='*', zorder=5)

for k in range(n_slices, len(axs.flat)):
    axs.flat[k].axis('off')

plt.suptitle(f'Basin slices at all $x_3$ values  |  '
             f'$c={GUARD_OFFSET:.2f}$  |  $\\delta={DELTA}$',
             fontsize=11)
plt.tight_layout()
plt.savefig('output/basin_slices.png', dpi=130, bbox_inches='tight')
print("Saved basin_slices.png")
plt.show()