"""
Basin of attraction explorer — accurate hybrid integration.

Uses a predictor-corrector step (same logic as OdePC in syncTime4D.py):
  - at each step, check if the vector field changes
  - if it does and h > tTol, halve h and retry
  - this bisects to the exact switching surface crossing before continuing

CONTROLS: edit CONFIG block below.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
GUARD_OFFSET = 0.0
DELTA        = 0.5
T_MAX        = 80.0
DT           = 0.05       # base timestep (OdePC adapts inside)
T_TOL        = 1e-5       # minimum step at switching surface
CONV_TOL     = 0.05
CONV_TIME    = 10.0

N_GRID       = 8
SHOW_SIMPLEX = True
SHOW_BASIN   = False
ANIMATE      = True
SAVE_MP4     = True
TAIL         = 120
SPEED        = 3

PI = np.pi
# TRACKED_POINTS = [
#     ( 0.30,  0.35,  0.32),
#     ( 0.20,  0.10,  0.05),
#     ( 0.60,  0.50,  0.40),
#     (-0.50,  0.60,  0.50),
#     (-1.50, -1.20, -1.80),
#     ( 2.50,  2.80,  2.60),
#     (-2.50, -2.80, -2.60),
#     (-PI+0.1, -PI+0.1, -PI+0.1),
# ]
TRACKED_POINTS = [
    (1/3, 1/3, 1/3),      # exactly on diagonal — should just travel the diagonal
    (0.30, 0.35, 0.32),   # very close — should spiral in quickly  
    (0.80, 0.90, 0.40),   # further out — does it converge or not?
]
# ══════════════════════════════════════════════════════════════════

def wrap(x):
    return ((x + PI) % (2*PI)) - PI

def vfield(x, guard_offset, delta):
    """Vector field — no wrapping applied here."""
    s = x.sum()
    return np.where((x < guard_offset) | (s > 1.0), 1.0, 1.0 - delta)

def integrate_pc(x0, guard_offset, delta, t_max, dt, t_tol,
                 store_every=4, conv_tol=0.05, conv_time=10.0):
    """
    Predictor-corrector integrator with event detection.
    Mirrors OdePC from syncTime4D.py:
      - predict: y_h = y + h * f(y)
      - evaluate f(y_h)
      - if f changed and h > t_tol: h /= 2, retry
      - else: accept step, grow h toward dt
    Returns (traj_wrapped, traj_unwrapped, converged).
    """
    x     = np.array(x0, dtype=float)
    x_un  = x.copy()
    traj_w = [x.copy()]
    traj_u = [x_un.copy()]

    h      = dt
    t      = 0.0
    dy0    = vfield(x, guard_offset, delta)
    step   = 0
    conv_for = 0.0
    converged = False

    while t < t_max:
        # predict
        x_h  = x + h * dy0
        x_h_wrapped = wrap(x_h)
        dy1  = vfield(x_h_wrapped, guard_offset, delta)

        # event detection: did the vector field change?
        if not np.allclose(dy1, dy0) and h > t_tol:
            h /= 2.0
            continue

        # accept step — advance unwrapped and wrapped
        x_un  = x_un + h * dy0
        x     = wrap(x + h * dy0)
        t    += h
        dy0   = vfield(x, guard_offset, delta)

        # grow step back toward dt
        h = min(h * 1.5, dt)

        # store
        step += 1
        if step % store_every == 0:
            traj_w.append(x.copy())
            traj_u.append(x_un.copy())

        # convergence check
        if np.abs(x - x.mean()).max() < CONV_TOL:
            conv_for += h
            if conv_for >= CONV_TIME:
                converged = True
                break
        else:
            conv_for = 0.0

    return np.array(traj_w), np.array(traj_u), converged


def break_wraps(traj_w, traj_u):
    tiles = np.floor((traj_u + PI) / (2*PI)).astype(int)
    jumps = np.any(np.diff(tiles, axis=0) != 0, axis=1)
    if not np.any(jumps):
        return traj_w
    out = traj_w.tolist()
    nan_row = [np.nan, np.nan, np.nan]
    for j in np.where(jumps)[0][::-1]:
        out.insert(j + 1, nan_row)
    return np.array(out)


def classify_grid(N, guard_offset, delta):
    vals = np.linspace(-PI+0.4, PI-0.4, N)
    g    = np.meshgrid(vals, vals, vals, indexing='ij')
    X    = np.stack([g[0].ravel(),g[1].ravel(),g[2].ravel()], axis=1).astype(float)
    x    = X.copy()
    cf   = np.zeros(len(x)); done = np.zeros(len(x), dtype=bool)
    dt_g = 0.01
    for _ in range(int(T_MAX/dt_g)):
        s    = x.sum(axis=1)
        fast = (x < guard_offset) | (s[:,None] > 1.0)
        x   += dt_g * np.where(fast, 1.0, 1.0-delta)
        x    = wrap(x)
        sp   = np.abs(x - x.mean(axis=1,keepdims=True)).max(axis=1)
        cf   = np.where(sp < CONV_TOL, cf+dt_g, 0.0)
        done |= (cf >= CONV_TIME)
    return X, done


# ── Simulate ──────────────────────────────────────────────────────
print("Integrating tracked trajectories (predictor-corrector)...")
trajs_w, trajs_u, trajs_raw, conv_flags = [], [], [], []
for pt in TRACKED_POINTS:
    tw, tu, conv = integrate_pc(pt, GUARD_OFFSET, DELTA, T_MAX,
                                DT, T_TOL, conv_tol=CONV_TOL,
                                conv_time=CONV_TIME)
    trajs_w.append(break_wraps(tw, tu))
    trajs_u.append(tu)
    trajs_raw.append(tw)
    conv_flags.append(conv)
    print(f"  {np.array(pt).round(2)}  "
          f"{'CONVERGES' if conv else 'diverges ':9s}  ({len(tw)} pts)")

if SHOW_BASIN and N_GRID > 0:
    print(f"Classifying {N_GRID**3} background points...")
    grid_pts, grid_conv = classify_grid(N_GRID, GUARD_OFFSET, DELTA)
    print(f"  {grid_conv.sum()}/{len(grid_conv)} converge "
          f"({100*grid_conv.sum()/len(grid_conv):.1f}%)")

# ── Figure ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
ax  = fig.add_subplot(111, projection='3d')
ax.set_xlim(-PI,PI); ax.set_ylim(-PI,PI); ax.set_zlim(-PI,PI)
ax.set_xlabel('$x_1$',fontsize=11); ax.set_ylabel('$x_2$',fontsize=11)
ax.set_zlabel('$x_3$',fontsize=11)

if SHOW_BASIN and N_GRID > 0:
    cp,ncp = grid_pts[grid_conv], grid_pts[~grid_conv]
    if len(cp):  ax.scatter(cp[:,0], cp[:,1], cp[:,2], c='green',s=4,alpha=0.25)
    if len(ncp): ax.scatter(ncp[:,0],ncp[:,1],ncp[:,2],c='red',  s=4,alpha=0.08)

if SHOW_SIMPLEX:
    tri = Poly3DCollection([[np.array([1,0,0]),np.array([0,1,0]),
                             np.array([0,0,1])]], alpha=0.15)
    tri.set_facecolor('royalblue'); tri.set_edgecolor('royalblue')
    ax.add_collection3d(tri)

lo, hi, c = -PI, PI, GUARD_OFFSET
for col, plane in zip(['darkorange','purple','saddlebrown'],
                      [([c,c,c,c],[lo,lo,hi,hi],[lo,hi,hi,lo]),
                       ([lo,lo,hi,hi],[c,c,c,c],[lo,hi,hi,lo]),
                       ([lo,lo,hi,hi],[lo,hi,hi,lo],[c,c,c,c])]):
    sq = Poly3DCollection([list(zip(*plane))], alpha=0.06)
    sq.set_facecolor(col); sq.set_edgecolor(col); ax.add_collection3d(sq)

t_d = np.linspace(-PI, PI/3+0.1, 100)
ax.plot(t_d,t_d,t_d,'k-',lw=2.5,label='Diagonal')
ax.scatter([1/3],[1/3],[1/3],color='black',s=60,zorder=11)

COLORS = ['#1f77b4','#ff7f0e','#9467bd','#8c564b',
          '#e377c2','#17becf','#bcbd22','#2ca02c']

for i,(pt,conv) in enumerate(zip(TRACKED_POINTS,conv_flags)):
    ax.scatter(*pt, color=COLORS[i%len(COLORS)],
               marker='^' if conv else 'x', s=80, zorder=12,
               label=f'P{i+1} {np.array(pt).round(1)} {"✓" if conv else "✗"}')

# ── Animation ─────────────────────────────────────────────────────
# Matplotlib 3D does NOT break lines at NaN via set_data/set_3d_properties.
# Fix: split tail into separate continuous segments at each torus wrap,
# draw each segment as its own Line3D artist (same idea as syncTime4D.py).
if ANIMATE:
    MAX_SEGS = 8   # max separate trail segments per trajectory
    seg_lines = []
    heads = []
    for i in range(len(TRACKED_POINTS)):
        col = COLORS[i % len(COLORS)]
        lw  = 2.0 if conv_flags[i] else 1.0
        segs = [ax.plot([],[],[], color=col, lw=lw, alpha=0.85)[0]
                for _ in range(MAX_SEGS)]
        seg_lines.append(segs)
        head, = ax.plot([],[],[], 'o', color=col, ms=8, zorder=20)
        heads.append(head)

    max_raw = max(len(t) for t in trajs_raw)

    def get_segments(raw_seg, u_seg):
        """Split raw_seg at torus wrap points detected via unwrapped u_seg."""
        if len(raw_seg) < 2:
            return [raw_seg]
        tiles    = np.floor((u_seg + PI) / (2*PI)).astype(int)
        jump_idx = np.where(np.any(np.diff(tiles, axis=0) != 0, axis=1))[0]
        if len(jump_idx) == 0:
            return [raw_seg]
        cuts = np.concatenate([[0], jump_idx + 1, [len(raw_seg)]])
        return [raw_seg[cuts[k]:cuts[k+1]] for k in range(len(cuts)-1)
                if cuts[k+1] > cuts[k]]

    def update(frame):
        f = frame * SPEED
        all_artists = []
        for i, (tr, tu, segs, head) in enumerate(
                zip(trajs_raw, trajs_u, seg_lines, heads)):
            idx   = min(f, len(tr)-1)
            start = max(0, idx - TAIL)

            # head: always at current raw position
            head.set_data([tr[idx, 0]], [tr[idx, 1]])
            head.set_3d_properties([tr[idx, 2]])

            # split tail into continuous segments — no wraps within any segment
            parts = get_segments(tr[start:idx+1], tu[start:idx+1])

            for k, ln in enumerate(segs):
                if k < len(parts) and len(parts[k]) > 1:
                    p = parts[k]
                    ln.set_data(p[:, 0], p[:, 1])
                    ln.set_3d_properties(p[:, 2])
                else:
                    ln.set_data([], [])
                    ln.set_3d_properties([])
            all_artists += segs + [head]

        ax.set_title(f'Basin explorer  |  $c={GUARD_OFFSET:.2f}$  |  '
                     f'$\\delta={DELTA}$', fontsize=11)
        return all_artists

    n_frames = max_raw // SPEED + 1
    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=15, blit=False, repeat=False)

    ax.legend(fontsize=7, loc='upper left', ncol=2)
    plt.tight_layout()

    if SAVE_MP4:
        print("Saving...")
        ani.save('basin_animation.mp4', writer='ffmpeg', fps=30, dpi=150)
        print("Saved.")
    else:
        plt.show()
else:
    for i,tw in enumerate(trajs_w):
        ax.plot(tw[:,0],tw[:,1],tw[:,2],
                color=COLORS[i%len(COLORS)],
                lw=2.0 if conv_flags[i] else 1.0, alpha=0.8)
    ax.legend(fontsize=7,loc='upper left',ncol=2)
    plt.tight_layout(); plt.show()