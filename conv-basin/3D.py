"""
Professor-requested 3D animation for the hybrid oscillator (D=3).

What is shown per panel:
  - The cube [-pi, pi]^3 (torus fundamental domain)
  - Three coordinate guard planes  x_i = c  (one per axis)
  - The simplex face  x1+x2+x3 = 1  as a filled triangle
  - The triangular intersection of the three guard planes with the simplex
    face — the "blue triangle" the professor drew
  - The diagonal line  x1=x2=x3  and the fixed point x* = (1/3,1/3,1/3)
  - 10 trajectories as dots-with-trails; trails break at torus wraps (NaNs)
  - Dot flashes with a black ring when it crosses a guard surface
  - Title updates each frame to show current simulation time

One MP4 is saved per guard offset c.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hybrid_tools import PI, integrate_pc, vfield_x

# ── global parameters ─────────────────────────────────────────────────────────
DELTA       = 0.5
T_MAX       = 150.0
DT          = 0.05
T_TOL       = 1e-5
CONV_TOL    = 0.05
CONV_TIME   = 10.0
STORE_EVERY = 4
TAIL        = 8     # trail length in stored steps
SPEED       = 1     # stored steps advanced per animation frame
FPS         = 24
OUTPUT_DIR  = "conv-basin/output/"

# Change these to produce different animations
GUARD_OFFSETS = [0.00, 0.15, 0.25]
GUARD_LABELS  = ["c = 0.00  (standard)", "c = 0.15", "c = 0.25"]

# 10 representative initial conditions (same as professor_animation.py)
POINT_SPECS = [
    ((1/3,  1/3,  1/3),   "diagonal x*"),
    ((0.28, 0.28, 0.28),  "deep in simplex"),
    ((0.18, 0.18, 0.55),  "straddles c=0.25"),
    ((0.05, 0.05, 0.70),  "straddles c=0.15"),
    ((0.08, 0.20, 0.63),  "layered transitions"),
    ((-0.20, 0.40, 0.40), "one coord negative"),
    ((-2.99,-2.99, 0.05), "wrap, x3 near c=0"),
    ((-2.99,-2.99, 0.20), "wrap, x3 near c=0.25"),
    ((-2.99,-2.99, 0.28), "wrap, x3 always slow"),
    ((2.80,  0.20, 0.20), "positive edge wrap"),
]

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ── helper functions ──────────────────────────────────────────────────────────

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


def break_trail_3d(tw, tu):
    """
    tw : (N,3) wrapped trajectory in [-pi,pi]^3
    tu : (N,3) unwrapped trajectory in R^3
    Returns (N',3) array with NaN rows inserted wherever a torus wrap occurs.
    NaN rows cause matplotlib to lift the pen — clean trail breaks.
    """
    if len(tw) < 2:
        return tw.copy()
    tiles  = np.floor((tu + PI) / (2.0 * PI)).astype(int)
    jumps  = np.where(np.any(np.diff(tiles, axis=0) != 0, axis=1))[0]
    if len(jumps) == 0:
        return tw.copy()
    out = tw.tolist()
    nan_row = [np.nan, np.nan, np.nan]
    for j in jumps[::-1]:
        out.insert(j + 1, nan_row)
    return np.array(out)


def guard_cross_mask(traj, guard_offset):
    """True at index i when trajectory crossed a guard between step i-1 and i."""
    if len(traj) <= 1:
        return np.zeros(len(traj), dtype=bool)
    coord_cross = np.any(
        (traj[:-1] < guard_offset) != (traj[1:] < guard_offset), axis=1)
    sum_cross = (
        (traj[:-1].sum(axis=1) > 1.0) != (traj[1:].sum(axis=1) > 1.0))
    mask = np.zeros(len(traj), dtype=bool)
    mask[1:] = coord_cross | sum_cross
    return mask


# ── static 3D decorations ─────────────────────────────────────────────────────

def draw_simplex_face(ax):
    """
    The simplex face  x1+x2+x3=1, x_i>=0  is a triangle with vertices
    at (1,0,0), (0,1,0), (0,0,1).  Shown in translucent royalblue.
    """
    verts = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)]
    tri = Poly3DCollection(verts, alpha=0.18)
    tri.set_facecolor("royalblue")
    tri.set_edgecolor("royalblue")
    tri.set_linewidth(1.2)
    ax.add_collection3d(tri)


def draw_guard_planes(ax, c):
    """
    Three coordinate guard planes  x_i = c  clipped to [-pi,pi]^3.
    Each plane is a square at x_i = c over the other two dims.
    Colors: x1-plane=darkorange, x2-plane=purple, x3-plane=saddlebrown.
    """
    lo, hi = -PI, PI
    specs = [
        ("darkorange", [c, c, c, c],   [lo, lo, hi, hi], [lo, hi, hi, lo]),
        ("purple",     [lo, lo, hi, hi],[c,  c,  c,  c],  [lo, hi, hi, lo]),
        ("saddlebrown",[lo, lo, hi, hi],[lo, hi, hi, lo],  [c,  c,  c,  c]),
    ]
    for col, xs, ys, zs in specs:
        verts = [list(zip(xs, ys, zs))]
        sq = Poly3DCollection(verts, alpha=0.07)
        sq.set_facecolor(col)
        sq.set_edgecolor(col)
        sq.set_linewidth(0.6)
        ax.add_collection3d(sq)


def draw_guard_simplex_intersection(ax, c):
    """
    The intersection of the three guard planes with the simplex face is a
    smaller triangle — the 'blue triangle' the professor sketched.
    Its vertices are at:
      (c, c, 1-2c),  (c, 1-2c, c),  (1-2c, c, c)
    only valid when 1-2c >= c, i.e. c <= 1/3.
    This is the guaranteed convergence region: any trajectory landing here
    will converge.
    """
    if c > 1.0 / 3.0 + 1e-6:
        return   # triangle degenerates
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
    """
    Diagonal line x1=x2=x3 from (-pi,-pi,-pi) to (pi/3, pi/3, pi/3).
    Fixed point x* = (1/3,1/3,1/3) marked as a black star.
    """
    t = np.linspace(-PI, 1.0 / 3.0 + 0.05, 120)
    ax.plot(t, t, t, "k-", lw=2.2, alpha=0.9, zorder=10, label="diagonal")
    ax.scatter([1/3], [1/3], [1/3], color="black", s=80,
               marker="*", zorder=15, label="$x^*$")


def setup_ax(ax, title, c):
    """Configure one 3D axis with all static decorations."""
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


# ── main loop: one animation per guard value ──────────────────────────────────

for guard_offset, guard_label in zip(GUARD_OFFSETS, GUARD_LABELS):

    print(f"\n=== Integrating trajectories for {guard_label} ===")
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
            stop_on_convergence=False,
        )
        label = classify_label(res)
        results.append({
            "point":      np.asarray(pt, dtype=float),
            "short_name": short_name,
            "label":      label,
            "color":      color,
            "traj_w":     res["traj_wrapped"],    # (N,3) in [-pi,pi]
            "traj_u":     res["traj_unwrapped"],  # (N,3) unwrapped
            "cross_mask": guard_cross_mask(res["traj_wrapped"], guard_offset),
        })
        print(f"  {short_name:22s}  {np.round(pt,2)}  ->  {label}")

    # ── figure layout: one large 3D panel + info panel ────────────────────────
    fig = plt.figure(figsize=(16, 9))
    ax3d   = fig.add_subplot(121, projection="3d")
    ax_info = fig.add_subplot(122)

    setup_ax(ax3d, f"3D hybrid oscillator  |  {guard_label}  |  δ=0.5", guard_offset)

    # Trajectory legend
    legend_handles = [
        Line2D([0], [0], color=COLORS[i], marker="o", lw=1.5, markersize=5,
               label=f"P{i+1:02d} {POINT_SPECS[i][1]}")
        for i in range(len(POINT_SPECS))
    ]
    # Decoration legend
    deco_handles = [
        Line2D([0], [0], color="royalblue",    lw=3, alpha=0.5,
               label="Simplex face  x₁+x₂+x₃=1"),
        Line2D([0], [0], color="dodgerblue",   lw=3, alpha=0.9,
               label="Inner triangle (guard∩simplex)"),
        Line2D([0], [0], color="darkorange",   lw=2, ls="--",
               label="Guard plane  x₁=c"),
        Line2D([0], [0], color="purple",       lw=2, ls="--",
               label="Guard plane  x₂=c"),
        Line2D([0], [0], color="saddlebrown",  lw=2, ls="--",
               label="Guard plane  x₃=c"),
        Line2D([0], [0], color="black",        lw=2,
               label="Diagonal  x₁=x₂=x₃"),
        Line2D([0], [0], color="black",        marker="*", ms=10, lw=0,
               label="Fixed point x*=(1/3,1/3,1/3)"),
    ]
    ax3d.legend(handles=legend_handles + deco_handles,
                fontsize=6.5, loc="upper left", ncol=1)

    # Mark initial conditions as triangles
    for item in results:
        pt = item["point"]
        ax3d.scatter([pt[0]], [pt[1]], [pt[2]],
                     color=item["color"], marker="^", s=40, alpha=0.8, zorder=5)

    # Info panel
    ax_info.axis("off")
    info_lines = [
        f"Guard offset  c = {guard_offset:.2f}",
        "δ = 0.5     T_max = 120 s",
        "",
        "Blue triangle = inner convergence region",
        "  (guard planes ∩ simplex face)",
        "Blue face     = full simplex x₁+x₂+x₃≤1",
        "Orange/purple/brown planes = x_i = c",
        "Black star    = fixed point x*",
        "Trail break   = torus wrap event",
        "Ring on dot   = guard surface crossing",
        "Triangle marker = initial condition",
        "",
        "",
        "Point classifications:",
        "  ✓ = converges   ✗ = does not converge",
        "  Route shown on second line",
        "-" * 38,
    ]

    # Route short labels — what road did it take to converge?
    ROUTE_SHORT = {
        "starts in simplex, converges":   "via simplex (started inside)",
        "late simplex entry, no wrap":    "via slow drift into simplex",
        "wrapped-entry -> inner basin":   "via torus wrap -> inner basin",
        "converges":                      "converges (other route)",
        "does not converge":              "does not converge",
    }

    for i, item in enumerate(results, 1):
        converged = item["label"] != "does not converge"
        conv_sym  = "✓" if converged else "✗"
        route     = ROUTE_SHORT.get(item["label"], item["label"])
        info_lines.append(
            f"{conv_sym} P{i:02d} {item['short_name'][:18]:18s}")
        info_lines.append(
            f"     {route}")
    time_text = ax_info.text(
        0.02, 0.98, "\n".join(info_lines),
        va="top", ha="left", fontsize=8.5, family="monospace",
        transform=ax_info.transAxes)

    # ── animated artists ──────────────────────────────────────────────────────
    # One Line3D trail + one dot per trajectory
    trail_lines = []
    head_dots   = []
    for item in results:
        line, = ax3d.plot([], [], [], "-",
                          color=item["color"], alpha=0.75, lw=1.8, zorder=3)
        dot,  = ax3d.plot([], [], [], "o",
                          color=item["color"], ms=6, zorder=6)
        trail_lines.append(line)
        head_dots.append(dot)

    max_len  = max(len(item["traj_w"]) for item in results)
    n_frames = max_len // SPEED + 1

    # Frame counter text on the 3D axis
    time_annot = ax3d.text2D(
        0.02, 0.96, "t = 0.00",
        transform=ax3d.transAxes, fontsize=9,
        color="black", family="monospace")

    def make_init(trail_lines, head_dots):
        def init():
            for ln, dot in zip(trail_lines, head_dots):
                ln.set_data([], [])
                ln.set_3d_properties([])
                dot.set_data([], [])
                dot.set_3d_properties([])
            return trail_lines + head_dots + [time_annot]
        return init

    def make_update(results, trail_lines, head_dots):
        def update(frame):
            f = frame * SPEED
            art = [time_annot]

            # approximate time shown (use first trajectory's time array)
            t_approx = min(f * DT * STORE_EVERY, T_MAX)
            time_annot.set_text(f"t ≈ {t_approx:.1f} s")

            for k, (item, ln, dot) in enumerate(
                    zip(results, trail_lines, head_dots)):
                tw = item["traj_w"]
                tu = item["traj_u"]
                idx   = min(f, len(tw) - 1)
                start = max(0, idx - TAIL)

                # build trail with NaN breaks at torus wraps
                trail_seg = break_trail_3d(
                    tw[start:idx + 1],
                    tu[start:idx + 1],
                )
                ln.set_data(trail_seg[:, 0], trail_seg[:, 1])
                ln.set_3d_properties(trail_seg[:, 2])

                # head position
                hx, hy, hz = tw[idx]
                dot.set_data([hx], [hy])
                dot.set_3d_properties([hz])

                # flash ring at guard crossing
                if item["cross_mask"][idx]:
                    dot.set_markeredgecolor("black")
                    dot.set_markeredgewidth(2.0)
                    dot.set_markersize(9)
                else:
                    dot.set_markeredgecolor(item["color"])
                    dot.set_markeredgewidth(0)
                    dot.set_markersize(6)

                art += [ln, dot]
            return art
        return update

    ani = animation.FuncAnimation(
        fig,
        make_update(results, trail_lines, head_dots),
        frames=n_frames,
        init_func=make_init(trail_lines, head_dots),
        interval=1000 / FPS,
        blit=False,   # blit=False needed for 3D axes in matplotlib
        repeat=False,
    )

    safe = guard_label.replace(" ", "").replace("=","").replace(".","p") \
                      .replace("(","").replace(")","").replace("-","neg")
    save_path = f"{OUTPUT_DIR}professor_animation_3d_{safe}.mp4"
    plt.tight_layout()
    print(f"Saving {save_path} ...")
    ani.save(save_path, writer="ffmpeg", fps=FPS, dpi=150, bitrate=3000)
    print(f"Saved  {save_path}")
    plt.close(fig)

print("\nAll done.")
