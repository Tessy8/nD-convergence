"""
2D sanity-check plot for the D=3 hybrid vector field on a fixed x3 slice.

This is the quick visual model check Professor Shai asked for: it shows the
piecewise-constant vector field projected onto the (x1, x2) plane together
with the coordinate guards and the projected sum guard.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from hybrid_tools import PI, slice_vector_field

GUARD_OFFSET = 0.0
DELTA        = 0.5
X3_SLICE     = 1.0 / 3.0
N            = 23
MARGIN       = 0.15
QUIVER_SCALE = 12.0

vals, pts, vf = slice_vector_field(X3_SLICE, N, GUARD_OFFSET, DELTA, margin=MARGIN)
x1 = pts[:, 0].reshape(N, N)
x2 = pts[:, 1].reshape(N, N)
u1 = vf[:, 0].reshape(N, N)
u2 = vf[:, 1].reshape(N, N)
fast_mask = np.isclose(u1, 1.0)
region = fast_mask.reshape(N, N).astype(float)

os.makedirs("conv-basin/output", exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(
    region.T,
    origin="lower",
    extent=[vals[0], vals[-1], vals[0], vals[-1]],
    cmap="coolwarm",
    alpha=0.25,
    vmin=0,
    vmax=1,
)
ax.quiver(x1, x2, u1, u2, region, cmap="coolwarm", angles="xy", scale_units="xy", scale=QUIVER_SCALE)

ax.axvline(GUARD_OFFSET, color="darkorange", lw=1.2, ls="--", alpha=0.8, label="$x_1 = 0$")
ax.axhline(GUARD_OFFSET, color="purple", lw=1.2, ls="--", alpha=0.8, label="$x_2 = 0$")

sum_line = 1.0 - X3_SLICE
x_line = np.array([max(-PI, sum_line - PI), min(PI, sum_line + PI)])
y_line = sum_line - x_line
ax.plot(x_line, y_line, color="steelblue", lw=2.0, label=fr"$x_1 + x_2 + x_3 = 1$, $x_3={X3_SLICE:.2f}$")

ax.plot([-PI, PI], [-PI, PI], color="black", lw=1.0, alpha=0.35, label="diagonal")
ax.scatter([X3_SLICE], [X3_SLICE], color="black", s=45, zorder=5)

ax.set_xlim(-PI, PI)
ax.set_ylim(-PI, PI)
ax.set_aspect("equal")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("2D vector field sanity check on a fixed $x_3$ slice")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig("conv-basin/output/vector_field_slice_x3.png", dpi=150, bbox_inches="tight")
print("Saved conv-basin/output/vector_field_slice_x3.png")
plt.show()
