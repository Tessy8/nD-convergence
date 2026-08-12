import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 1. VISUALIZE THE 3D SIMPLEX (S3)
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')

# Vertices of the simplex: x_i >= 0 and sum(x_i) <= 1
v0 = [0, 0, 0]
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [0, 0, 1]

# Define the 4 triangular faces of the tetrahedron
faces = [[v0, v1, v2], [v0, v1, v3], [v0, v2, v3], [v1, v2, v3]]
simplex = Poly3DCollection(faces, alpha=0.25, edgecolor='k', facecolor='cyan')
ax1.add_collection3d(simplex)

# The diagonal orbit line
ax1.plot([0, 0.8], [0, 0.8], [0, 0.8], color='red', lw=2, label='Diagonal Orbit')
ax1.scatter([1/3], [1/3], [1/3], color='red', s=50, label='Crossing Point $x^*$')

ax1.set_title("Simplex $S_3$ in State Space")
ax1.set_xlabel('$x_1$'); ax1.set_ylabel('$x_2$'); ax1.set_zlabel('$x_3$')
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_zlim(0, 1)
ax1.legend()

# 2. VISUALIZE THE TRANSVERSE SECTION (The Basin "Target")
ax2 = fig.add_subplot(122)

# Basis for the plane x1 + x2 + x3 = 1
u1 = np.array([1, -1, 0]) / np.sqrt(2)
u2 = np.array([1, 1, -2]) / np.sqrt(6)

# Project the vertices of the triangle x1+x2+x3=1 into 2D
center = np.array([1/3, 1/3, 1/3])
def project(v):
    v_shifted = v - center
    return np.array([np.dot(v_shifted, u1), np.dot(v_shifted, u2)])

p1, p2, p3 = project(np.array(v1)), project(np.array(v2)), project(np.array(v3))
triangle_pts = np.array([p1, p2, p3])

# Draw the triangle
poly_2d = plt.Polygon(triangle_pts, fill=True, color='blue', alpha=0.1, edgecolor='blue', label='Basin Cross-section')
ax2.add_patch(poly_2d)

# Draw the largest inscribed ball (the 'Core' stability zone)
r_max = 1 / np.sqrt(6)
circle = plt.Circle((0, 0), r_max, color='green', fill=False, lw=2, linestyle='--', label='Max Radius $r_{max}$')
ax2.add_patch(circle)

ax2.scatter([0], [0], color='red', label='Diagonal Center')
ax2.set_aspect('equal')
ax2.set_xlim(-0.7, 0.7); ax2.set_ylim(-0.7, 0.7)
ax2.set_title("Transverse Plane at Sum Guard")
ax2.set_xlabel("Transverse $v_1$"); ax2.set_ylabel("Transverse $v_2$")
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# plt.savefig("simplex_and_transverse_section.png", dpi=300)