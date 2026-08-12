import numpy as np
import matplotlib.pyplot as plt

# --- Model (D=3) in x-coordinates ---
def rhs_x_3(x, delta=0.5):
    """
    x in R^3.
    Rule: dx_i = 1 if (x_i < 0) AND (sum_j x_j > 1), else 1-delta.
    """
    s = x.sum()
    dx = np.full(3, 1.0 - delta, dtype=float)
    if s > 1.0:
        neg = x < 0
        dx[neg] = 1.0
    return dx

def simulate_euler(x0, delta=0.5, dt=1e-3, T=50.0):
    """Simple Euler integration (OK here since vector field is piecewise constant)."""
    x = np.array(x0, dtype=float)
    nsteps = int(T / dt)
    for _ in range(nsteps):
        x += dt * rhs_x_3(x, delta)
    return x

def d_perp(x):
    """Distance to diagonal: ||x - mean(x)*1||."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    return np.linalg.norm(x - m*np.ones(3))

# --- Basin map in transverse plane V0 via v=(a,b,-a-b) ---
def basin_ratio_map(delta=0.5, A=0.15, n=4, dt=1e-2, T=5.0):
    """
    Returns:
      a_vals, b_vals, R where R(a,b)=d_perp(x(T))/d_perp(x(0)).
    """
    a_vals = np.linspace(-A, A, n)
    b_vals = np.linspace(-A, A, n)

    x_star = np.array([1/3, 1/3, 1/3], dtype=float)  # point on diagonal at sum-guard
    R = np.full((n, n), np.nan, dtype=float)

    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            v = np.array([a, b, -a-b], dtype=float)     # v in V0 automatically
            x0 = x_star + v

            d0 = d_perp(x0)
            if d0 < 1e-12:
                R[j, i] = 0.0
                continue

            xT = simulate_euler(x0, delta=delta, dt=dt, T=T)
            dT = d_perp(xT)
            R[j, i] = dT / d0

    return a_vals, b_vals, R

def triangle_vertices():
    """
    Triangle in (a,b) describing x_i>=0 at x*= (1/3,1/3,1/3):
      a >= -1/3
      b >= -1/3
      a+b <= 1/3
    Vertices are:
      (-1/3,-1/3), (1/3,-1/3), (-1/3,1/3)
    """
    return np.array([[-1/3, -1/3],
                     [ 1/3, -1/3],
                     [-1/3,  1/3],
                     [-1/3, -1/3]], dtype=float)



if __name__ == "__main__":
    # ---------- Run and plot ----------
    delta = 0.5

    # Zoom in: you WILL see the contracting "blob" near (0,0) if it exists
    A = 0.2     # try 0.15 if still too coarse
    n = 360     # resolution
    dt = 1e-3
    T = 80.0    # increase to 150 if you want stronger separation

    a_vals, b_vals, R = basin_ratio_map(delta=delta, A=A, n=n, dt=dt, T=T)

    # Plot 1: contraction ratio map (log scale helps)
    plt.figure(figsize=(7, 6))
    plt.imshow(np.log10(R), origin="lower",
              extent=[a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]],
              aspect="equal")
    plt.colorbar(label=r"$\log_{10}\left(d_\perp(T)/d_\perp(0)\right)$")
    plt.xlabel("a")
    plt.ylabel("b")
    plt.title("Transverse contraction map in $V_0$ (D=3)")

    tri = triangle_vertices()
    plt.plot(tri[:,0], tri[:,1], linewidth=2)

    # Mark the origin (exact diagonal)
    plt.scatter([0],[0], s=30)

    plt.show()
    plt.savefig("contraction_ratio_map.png", dpi=300)

    # Plot 2: a clean basin-style mask (choose a threshold)
    # "Contracting" if distance shrinks by at least 10% over time T
    threshold = 0.9
    mask = (R < threshold)

    plt.figure(figsize=(7, 6))
    plt.imshow(mask.astype(int), origin="lower",
              extent=[a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]],
              aspect="equal")
    plt.xlabel("a")
    plt.ylabel("b")
    plt.title(f"Contracting region (R < {threshold}) in $V_0$ (D=3)")

    plt.plot(tri[:,0], tri[:,1], linewidth=2)
    plt.scatter([0],[0], s=30)
    plt.show()
    plt.savefig("contracting_region.png", dpi=300)

