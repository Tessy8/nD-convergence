import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) Your D=3 hybrid model in x-coordinates
# ----------------------------
def rhs_x(x, delta=0.5):
    """
    x in R^3
    dx_i = 1 if (x_i < 0 and sum(x) > 1), else 1-delta
    """
    s = x.sum()
    dx = np.full(3, 1.0 - delta)
    if s > 1.0:
        mask = (x < 0.0)
        dx[mask] = 1.0
    return dx

# ----------------------------
# 2) Distance-to-diagonal (transverse distance)
# ----------------------------
def d_perp(x):
    xbar = x.mean()
    return np.linalg.norm(x - xbar * np.ones(3))

# ----------------------------
# 3) Simple integrator (RK4 works nicely for piecewise-constant too)
# ----------------------------
def simulate_rk4(x0, delta=0.5, dt=1e-3, T=50.0):
    x = x0.astype(float).copy()
    nsteps = int(np.ceil(T / dt))
    for _ in range(nsteps):
        k1 = rhs_x(x, delta)
        k2 = rhs_x(x + 0.5 * dt * k1, delta)
        k3 = rhs_x(x + 0.5 * dt * k2, delta)
        k4 = rhs_x(x + dt * k3, delta)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x

# ----------------------------
# 4) Build a "contraction ratio map" over the transverse plane V0
#    v(a,b) = (a, b, -a-b) so v1+v2+v3=0 automatically
# ----------------------------
def contraction_ratio_map(
    delta=0.5,
    A=0.20,          # zoom size: try 0.15, 0.20, 0.30
    n=401,           # resolution
    dt=1e-3,
    T=80.0,          # simulate longer so contraction shows
    eps0=1e-10       # avoid division by zero near origin
):
    a_vals = np.linspace(-A, A, n)
    b_vals = np.linspace(-A, A, n)

    x_star = np.array([1/3, 1/3, 1/3], dtype=float)

    R = np.full((n, n), np.nan)  # ratio map: dT/d0
    d0_map = np.full((n, n), np.nan)
    dT_map = np.full((n, n), np.nan)

    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            v = np.array([a, b, -a-b], dtype=float)
            x0 = x_star + v

            d0 = d_perp(x0)
            # if essentially on the diagonal already, define ratio = 0
            if d0 < eps0:
                R[j, i] = 0.0
                d0_map[j, i] = d0
                dT_map[j, i] = 0.0
                continue

            xT = simulate_rk4(x0, delta=delta, dt=dt, T=T)
            dT = d_perp(xT)

            R[j, i] = dT / d0
            d0_map[j, i] = d0
            dT_map[j, i] = dT

    return a_vals, b_vals, R, d0_map, dT_map

# ----------------------------
# 5) Plot: ratio map + "contracting region" overlay + triangle overlay
# ----------------------------
def plot_basin_like(delta=0.5, A=0.15, n=81, dt=1e-2, T=30.0,
                    contract_thresh=0.9):
    a_vals, b_vals, R, *_ = contraction_ratio_map(
        delta=delta, A=A, n=n, dt=dt, T=T
    )

    # Make two plots:
    # (i) continuous ratio map
    # (ii) binary "contracting" region where R < contract_thresh
    extent = [a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]]

    plt.figure(figsize=(8, 6))
    # log helps reveal small contraction differences
    # clip to avoid log(0)
    R_clip = np.clip(R, 1e-6, 1e6)
    img = plt.imshow(np.log10(R_clip), origin="lower", extent=extent, aspect="equal")
    plt.colorbar(img, label=r"$\log_{10}(d_\perp(T)/d_\perp(0))$")
    plt.xlabel("a")
    plt.ylabel("b")
    plt.title(f"Contraction ratio map in V0 (D=3), delta={delta}, T={T}")

    # Overlay triangle = simplex slice: a>=-1/3, b>=-1/3, a+b<=1/3
    tri = np.array([
        [-1/3, -1/3],
        [ 1/3, -1/3],
        [-1/3,  1/3],
        [-1/3, -1/3]
    ])
    plt.plot(tri[:, 0], tri[:, 1], linewidth=2)

    plt.show()
    plt.savefig("contraction_ratio_map_copy.png", dpi=300)

    # Binary contracting region
    plt.figure(figsize=(8, 6))
    contracting = (R < contract_thresh).astype(float)
    img2 = plt.imshow(contracting, origin="lower", extent=extent, aspect="equal")
    plt.xlabel("a")
    plt.ylabel("b")
    plt.title(f"Region where d_perp shrinks: R < {contract_thresh} (delta={delta}, T={T})")
    plt.plot(tri[:, 0], tri[:, 1], linewidth=2)
    plt.show()
    plt.savefig("contracting_region_copy.png", dpi=300)



if __name__ == "__main__":
    # ----------------------------
    # RUN IT
    # ----------------------------
    plot_basin_like(
        delta=0.5,
        A=0.20,          # start zoomed in so you can SEE the basin
        n=401,
        dt=1e-3,
        T=120.0,         # increase if contraction is slow
        contract_thresh=0.9
    )

