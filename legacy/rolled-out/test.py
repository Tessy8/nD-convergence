import numpy as np

# ---------- Single-agent saltation pieces (same as in the note) ----------

def xi_coord(D: int, i: int, delta: float) -> np.ndarray:
    """
    Saltation matrix for crossing the coordinate guard x_i = 0
    (fast -> slow on coordinate i only):
        Xi_{x_i=0} = I - delta * e_i e_i^T
    """
    I = np.eye(D)
    e = np.zeros((D, 1))
    e[i, 0] = 1.0
    Xi = I - delta * (e @ e.T)
    return Xi


def xi_sum_diag(D: int, delta: float) -> np.ndarray:
    """
    Saltation matrix at the sum guard along the diagonal:
        Xi_sum,diag = I + [δ / (D(1-δ))] 1 1^T
    where 1 is the all-ones vector in R^D.
    """
    I = np.eye(D)
    one = np.ones((D, 1))
    Xi = I + (delta / (D * (1.0 - delta))) * (one @ one.T)
    return Xi


def J_single_agent(D: int, delta: float) -> np.ndarray:
    """
    One-cycle Jacobian J for a single D-dimensional agent,
    using the product
        J = Xi_sum_diag * ∏_i Xi_{x_i=0}
    (coordinate-guard matrices commute, so order does not matter).
    """
    # product over all coordinate guards
    Xi_coords = np.eye(D)
    for i in range(D):
        Xi_coords = xi_coord(D, i, delta) @ Xi_coords

    # sum-guard along the diagonal
    Xi_sum = xi_sum_diag(D, delta)

    # one-cycle map
    J = Xi_sum @ Xi_coords
    return J


# ---------- Four-agent Jacobian (block diagonal) ----------

def J_four_agents(D: int, delta: float) -> np.ndarray:
    """
    Ideal one-cycle Jacobian for 4 uncoupled agents:
      J4 = diag(J, J, J, J)
    where J is the single-agent D×D Jacobian.
    """
    J1 = J_single_agent(D, delta)
    # block-diagonal with 4 copies of J1
    J4 = np.kron(np.eye(4), J1)
    return J4


def analyze_four_agents(D: int, delta: float) -> None:
    J1 = J_single_agent(D, delta)
    J4 = J_four_agents(D, delta)

    eig1, _ = np.linalg.eig(J1)
    eig4, _ = np.linalg.eig(J4)

    # sort by real part for readability
    eig1 = np.real_if_close(eig1)
    eig4 = np.real_if_close(eig4)
    eig1_sorted = np.sort(eig1)
    eig4_sorted = np.sort(eig4)

    print(f"\n==============================")
    print(f"D = {D}, δ = {delta}")
    print("Single-agent J eigenvalues:")
    print(eig1_sorted)
    print(f"  -> one ≈ 1, {D-1} ≈ {1.0 - delta}")

    print("\nFour-agent block J4 eigenvalues:")
    print(eig4_sorted)
    print(f"  -> four ≈ 1, {4*(D-1)} ≈ {1.0 - delta}")


if __name__ == "__main__":
    delta = 0.5
    for D in [2, 4, 7]:
        analyze_four_agents(D, delta)
