import numpy as np
import networkx as nx

from parberry import parberry_comparators

PI = np.pi
TWO_PI = 2 * np.pi


def complete_graph(D):
    return {(i, j) for i in range(D) for j in range(i + 1, D)}


def parberry_graph(D):
    return {(min(i, j), max(i, j)) for i, j in parberry_comparators(D)}


def butterfly_graph(D):
    k = 1
    while k < D:
        k *= 2

    edges = set()
    r = 1
    while r < k:
        for a in range(0, k, 2 * r):
            for b in range(r):
                i = a + b
                j = a + b + r
                if i < D and j < D:
                    edges.add((min(i, j), max(i, j)))
        r *= 2
    return edges


def shell_graph(D):
    edges = set()
    h = D // 2
    while h >= 1:
        for i in range(D - h):
            edges.add((i, i + h))
        h //= 2
    return edges


def ring_graph(D):
    return {(min(i, (i + 1) % D), max(i, (i + 1) % D)) for i in range(D)}


def random_regular_graph(D, degree, seed):
    G = nx.random_regular_graph(degree, D, seed=seed)
    s = seed
    while not nx.is_connected(G):
        s += 1
        G = nx.random_regular_graph(degree, D, seed=s)
    return {(min(i, j), max(i, j)) for i, j in G.edges()}


def adjacency(D, edges):
    adj = [[] for _ in range(D)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def vfield(adj):
    D = len(adj)

    def f(x, c=2.2, delta=0.5):
        v = np.ones(D)
        for k in range(D):
            if x[k] <= c or not adj[k]:
                continue
            if all(x[k] + x[j] < TWO_PI for j in adj[k]):
                v[k] = 1.0 - delta
        return v

    return f


def spread(x):
    center = np.arctan2(np.sin(x).mean(), np.cos(x).mean())
    return float(np.abs(((x - center + PI) % TWO_PI) - PI).max())


def simulate(x0, adj, c=2.2, delta=0.5, t_max=200.0,
             dt=0.05, t_tol=1e-5, conv_tol=0.05, conv_time=10.0):
    f = vfield(adj)
    x = np.mod(np.asarray(x0, float), TWO_PI)
    v = f(x, c, delta)
    t = 0.0
    h = dt
    good_for = 0.0

    while t < t_max:
        x_try = x + h * v
        v_try = f(np.mod(x_try, TWO_PI), c, delta)

        if not np.array_equal(v_try, v) and h > t_tol:
            h /= 2.0
            continue

        h_used = h
        x = np.mod(x + h_used * v, TWO_PI)
        t += h_used
        v = f(x, c, delta)

        if spread(x) < conv_tol:
            good_for += h_used
            if good_for >= conv_time:
                return t - good_for
        else:
            good_for = 0.0

        h = min(h_used * 1.5, dt)

    return None


def initial_conditions(D, N, seed=42):
    rng = np.random.default_rng(seed)
    n1 = N // 2
    X1 = rng.uniform(0, TWO_PI, size=(n1, D))
    X2 = rng.uniform(0, min(TWO_PI, 3 * TWO_PI / D), size=(N - n1, D))
    return np.vstack([X1, X2])
