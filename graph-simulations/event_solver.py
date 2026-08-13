import numpy as np
import networkx as nx

TWO_PI = 2 * np.pi
PI = np.pi
CONV_TOL = 0.05
CONV_TIME = 10.0
ATOL = 1e-12
RTOL = 1e-10


def spread(x):
    center = np.arctan2(np.sin(x).mean(), np.cos(x).mean())
    return float(np.abs(((x - center + PI) % TWO_PI) - PI).max())


def coincident(a, b):
    return abs(a - b) <= ATOL + RTOL * max(1.0, abs(b))


class EventSim:
    def __init__(self, G, c=2.2, delta=0.5):
        self.D = G.number_of_nodes()
        self.edges = list(G.edges())
        self.deg = np.array([G.degree(i) for i in range(self.D)])
        self.c = c
        self.delta = delta

    def velocity(self, above, below):
        slow = above.copy()
        for e, (i, j) in enumerate(self.edges):
            if not below[e]:
                slow[i] = False
                slow[j] = False
        slow &= self.deg > 0
        return np.where(slow, 1.0 - self.delta, 1.0)

    def flags(self, x):
        above = x > self.c
        below = np.array([x[i] + x[j] < TWO_PI for i, j in self.edges])
        return above, below

    def events(self, x, v, above, below):
        out = []
        for i in range(self.D):
            if not above[i] and x[i] < self.c:
                out.append(((self.c - x[i]) / v[i], "threshold", i))
            out.append(((TWO_PI - x[i]) / v[i], "wrap", i))

        for e, (i, j) in enumerate(self.edges):
            if below[e]:
                s = x[i] + x[j]
                vv = v[i] + v[j]
                if s < TWO_PI:
                    out.append(((TWO_PI - s) / vv, "pair", e))
        return [a for a in out if a[0] > 0]

    def last_entry(self, x, v, h):
        def g(s):
            return spread(np.mod(x + s * v, TWO_PI)) - CONV_TOL

        if g(h) >= 0:
            return None

        grid = np.linspace(0.0, h, 129)
        vals = [g(s) for s in grid]

        k = -1
        for i in range(128, -1, -1):
            if vals[i] >= 0:
                k = i
                break

        if k == -1:
            return 0.0

        lo = grid[k]
        hi = grid[k + 1]
        for _ in range(60):
            mid = (lo + hi) / 2
            if g(mid) >= 0:
                lo = mid
            else:
                hi = mid
        return hi

    def run(self, x0, t_max=4000.0):
        x = np.mod(np.asarray(x0, float), TWO_PI)
        above, below = self.flags(x)
        t = 0.0
        band_start = None
        n_events = 0

        while t < t_max:
            v = self.velocity(above, below)
            cand = self.events(x, v, above, below)
            if not cand:
                break

            h = min(a[0] for a in cand)
            active = [a for a in cand if coincident(a[0], h)]

            s0 = self.last_entry(x, v, h)
            if s0 is None:
                band_start = None
            else:
                if s0 != 0.0 or band_start is None:
                    band_start = t + s0
                if t + h - band_start >= CONV_TIME:
                    return t_sync_result(band_start, n_events)

            x2 = x + h * v
            wraps = [p for _, kind, p in active if kind == "wrap"]

            for _, kind, p in active:
                if kind == "threshold":
                    above[p] = True
                elif kind == "pair":
                    below[p] = False

            for i in wraps:
                x2[i] = 0.0
                above[i] = False
                for e, (a, b) in enumerate(self.edges):
                    if a == i or b == i:
                        below[e] = x2[a] + x2[b] < TWO_PI

            for i in range(self.D):
                if i not in wraps and x2[i] >= TWO_PI:
                    x2[i] %= TWO_PI

            x = x2
            t += h
            n_events += 1

        return {"t_sync": None, "converged": False, "n_events": n_events}


def t_sync_result(t, n):
    return {"t_sync": float(t), "converged": True, "n_events": n}


def spider(q, L):
    G = nx.Graph()
    node = 1
    for _ in range(q):
        prev = 0
        for _ in range(L):
            G.add_edge(prev, node)
            prev = node
            node += 1
    return G
