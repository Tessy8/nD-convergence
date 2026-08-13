import json
import numpy as np

import graph_guards as gg

N = 200
DIMS = [6, 8]
C_VALUES = [1.5, 1.8, 2.0, 2.2]
DELTA = 0.5
T_MAX = 200.0
SEED = 42

GRAPH_BUILDERS = {
    "complete": lambda D: gg.complete_graph(D),
    "parberry": lambda D: gg.parberry_graph(D),
    "butterfly": lambda D: gg.butterfly_graph(D),
    "shell": lambda D: gg.shell_graph(D),
    "ring": lambda D: gg.ring_graph(D),
    "random_2": lambda D: gg.random_regular_graph(D, 2, 1200 + D),
    "random_3": lambda D: gg.random_regular_graph(D, 3, 1300 + D),
    "random_4": lambda D: gg.random_regular_graph(D, 4, 1400 + D),
}

out = {}

for D in DIMS:
    X = gg.initial_conditions(D, N, SEED)
    out[str(D)] = {}

    for name, build in GRAPH_BUILDERS.items():
        edges = build(D)
        adj = gg.adjacency(D, edges)
        by_c = {}

        for c in C_VALUES:
            times = [gg.simulate(x0, adj, c, DELTA, T_MAX) for x0 in X]
            good = [t for t in times if t is not None]
            by_c[str(c)] = {
                "converged": len(good),
                "percent": 100 * len(good) / N,
                "median_time": float(np.median(good)) if good else None,
            }

        out[str(D)][name] = by_c
        best = max(v["percent"] for v in by_c.values())
        print(D, name, f"{best:.1f}%")

with open("graph_results.json", "w") as f:
    json.dump(out, f, indent=2)
