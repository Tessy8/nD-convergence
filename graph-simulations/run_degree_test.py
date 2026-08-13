import json
import networkx as nx
import numpy as np

import graph_guards as gg
from degree_graphs import families

N = 200
DIMS = [16, 32]
C_VALUES = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2]
DELTA = 0.5
T_MAX = 200.0
SEED = 42

out = {}

for D in DIMS:
    X = gg.initial_conditions(D, N, SEED)
    out[str(D)] = {}

    for name, G in families(D).items():
        edges = {(min(i, j), max(i, j)) for i, j in G.edges()}
        adj = gg.adjacency(D, edges)

        row = {
            "degree": int(dict(G.degree())[0]),
            "diameter": nx.diameter(G),
            "apl": nx.average_shortest_path_length(G),
            "by_c": {},
        }

        for c in C_VALUES:
            times = [gg.simulate(x0, adj, c, DELTA, T_MAX) for x0 in X]
            good = [t for t in times if t is not None]
            row["by_c"][str(c)] = 100 * len(good) / N

        out[str(D)][name] = row
        print(D, name, row["diameter"], max(row["by_c"].values()))

with open("degree_results.json", "w") as f:
    json.dump(out, f, indent=2)
