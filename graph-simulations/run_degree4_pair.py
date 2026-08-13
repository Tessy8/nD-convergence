import json
from multiprocessing import Pool

import networkx as nx
import numpy as np

from degree_graphs import bottleneck_4, circulant_4
from event_solver import EventSim

DIMS = [16, 32, 64]
N = 300
SEED = 4042
W = 12 * 2 * np.pi / 32
T_CYCLE = 7.225
WORKERS = 3

SIM = None


def init_worker(G):
    global SIM
    SIM = EventSim(G)


def run_one(x0):
    return SIM.run(x0)["t_sync"]


out = {}

for D in DIMS:
    X = np.random.default_rng(SEED).uniform(0, W, size=(N, D))
    graphs = {
        "d4_bottleneck": bottleneck_4(D),
        "d4_circulant": circulant_4(D),
    }

    for name, G in graphs.items():
        with Pool(WORKERS, initializer=init_worker, initargs=(G,)) as pool:
            times = pool.map(run_one, X)

        if any(t is None for t in times):
            raise RuntimeError((name, D))

        times = np.array(times)
        key = f"{name}_D{D}"
        out[key] = {
            "D": D,
            "degree": 4,
            "edges": G.number_of_edges(),
            "diameter": nx.diameter(G),
            "apl": nx.average_shortest_path_length(G),
            "median": float(np.median(times)),
            "median_cycles": float(np.median(times) / T_CYCLE),
            "p95": float(np.percentile(times, 95)),
            "max": float(times.max()),
        }

        print(key, out[key]["diameter"], out[key]["median_cycles"])

with open("degree4_pair_results.json", "w") as f:
    json.dump(out, f, indent=2)
