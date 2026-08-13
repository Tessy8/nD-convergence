import json
from multiprocessing import Pool

import numpy as np

from event_solver import EventSim, spider

Q = [4, 8, 16]
L_VALUES = [1, 2, 4, 8]
N = 300
SEED = 4042
W = 12 * 2 * np.pi / 32
T_CYCLE = 7.225
WORKERS = 3

SIM = None


def init_worker(q, L):
    global SIM
    SIM = EventSim(spider(q, L))


def run_one(x0):
    return SIM.run(x0)["t_sync"]


out = {}

for q in Q:
    for L in L_VALUES:
        D = q * L + 1
        X = np.random.default_rng(SEED).uniform(0, W, size=(N, D))

        with Pool(WORKERS, initializer=init_worker, initargs=(q, L)) as pool:
            times = pool.map(run_one, X)

        if any(t is None for t in times):
            raise RuntimeError((q, L))

        times = np.array(times)
        key = f"q{q}_L{L}"
        out[key] = {
            "q": q,
            "L": L,
            "D": D,
            "diameter": 2 * L,
            "median": float(np.median(times)),
            "median_cycles": float(np.median(times) / T_CYCLE),
            "p95": float(np.percentile(times, 95)),
            "max": float(times.max()),
        }

        print(key, out[key]["median_cycles"])

with open("spider_results.json", "w") as f:
    json.dump(out, f, indent=2)
