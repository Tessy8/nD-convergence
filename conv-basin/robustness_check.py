"""
Targeted robustness checks for the D=3 hybrid basin computations.

This script does two things:
1. Compares the event-accurate classifier across several ``dt`` and
   ``conv_time`` choices on representative points.
2. Reports wrapped-entry / inner-basin diagnostics for those same points.

The goal is not to remap the full basin, but to test whether specific
classification decisions are stable under reasonable numerical choices.
"""

import csv
import json
import os
import numpy as np

from hybrid_tools import PI, integrate_pc

GUARD_OFFSET = 0.0
DELTA = 0.5
T_MAX = 120.0
T_TOL = 1e-5
CONV_TOL = 0.05

DT_VALUES = [0.05, 0.02, 0.01]
CONV_TIME_VALUES = [6.5, 10.0, 15.0]

# A small mix of points drawn from the currently interesting regions:
# near the diagonal, wrapped-entry examples, and a few ambiguous-looking slice points.
POINTS = [
    (1 / 3, 1 / 3, 1 / 3),
    (-2.991592653589793, 2.991592653589793, 2.991592653589793),
    (-2.991592653589793, -2.991592653589793, 0.8158889055244893),
    (-1.359814842540815, -1.359814842540815, 0.27196296850816326),
    (-1.359814842540815, -0.8158889055244889, 0.27196296850816326),
    (-0.8158889055244889, -1.359814842540815, 0.27196296850816326),
    (-0.8158889055244889, -0.8158889055244889, 0.27196296850816326),
    (-1.359814842540815, -0.2719629685081628, 0.27196296850816326),
    (-0.2719629685081628, -1.359814842540815, 0.27196296850816326),
]


def point_label(x):
    return f"({x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f})"


def main():
    os.makedirs("conv-basin/output", exist_ok=True)

    rows = []
    grouped = {}

    for x0 in POINTS:
        label = point_label(x0)
        grouped[label] = []
        print(f"\nPoint {label}")
        for dt in DT_VALUES:
            for conv_time in CONV_TIME_VALUES:
                res = integrate_pc(
                    x0,
                    guard_offset=GUARD_OFFSET,
                    delta=DELTA,
                    t_max=T_MAX,
                    dt=dt,
                    t_tol=T_TOL,
                    conv_tol=CONV_TOL,
                    conv_time=conv_time,
                    store_every=4,
                    keep_trajectory=False,
                )
                row = {
                    "x1": float(x0[0]),
                    "x2": float(x0[1]),
                    "x3": float(x0[2]),
                    "dt": float(dt),
                    "conv_time": float(conv_time),
                    "converged": bool(res["converged"]),
                    "t_final": float(res["t_final"]),
                    "first_simplex_after_wrap": bool(res["first_simplex_after_wrap"]),
                    "ever_enters_simplex": bool(res["enters_simplex"]),
                    "ever_enters_inner": bool(res["enters_contracting"]),
                    "wrapped_entry_to_inner_same_visit": bool(res["wrapped_entry_to_inner_same_visit"]),
                    "wraps_before_first_simplex": int(res["wrap_events_before_first_simplex"]),
                    "wraps_before_first_contracting": int(res["wrap_events_before_first_contracting"]),
                    "wraps_simplex_to_contracting": int(res["wrap_events_between_simplex_and_contracting"]),
                    "first_simplex_idx": int(res["first_simplex_idx"]),
                    "first_contracting_idx": int(res["first_contracting_idx"]),
                }
                rows.append(row)
                grouped[label].append(row)
                print(
                    f"  dt={dt:>4.2f}, conv_time={conv_time:>4.1f} -> "
                    f"converged={row['converged']}, "
                    f"ever_inner={row['ever_enters_inner']}, "
                    f"same_visit={row['wrapped_entry_to_inner_same_visit']}"
                )

    stable_convergence = {}
    stable_wrapped_entry = {}
    stable_same_visit = {}
    for label, entries in grouped.items():
        vals = {entry["converged"] for entry in entries}
        wvals = {entry["first_simplex_after_wrap"] for entry in entries}
        svals = {entry["wrapped_entry_to_inner_same_visit"] for entry in entries}
        stable_convergence[label] = (len(vals) == 1)
        stable_wrapped_entry[label] = (len(wvals) == 1)
        stable_same_visit[label] = (len(svals) == 1)

    summary = {
        "parameters": {
            "guard_offset": GUARD_OFFSET,
            "delta": DELTA,
            "t_max": T_MAX,
            "t_tol": T_TOL,
            "conv_tol": CONV_TOL,
            "dt_values": DT_VALUES,
            "conv_time_values": CONV_TIME_VALUES,
        },
        "points": [list(p) for p in POINTS],
        "stable_convergence_by_point": stable_convergence,
        "stable_wrapped_entry_by_point": stable_wrapped_entry,
        "stable_same_visit_by_point": stable_same_visit,
    }

    with open("conv-basin/output/robustness_check.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open("conv-basin/output/robustness_check.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved conv-basin/output/robustness_check.csv")
    print("Saved conv-basin/output/robustness_check.json")


if __name__ == "__main__":
    main()
