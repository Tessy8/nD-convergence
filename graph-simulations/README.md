# Graph simulations

This folder is the graph work I did after the earlier ODE simulations.

The autonomous rule is:

`x_k` is slow if `x_k > c` and `x_k + x_j < 2*pi` for every neighbor `j`.

Otherwise it is fast.

I kept this folder small. It has the main steps of the graph work, not every test file I made.

## Files

- `graph_guards.py` - the graph rule and the earlier simulation code.
- `parberry.py` - Parberry comparator pairs used as graph edges.
- `run_graphs.py` - first comparison: complete, Parberry, butterfly, shell, ring and random regular graphs.
- `degree_graphs.py` - the degree-controlled graph families.
- `run_degree_test.py` - compares graphs at the same degree and dimension.
- `event_solver.py` - event-driven solver used for the later graph tests.
- `run_spiders.py` - spider graphs. I varied branch count `q` and branch length `L`.
- `run_degree4_pair.py` - degree-4 bottleneck vs circulant with the same D, degree, edge count and diameter.

## Install

```bash
pip install numpy networkx
```

## Run

Early graph comparison:

```bash
python run_graphs.py
```

Degree-controlled test:

```bash
python run_degree_test.py
```

Spider test:

```bash
python run_spiders.py
```

Matched degree-4 test:

```bash
python run_degree4_pair.py
```

The full runs use a few hundred initial conditions, so the later scripts can take some time.
