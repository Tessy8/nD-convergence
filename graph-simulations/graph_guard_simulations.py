#!/usr/bin/env python3
"""Wrapped graph-guard simulations. Requires numpy and mpmath.
Run: python graph_guard_simulations.py
Default: 252 random starts and one designed start per graph; three cycles.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
import time
from collections import deque
from pathlib import Path
import numpy as np
from mpmath import mp


def number(text):
    s = str(text).replace(' ', '').replace('*', '')
    if 'pi' not in s:
        return mp.mpf(s)
    m = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?)?)pi([+-]\d+(?:\.\d*)?)?', s)
    if not m:
        raise ValueError('Use a number or an expression such as pi, 3*pi, 2*pi-1')
    k = {'': '1', '+': '1', '-': '-1'}.get(m[1], m[1])
    return mp.mpf(k) * mp.pi + mp.mpf(m[2] or '0')


def graph(name, n, edges):
    edges = sorted({tuple(sorted(e)) for e in edges})
    nb = [[] for _ in range(n)]
    for i, j in edges:
        if not 0 <= i < j < n:
            raise ValueError('Edges must join different vertices in 0,...,D-1')
        nb[i].append(j)
        nb[j].append(i)
    if n < 2 or any(not ns for ns in nb):
        raise ValueError('Use a connected graph with at least two vertices')
    distances = []
    for root in range(n):
        ds = [-1] * n
        ds[root] = 0
        queue = deque([root])
        while queue:
            i = queue.popleft()
            for j in nb[i]:
                if ds[j] < 0:
                    ds[j] = ds[i] + 1
                    queue.append(j)
        if min(ds) < 0:
            raise ValueError('This test suite requires a connected graph')
        distances.append(ds)
    color = [d % 2 for d in distances[0]]
    return dict(name=name, n=n, edges=edges, neighbors=nb, distances=distances,
                diameter=max(map(max, distances)), tree=len(edges) == n-1,
                bipartite=all(color[i] != color[j] for i, j in edges))


def named_graph(name):
    m = re.fullmatch(r'([PCK])(\d+)', name)
    if m:
        kind, n = m[1], int(m[2])
        edges = [(i, i+1) for i in range(n-1)]
        if kind == 'C':
            if n < 3:
                raise ValueError('A cycle needs at least three vertices')
            edges.append((n-1, 0))
        if kind == 'K':
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        return graph(name, n, edges)
    m = re.fullmatch(r'K(\d+)_(\d+)', name)
    if m:
        b, l = map(int, m.groups())
        return graph(name, b+l, [(i, b+j) for i in range(b) for j in range(l)])
    m = re.fullmatch(r'(spider|grid|tree)(\d+)x(\d+)', name)
    if m:
        kind, b, l = m[1], int(m[2]), int(m[3])
        if min(b, l) < 1:
            raise ValueError('Graph dimensions must be positive')
        if kind == 'spider':
            edges = []
            for leg in range(b):
                chain = [0] + [1+leg*l+j for j in range(l)]
                edges += list(zip(chain, chain[1:]))
            return graph(name, 1+b*l, edges)
        if kind == 'grid':
            edges = [(r*l+j, r*l+j+1) for r in range(b) for j in range(l-1)]
            edges += [(r*l+j, (r+1)*l+j) for r in range(b-1) for j in range(l)]
            return graph(name, b*l, edges)
        n = sum(b**level for level in range(l+1))
        return graph(name, n, [((i-1)//b, i) for i in range(1, n)])
    extras = {
        'double_star': (8, [(0,1)] + [(0,i) for i in (2,3,4)] + [(1,i) for i in (5,6,7)]),
        'caterpillar': (10, [(i,i+1) for i in range(5)] + [(i,i+5) for i in range(1,5)]),
        'double_broom': (11, [(i,i+1) for i in range(4)] + [(0,i) for i in (5,6,7)] + [(4,i) for i in (8,9,10)]),
        'lobster': (15, [(i,i+1) for i in range(4)] + [(i,5+2*i) for i in range(5)] + [(5+2*i,6+2*i) for i in range(5)]),
    }
    if name in extras:
        return graph(name, *extras[name])
    raise ValueError('Unknown graph: ' + name)


def initial_directions(g, rng, samples, kind):
    n, nb = g['n'], g['neighbors']
    result = []
    if kind in ('both', 'constructed'):
        v = np.zeros(n)
        if g['tree']:
            r = next(i for i, ds in enumerate(g['distances']) if max(ds) == g['diameter'])
            v[r] = 1
            if n > 2:
                v[nb[r][0]] = 1
        elif all(len(ns) == 2 for ns in nb):
            v[:2 if n % 2 else 3] = 1
        elif g['bipartite']:
            r = next(i for i, ds in enumerate(g['distances']) if max(ds) == g['diameter'])
            v[[r] + nb[r]] = 1
        else:
            v[:-1] = 1
        result.append(('constructed', v))
    if kind in ('both', 'random'):
        for k in range(samples):
            v = rng.normal(size=n)
            v = (v-v.min()) / np.ptp(v)
            result.append((f'random_{k+1}', v))
    return result


def full_returns(direction, g, q, c, h, amplitude, cycles, matrices=False):
    """Advance the original velocities and guards, not the predicted return map."""
    n, nb, length = g['n'], g['neighbors'], 2*mp.pi
    v = [mp.mpf(str(z)) for z in direction]
    mean = mp.fsum(v)/n
    v = [z-mean for z in v]
    s0 = c/5
    spread0 = amplitude*(max(v)-min(v))
    if not 0 < q < 1 or not 0 < c < length or not 0 < h <= 2*length:
        raise ValueError('Use 0<q<1, 0<c<2*pi, 0<h<=4*pi')
    if not 0 < spread0 < min(s0, c-s0, length-c)/4:
        raise ValueError('Reduce amplitude; the initial phases must be close and below c')
    if h == length and c < mp.pi and spread0 >= min(s0, c-s0, mp.pi-c)/4:
        raise ValueError('Initial spread exceeds the theorem neighbourhood')
    x = [s0+amplitude*z for z in v]
    y = x.copy()
    target, clock = s0+length, mp.mpf(0)
    tol = mp.power(10, -mp.dps+12)
    phi, valid, event_count, out = mp.eye(n), True, 0, []

    def mode(xx):
        return [q if xx[i] >= c-tol and all(xx[i]+xx[j] < h-tol for j in nb[i])
                else mp.mpf(1) for i in range(n)]

    for _ in range(max(10000, cycles*1000*n)):
        speed = mode(x)
        events = [((target-mp.fsum(y)/n)/(mp.fsum(speed)/n), 'return', ())]
        for i in range(n):
            if x[i] < c-tol:
                events.append(((c-x[i])/speed[i], 'threshold', (i,)))
            events.append(((length-x[i])/speed[i], 'wrap', (i,)))
        for i, j in g['edges']:
            if x[i]+x[j] < h-tol:
                events.append(((h-x[i]-x[j])/(speed[i]+speed[j]), 'pair', (i,j)))
        dt = min(t for t, _, _ in events)
        if dt < -tol:
            raise RuntimeError('Negative event time')
        hits = [e for e in events if abs(e[0]-dt) <= tol]
        for i in range(n):
            x[i] += speed[i]*dt
            y[i] += speed[i]*dt
        for _, kind, ids in hits:
            if kind == 'wrap':
                x[ids[0]] = mp.mpf(0)
            elif kind == 'threshold':
                x[ids[0]] = c
        after = mode(x)
        actual = [e for e in hits if e[1] != 'return']
        changed = after != speed
        event_count += bool(actual)
        if changed and len(actual) != 1:
            valid = False
        if matrices and changed and len(actual) == 1:
            normal = mp.zeros(n, 1)
            for i in actual[0][2]:
                normal[i] = 1
            before, following = mp.matrix(speed), mp.matrix(after)
            xi = mp.eye(n)+(following-before)*normal.T/(normal.T*before)[0]
            phi = xi*phi
        clock += dt
        if any(kind == 'return' for _, kind, _ in hits):
            mean = mp.fsum(y)/n
            offsets = [(z-mean)/amplitude for z in y]
            if max(offsets)-min(offsets) < 1000*tol/amplitude:
                raise RuntimeError('Precision exhausted; increase --digits or reduce --cycles')
            rec = dict(offsets=offsets, time=clock, events=event_count,
                       E=mp.sqrt(mp.fsum(z*z for z in offsets)),
                       S=max(offsets)-min(offsets), J=None)
            if matrices and valid:
                pi = mp.eye(n)-mp.ones(n,n)/n
                rec['J'] = np.asarray((pi*phi*pi).tolist(), dtype=float)
            out.append(rec)
            if len(out) == cycles:
                return out
            target += length
            phi, valid, event_count = mp.eye(n), True, 0
    raise RuntimeError('Event limit reached')


def local_map(v, g, q):
    """Separate analytical calculation, used only to check the integration."""
    v = np.asarray(v, dtype=float)
    a, n, nb = (1-q)/(1+q), g['n'], g['neighbors']
    p, scale = v.copy(), max(np.max(np.abs(v)), 1e-100)
    for _ in range(10000):
        new = (1-a)*v+a*np.array([max(p[j] for j in ns) for ns in nb])
        if np.max(abs(new-p)) <= 4e-15*(1-a)*scale:
            p = new
            break
        p = new
    else:
        raise RuntimeError('Local-map fixed point failed to converge')
    high = [max(ns, key=lambda j:p[j]) for ns in nb]
    low = [min(ns, key=lambda j:p[j]) for ns in nb]
    selection = np.zeros((n,n))
    selection[np.arange(n), high] = 1
    H = (1-a)*np.linalg.solve(np.eye(n)-a*selection, np.eye(n))
    p, W = H@v, np.eye(n)
    for i, (j,k) in enumerate(zip(high, low)):
        if 2*p[k]-p[j]-p[i] > 0:
            W[i] = 0
            W[i,i] += 1+a
            W[i,j] += a
            W[i,k] -= 2*a
    pi = np.eye(n)-np.ones((n,n))/n
    return pi@W@p, pi@W@H@pi


def predicted_factor(g, q, c, h):
    if h != 2*mp.pi:
        return None
    a = (1-q)/(1+q)
    if c < mp.pi:
        return 1-a**(g['diameter']+(0 if g['bipartite'] else 1))
    if g['n'] == 2:
        return 4*q/(1+q)**2 if c == mp.pi else 2/(1+q)
    return None


def save_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(args):
    mp.dps = args.digits
    qlist, c, h = [number(z) for z in args.q], number(args.c), number(args.h)
    amplitude = number(args.amplitude)
    names = args.graphs
    if args.suite == 'all':
        names = ['P2','P3','P4','P5','P10','C3','C4','C5','C6','C8','C10',
                 'spider3x2','spider4x3','double_star','tree2x3','tree3x2',
                 'caterpillar','double_broom','lobster','K3','K5','K3_3','K2_4','grid3x3','grid3x4']
    gs = [named_graph(name) for name in names]
    if args.edges:
        edges = [tuple(map(int, e.split('-'))) for e in args.edges.split(',')]
        gs = [graph('custom', 1+max(max(e) for e in edges), edges)]
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allrows, summaries, starts = [], [], []
    began = time.perf_counter()
    checks = dict(returns=0, event_batches=0, bound_violations=0,
                  matrix_returns=0, max_local_map_error=0., max_J_error=0.)
    for g in gs:
        directions = initial_directions(g, rng, args.samples, args.kind)
        if args.direction:
            v = np.array([float(z) for z in args.direction.split(',')])
            if len(v) != g['n']:
                raise ValueError('--direction must have one entry per vertex')
            directions = [('custom', v)]
        for q in qlist:
            theta = predicted_factor(g, q, c, h)
            local_regime = h == 2*mp.pi and c < mp.pi
            collected = []
            for kind, direction in directions:
                start = [mp.mpf(str(z)) for z in direction]
                mean = mp.fsum(start)/g['n']
                start = [z-mean for z in start]
                E0, S0 = mp.sqrt(mp.fsum(z*z for z in start)), max(start)-min(start)
                previous = start
                previous_S = S0
                starts.append(dict(graph=g['name'], q=str(q), kind=kind,
                                   centered_direction=[mp.nstr(z,25) for z in start]))
                trajectory = full_returns(direction,g,q,c,h,amplitude,args.cycles,args.matrices)
                for n, rec in enumerate(trajectory, 1):
                    sr, er = rec['S']/previous_S, rec['E']/E0
                    sb = theta**n if theta is not None else None
                    eb = mp.sqrt(mp.mpf(g['n'])/2)*sb if sb is not None else None
                    err, jerr = None, None
                    if local_regime:
                        pred, jac = local_map([float(z) for z in previous],g,float(q))
                        err = float(np.max(abs(pred-np.array([float(z) for z in rec['offsets']]))))
                        checks['max_local_map_error'] = max(checks['max_local_map_error'],err)
                        if err > 1e-10:
                            raise AssertionError('Full integration and local map disagree')
                        if rec['J'] is not None:
                            jerr = float(np.max(abs(rec['J']-jac)))
                            checks['matrix_returns'] += 1
                            checks['max_J_error'] = max(checks['max_J_error'],jerr)
                            if jerr > 1e-10:
                                raise AssertionError('Saltation product and branch Jacobian disagree')
                    passed = theta is None or (sr <= theta+mp.mpf('1e-11') and er <= eb+mp.mpf('1e-11'))
                    checks['bound_violations'] += not passed
                    checks['returns'] += 1
                    checks['event_batches'] += rec['events']
                    fmt = lambda z: '' if z is None else mp.nstr(z,25)
                    row = dict(graph=g['name'],D=g['n'],diameter=g['diameter'],q=fmt(q),c=args.c,h=args.h,
                               kind=kind,cycle=n,theta=fmt(theta),S_step=fmt(sr),S_over_S0=fmt(rec['S']/S0),
                               E_over_E0=fmt(er),S_bound=fmt(sb),E_bound=fmt(eb),
                               S_step_minus_theta=fmt(sr-theta if theta is not None else None),
                               local_map_error=err if err is not None else '',J_error=jerr if jerr is not None else '',
                               bound_pass=passed if theta is not None else 'not_applicable',
                               time=fmt(rec['time']),events=rec['events'],
                               offsets=json.dumps([fmt(z) for z in rec['offsets']]))
                    allrows.append(row)
                    collected.append(row)
                    previous, previous_S = rec['offsets'], rec['S']
            first = [r for r in collected if r['cycle'] == 1]
            special = next((r for r in first if r['kind'] in ('constructed','custom')), None)
            randoms = [float(r['S_step']) for r in first if r['kind'].startswith('random')]
            third = next((r for r in collected if r['cycle'] == 3 and r['kind'] in ('constructed','custom')), None)
            report = dict(graph=g['name'],q=str(q),D=g['n'],diameter=g['diameter'],theta='' if theta is None else float(theta),
                          constructed_S1_S0='' if special is None else special['S_step'],
                          constructed_E1_E0='' if special is None else special['E_over_E0'],
                          constructed_E3_E0='' if third is None else third['E_over_E0'],
                          E3_bound='' if third is None else third['E_bound'],
                          random_count=len(randoms),
                          random_S1_S0_min=min(randoms) if randoms else '',
                          random_S1_S0_median=float(np.median(randoms)) if randoms else '',
                          random_S1_S0_max=max(randoms) if randoms else '')
            for cycle_number in sorted(set((1, args.cycles))):
                random_rows = [r for r in collected if r['cycle'] == cycle_number and r['kind'].startswith('random')]
                for metric in ('E_over_E0', 'S_over_S0'):
                    values = [float(r[metric]) for r in random_rows]
                    for label, func in (('min', min), ('median', np.median), ('max', max)):
                        report[f'random_{metric}_cycle{cycle_number}_{label}'] = float(func(values)) if values else ''
                report[f'E_bound_cycle{cycle_number}'] = '' if theta is None else float(mp.sqrt(mp.mpf(g['n'])/2)*theta**cycle_number)
            summaries.append(report)
            print(f"{g['name']:14} q={float(q):.2f} theta={report['theta']}  "
                  f"constructed S1/S0={str(report['constructed_S1_S0'])[:13]}  "
                  f"random range={report['random_S1_S0_min']}..{report['random_S1_S0_max']}",flush=True)
    checks.update(seconds=time.perf_counter()-began,graphs=len(gs),seed=args.seed,
                  digits=args.digits,cycles=args.cycles,samples=args.samples,
                  random_initial_conditions=sum(z['kind'].startswith('random') for z in starts),
                  designed_initial_conditions=sum(z['kind']=='constructed' for z in starts),
                  matrix_checks_requested=args.matrices,
                  numpy_version=np.__version__,q_values=args.q,c=args.c,h=args.h)
    save_csv(out/'trajectories.csv', allrows)
    save_csv(out/'summary.csv', summaries)
    (out/'initial_conditions.json').write_text(json.dumps(starts,indent=2))
    (out/'graphs.json').write_text(json.dumps(gs,indent=2))
    (out/'run_report.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))
    if checks['bound_violations']:
        raise AssertionError('A tested bound was violated; inspect trajectories.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphs',nargs='+',default=['C4','C5','C6','C8','K3_3','grid3x3','P5','P10','spider3x2'])
    parser.add_argument('--suite',choices=['selected','all'],default='selected')
    parser.add_argument('--q',nargs='+',default=['0.5'])
    parser.add_argument('--c',default='1')
    parser.add_argument('--h',default='2*pi')
    parser.add_argument('--amplitude',default='1e-7')
    parser.add_argument('--cycles',type=int,default=3)
    parser.add_argument('--samples',type=int,default=252)
    parser.add_argument('--seed',type=int,default=20261010)
    parser.add_argument('--digits',type=int,default=65)
    parser.add_argument('--kind',choices=['both','constructed','random'],default='both')
    parser.add_argument('--matrices',action='store_true')
    parser.add_argument('--direction',help='Comma-separated custom offsets, e.g. 1,1,1,0,0,0')
    parser.add_argument('--edges',help='Custom zero-based edges, e.g. 0-1,1-2,2-0')
    parser.add_argument('--out',default='simulation_results')
    args = parser.parse_args()
    if args.cycles < 1 or args.samples < 0 or args.digits < 35:
        parser.error('Use cycles>=1, samples>=0, digits>=35')
    run(args)


if __name__ == '__main__':
    main()
