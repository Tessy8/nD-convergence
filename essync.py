import numpy as np
from numpy.random import rand
from matplotlib.pylab import figure,pause
from numpy import asarray, meshgrid, linspace, c_, pi

beta = 0.5
alpha = 0.66

class OdePC:
    def __init__(self, fun):
        self._fun = fun

    def __call__(self, y0, t0, t1=None, dt=None, tTol=1e-6,
                 pars=None, allclose=np.allclose, withDy=False):
        if t1 is None:
            ts = list(t0)
            assert len(ts) > 1 and all(np.diff(ts) > 0)
        else:
            assert dt is not None
            ts = list(np.arange(t0, t1, dt))

        y = [np.asarray(y0, dtype=float)]
        t = [ts.pop(0)]

        if dt is None:
            dt = ts[-1]

        h = min(dt, ts[0] - t0)
        dy0 = [self._fun(t[0], y[0], pars)]

        while ts:
            th = t[-1] + h
            yh = y[-1] + h * dy0[-1]
            dy = self._fun(th, yh, pars)

            if not allclose(dy, dy0[-1]) and h > tTol:
                h /= 2.0
                continue

            if th < ts[0]:
                h0 = h
            else:
                th = ts.pop(0)
                h0 = th - t[-1]
                yh = y[-1] + h0 * dy0[-1]
                dy1 = self._fun(th, yh, pars)
                if h0 > tTol and not allclose(dy1, dy):
                    h0 = tTol
                    th = t[-1] + h0
                    yh = y[-1] + h0 * dy0[-1]
                    dy1 = self._fun(th, yh, pars)
                dy = dy1

            dy0.append(dy)
            y.append(y[-1] + dy * h0)
            t.append(th)
            h = min(h * 1.5, dt)

        if withDy:
            return (np.asarray(t, dtype=float),
                    np.asarray(y, dtype=float),
                    np.asarray(dy0, dtype=float))
        else:
            return (np.asarray(t, dtype=float),
                    np.asarray(y, dtype=float))


def angPlotPlan(a, wrap=2*np.pi, jump_thresh=None):
    """
    Create plotable polylines from angular time series on a torus.

    Parameters
    ----------
    a : array_like, shape (T, D)
        Angles (radians). Values can be anywhere; they’ll be wrapped to (-pi, pi].
    wrap : float
        Full wrap period (2*pi for radians).
    jump_thresh : float or None
        If None, set to 0.9*pi. Any step larger than this in ANY dimension
        starts a new segment to avoid drawing across the wrap seam.

    Returns
    -------
    segments, meta : (list of (Ni, D) arrays, dict)
        segments: list of contiguous polyline segments safe to plot
        meta: dict with wrap and threshold (for debugging)
    """
    a = np.asarray(a, dtype=float)
    assert a.ndim == 2, "angPlotPlan expects shape (T, D)"
    T, D = a.shape
    if T < 2:
        return [], {"wrap": wrap, "jump_thresh": jump_thresh}

    if jump_thresh is None:
        jump_thresh = 0.9*np.pi  # opinionated: strict enough to prevent seam jumps

    # Wrap into (-pi, pi]
    aw = ((a + np.pi) % (2*np.pi)) - np.pi

    # Identify big jumps (per-step, in any dimension)
    diffs = np.diff(aw, axis=0)
    # Shortest angular difference per component
    diffs = (diffs + np.pi) % (2*np.pi) - np.pi
    jumps = np.any(np.abs(diffs) > jump_thresh, axis=1)  # shape (T-1,)

    # Split into segments where no big jump occurs
    segments = []
    start = 0
    for i, is_jump in enumerate(jumps, start=0):
        if is_jump:
            seg = aw[start:i+1]
            if seg.shape[0] >= 2:
                segments.append(seg)
            start = i+1
    # tail
    seg = aw[start:]
    if seg.shape[0] >= 2:
        segments.append(seg)

    return segments, {"wrap": wrap, "jump_thresh": jump_thresh}


def fun(t,y,*pars):
  # We are on a torus, so only the fractional part of corrdinates matters
  y = asarray(y)
  D = len(y)
  yf = y - asarray(y,int)
  dy = 1 - beta*(
     (yf>(alpha/D)) | (yf.mean(0)>1/D)
  )
  return dy


if 1: # Visualize the 2D version
  t = linspace(0,1,31,endpoint=False)
  x,y = meshgrid(t,t)
  xy = c_[x.flatten(), y.flatten()]
  vxy = fun(0,xy.T)
  fig = figure(); fig.clf()
  ax = fig.gca()
  ax.quiver(x*(2*pi),y*(2*pi),vxy[0],vxy[1],pivot='mid',color=[0.75,0.75,0.75])
  ax.axis('equal')

if 1: # Add 2D trajectories to 2D quiver
  o = OdePC( fun )
  t,yt = o(rand(2)/2,0,10,dt=0.1)

  if 1:
    a0 = yt * (2*pi) + pi
    apl = angPlotPlan(a0)[0]
    for ai in apl:
      ax.plot( ai[:,1]+pi, ai[:,0]+pi, '-')
    ax.plot( apl[0][0,1]+pi, apl[0][0,0]+pi, 'o' )

if 1: # Show convergence rates at various dimensions
  fig = figure(); ax = fig.gca()
  for k in [2,6,12]:
    for kk in range(15):
      tk,ytk = o(rand(k)/k,0,10,dt=0.1)
      if kk == 0:
        h = ax.semilogy( tk, ytk.std(-1), '-', label=str(k))
      else:
        ax.semilogy( tk, ytk.std(-1), '-', color=h[0].get_color(),lw=0.5)
  ax.legend()
  
pause(200)


