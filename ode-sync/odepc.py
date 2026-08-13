from numpy import allclose, arange, asarray
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class OdePC( object ):
  """Integrator for piecewise constant systems
  
     This integrator should really only be used with piecewise constant systems.
     It uses a line search algorithm to find the time-step that brings it to within
     tTol of the transition time between constant valued domain.
     
     However, the notion of "constant" is flexible, and be default tested with
     allclose(). Since this means values can actually change, uses forward Euler.
     
     Can be used as forward Euler integrator by setting dt and tTol equal.
     
     In addition, this integrator will respect any requested time-steps, but will
     add additional time points as needed.
  """
  def __init__(self, fun):
    self._fun = fun

  def __call__( self, y0, t0, t1=None, dt=None, tTol=1e-6, pars=None, allclose=allclose, withDy=False ):
    """Integrate a piecewise constant vector field"""
    if t1 is None:
      ts = list(t0)
      assert len(ts)>1 and all(np.diff(ts)>0)
    else:
      assert dt is not None
      ts = list(arange(t0,t1,dt))
    # Output trajectories
    y = [asarray(y0, dtype=float)]
    # Initial time
    t = [ts.pop(0)]
    if dt is None:
      dt = ts[-1]
    # Initial time-step
    h = min(dt,ts[0]-t0)
    # Initial vector
    dy0 = [self._fun( t[0], y[0], pars )]
    # loop until hit end time
    while ts:
      # Estimate next time step
      th = t[-1]+h
      yh = y[-1]+h*dy0[-1]
      dy = self._fun( th, yh, pars )
      ### print t[-1],"try ",th,yh,"-->",dy,"h",h
      # If vector changed too much and timestep can be shrunk 
      if not allclose(dy,dy0[-1]) and h>tTol:
        # --> decrease step and try again
        h = h/2.0
        #!!!assert not len(raw_input("hit <enter>"))
        continue
      # If overshot required time-point --> compute at that point
      if th<ts[0]:
        h0 = h
      else:
        th = ts.pop(0)
        h0 = th-t[-1]
        yh = y[-1]+h0*dy0[-1]
        dy1 = self._fun( th, yh, pars )
        # If the following allclose() fails, it means that even though dy didn't
        #   change before, it did when we took a smaller step --> take minimal step
        if h0>tTol and not allclose(dy1,dy):
          h0 = tTol
          th = t[-1]+h0
          yh = y[-1]+h0*dy0[-1]
          dy1 = self._fun( th, yh, pars )            
        dy = dy1
      # Compute forward Euler step 
      dy0.append(dy)
      y.append(y[-1]+dy0[-1]*h0)
      assert th == t[-1]+h0
      t.append(th)
      h = min(h*1.5, dt)
      continue
    if withDy:
      return asarray(t, dtype=float), asarray(y, dtype=float), asarray(dy0, dtype=float)
    else:
      return asarray(t, dtype=float), asarray(y, dtype=float)

def _test_odePC():
  def fun( t, x, pars ):
    return asarray(x, dtype=float).round()
  ode = OdePC(fun)
  t,y,dy = ode(asarray([1,1.4,1.9], dtype=float), 0, 3, dt=0.9, tTol = 0.001, withDy=True)
  plt.figure(); plt.clf()
  plt.semilogy( t[1:],fun(None,y[1:,:],None)-np.diff(y.T).T/np.diff(t)[:,np.newaxis],'.')
  plt.title('Secant error')
  plt.xlabel('time')
  plt.ylabel('error')
  plt.savefig("Secant error")
  plt.close()

def _test_odeEU():
  """Test using as forward Euler integrator"""
  def f(t,x,p):
    """Hopf oscillator"""
    om,gam = p
    v = np.empty_like(x)
    v[...,0] = -x[...,1]*om
    v[...,1] = x[...,0]*om
    r = (x[...,1]*x[...,1]+x[...,0]*x[...,0])[...,np.newaxis]
    v += (1-r)*x*gam
    return v
  o = OdePC(f)
  t = np.linspace(-2,2,14)
  ic = asarray(np.meshgrid(t,t)).T
  dt = 0.08
  t,y = o(ic,t0=0,t1=3,dt=dt,tTol=dt,pars=(1,0.5  ))
  y2 = y.reshape(y.shape[0],y.shape[1]*y.shape[2],y.shape[3])
  th = np.linspace(-np.pi,np.pi,256)
  plt.figure(); plt.clf()
  plt.title('Hopf oscillator, integrated with dt='+str(dt))
  plt.plot( np.sin(th),np.cos(th),'k-',lw=9)
  plt.plot( np.sin(th),np.cos(th),'w-',lw=3)
  plt.plot( y2[...,0], y2[...,1],'.-',alpha=0.5)
  plt.grid(1)
  plt.axis('equal')
  plt.savefig("Hopf oscillator.png")

def _test_odePC_animated():
    def fun(t, x, pars):
        return asarray(x, dtype=float).round()

    ode = OdePC(fun)
    t, y, dy = ode(asarray([1, 1.4, 1.9], dtype=float), 0, 3, dt=0.9, tTol=0.001, withDy=True)

    fig, ax = plt.subplots()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, np.max(y) + 1)
    ax.set_title("PCS Animation")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    
    # One line per component of y
    lines = [ax.plot([], [], marker='o', label=f'y[{i}]')[0] for i in range(y.shape[1])]
    ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(t[:frame+1], y[:frame+1, i])
        return lines

    anim = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=10)

    # Save animation as .mp4
    anim.save("odepc_animation.mp4", writer="ffmpeg")

    plt.close()

def _polygonal_vector_single():
   # Polygonal vector field fn
  def polygonal_field(t, x, pars=None):
      x0 = x[0]
      if x0 < 1:
          return np.array([1, 1])
      elif x0 < 2:
          return np.array([1, 2])
      elif x0 < 4:
          return np.array([2, 1])
      else:
          return np.array([0, 0])  # stop

  ode = OdePC(polygonal_field)

  # Starting point
  y0 = np.array([0.0, 0.0])
  t, y = ode(y0, t0=0, t1=5, dt=0.1, tTol=1e-3)

  # Plot
  plt.figure(figsize=(6,6))
  plt.plot(y[:,0], y[:,1], 'bo-', label="Trajectory")
  plt.quiver(y[:,0], y[:,1], [polygonal_field(None, pt)[0] for pt in y], [polygonal_field(None, pt)[1] for pt in y], 
            color='gray', alpha=0.5, label="Vector Field")
  plt.title("PCV Field (Polygonal Path)")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.axis('equal')
  plt.grid(True)
  plt.legend()  
  plt.savefig("PCV Single")
  plt.close()

  # Create animation
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.set_xlim(0, 5)
  ax.set_ylim(0, 5)
  ax.set_title("PCV Animation")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.grid(True)
  ax.set_aspect('equal')
  point, = ax.plot([], [], 'ro')
  trail, = ax.plot([], [], 'b--', lw=1)

  def init():
      point.set_data([], [])
      trail.set_data([], [])
      return point, trail

  def update(frame):
      point.set_data([y[frame][0]], [y[frame][1]])
      trail.set_data(y[:frame + 1, 0], y[:frame + 1, 1])
      return point, trail

  anim = FuncAnimation(fig, update, frames=len(y), init_func=init, interval=200, blit=True)
  anim.save("PCV_Single_Trajectory_Animation.mp4", writer="ffmpeg")

def _polygonal_vector_multiple():
    def polygonal_field(t, x, pars=None):
        x0 = x[0]
        if x0 < 1:
            return np.array([1, 1])
        elif x0 < 2:
            return np.array([1, 2])
        elif x0 < 4:
            return np.array([2, 1])
        else:
            return np.array([0, 0])  # stop

    ode = OdePC(polygonal_field)

    initial_positions = [
        np.array([0.0, 0.0]),
        np.array([0.2, 0.5]),
        np.array([0.5, 0.1]),
        np.array([0.1, 0.9])
    ]

    num_trajectories = []
    t_vals = None

    for y0 in initial_positions:
        t, y = ode(y0, t0=0, t1=5, dt=0.1, tTol=1e-3)
        num_trajectories.append(y)
        if t_vals is None:
            t_vals = t

    # Image
    plt.figure(figsize=(6, 6))
    for i, y in enumerate(num_trajectories):
        plt.plot(y[:, 0], y[:, 1], label=f"Trajectory {i+1}")
    plt.title("Multi-PCV (Polygonal Path)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig("multi_pcv.png")
    plt.close()

    # Animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_title("Multi-PCV Animation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect('equal')

    points = [ax.plot([], [], 'o')[0] for _ in initial_positions]
    trails = [ax.plot([], [], '--', lw=1)[0] for _ in initial_positions]

    def init():
        for pt, tr in zip(points, trails):
            pt.set_data([], [])
            tr.set_data([], [])
        return points + trails

    def update(frame):
        for i, y in enumerate(num_trajectories):
            if frame < len(y):
                points[i].set_data([y[frame][0]], [y[frame][1]])
                trails[i].set_data(y[:frame+1, 0], y[:frame+1, 1])
        return points + trails

    anim = FuncAnimation(fig, update, frames=len(t_vals), init_func=init, interval=200, blit=True)
    anim.save("multi_pcv_animation.mp4", writer="ffmpeg")

def _mod_polygonal_vector_multiple():
    def polygonal_field(t, x, pars=None):
        x0, y0 = x[0], x[1]
        if x0 < 1 and y0 < 1:
            return np.array([1, 1])
        elif x0 >= 1 and y0 < 1:
            return np.array([1, 2])
        elif x0 < 1 and y0 >= 1:
            return np.array([2, 1])
        elif 1 <= x0 < 4 and 1 <= y0 < 4:
           return np.array([1, 1])
        else:
            return np.array([0, 0])  # stop

    ode = OdePC(polygonal_field)

    initial_positions = [
        np.array([0.0, 0.0]),
        np.array([2, 0.1]),
        np.array([0.1, 2]),
        np.array([1.1, 0.1])
    ]

    num_trajectories = []
    t_vals = None

    for y0 in initial_positions:
        t, y = ode(y0, t0=0, t1=5, dt=0.1, tTol=1e-3)
        num_trajectories.append(y)
        if t_vals is None:
            t_vals = t

    # Image
    plt.figure(figsize=(6, 6))
    for i, y in enumerate(num_trajectories):
        plt.plot(y[:, 0], y[:, 1], label=f"Trajectory {i+1}")
    plt.title("Mod-Multi-PCV (Polygonal Path)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig("mod_multi_pcv.png")
    plt.close()

    # Animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_title("Mod-Multi-PCV Animation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect('equal')

    points = [ax.plot([], [], 'o')[0] for _ in initial_positions]
    trails = [ax.plot([], [], '--', lw=1)[0] for _ in initial_positions]

    def init():
        for pt, tr in zip(points, trails):
            pt.set_data([], [])
            tr.set_data([], [])
        return points + trails

    def update(frame):
        for i, y in enumerate(num_trajectories):
            if frame < len(y):
                points[i].set_data([y[frame][0]], [y[frame][1]])
                trails[i].set_data(y[:frame+1, 0], y[:frame+1, 1])
        return points + trails

    anim = FuncAnimation(fig, update, frames=len(t_vals), init_func=init, interval=200, blit=True)
    anim.save("mod_multi_pcv_animation.mp4", writer="ffmpeg")

def multiple_cycles_polygonal_vector():
    def polygonal_field(t, x, pars=None):
        x0, y0 = x[0], x[1]
        if x0 < 1 and y0 < 1:
            return np.array([1, 1])
        elif x0 >= 1 and y0 < 1:
            return np.array([1, 2])
        elif x0 < 1 and y0 >= 1:
            return np.array([2, 1])
        elif 1 <= x0 < 3 and 1 <= y0 < 3:
            return np.array([1, 1])
        else:
            return np.array([0, 0])

    def project_back(last_pos, last_dir, fixed_y=0):
        if np.allclose(last_dir, 0):
            return last_pos
        slope = last_dir[1] / last_dir[0] if last_dir[0] != 0 else np.inf
        if slope == 0:
            return np.array([last_pos[0], fixed_y])
        x_new = last_pos[0] - (last_pos[1] - fixed_y) / slope
        return np.array([x_new, fixed_y])

    def simulate_convergence():
        ode = OdePC(polygonal_field)
        initial_positions = [
            np.array([0.0, 0.0]),
            np.array([2.0, 0.1]),
            np.array([0.1, 2.0]),
            np.array([1.1, 0.1])
        ]

        all_trajectories = [[] for _ in initial_positions]
        curr_positions = initial_positions[:]
        cycles = 10

        for _ in range(cycles):
            new_positions = []
            cycle_trajs = []
            max_steps = 0

            for i, y0 in enumerate(curr_positions):
                t, y, dy = ode(y0, t0=0, t1=2, dt=0.1, tTol=1e-3, withDy=True)
                cycle_trajs.append((y, dy))
                max_steps = max(max_steps, len(y))

            for i, (y, dy) in enumerate(cycle_trajs):
                pad_len = max_steps - len(y)
                if pad_len > 0:
                    y = np.vstack([y, np.tile(y[-1], (pad_len, 1))])
                all_trajectories[i].append(y)

            for y, dy in cycle_trajs:
                last_dy = dy[-1]
                if np.allclose(last_dy, 0):
                    for past_dy in reversed(dy):
                        if not np.allclose(past_dy, 0):
                            last_dy = past_dy
                            break
                new_start = project_back(y[-1], last_dy)
                new_positions.append(new_start)

            curr_positions = new_positions

        return all_trajectories

    trajectories = simulate_convergence()
    num = len(trajectories)
    n_cycles = len(trajectories[0])
    max_steps_per_cycle = [max(len(trajectories[i][c]) for i in range(num)) for c in range(n_cycles)]

    frames = []
    for c in range(n_cycles):
        for step in range(max_steps_per_cycle[c]):
            frames.append((c, step))

    fig, ax = plt.subplots(figsize=(5, 5))

    def update(frame):
        cycle, step = frame
        ax.clear()
        ax.set_xlim(-1, 1.8)
        ax.set_ylim(0, 1.8)
        ax.set_title(f"Convergence - Cycle {cycle + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.set_aspect('equal')

        for i in range(num):
            for c in range(cycle):
                prev_traj = trajectories[i][c]
                ax.plot(prev_traj[:, 0], prev_traj[:, 1], linestyle=':', alpha=0.4)

            curr_traj = trajectories[i][cycle]
            if step < len(curr_traj):
                ax.plot(curr_traj[:step+1, 0], curr_traj[:step+1, 1], linestyle='-')
                ax.plot(curr_traj[step, 0], curr_traj[step, 1], 'o')
            else:
                ax.plot(curr_traj[:, 0], curr_traj[:, 1], linestyle='-')
                ax.plot(curr_traj[-1, 0], curr_traj[-1, 1], 'o')

        return ax.patches + ax.lines

    anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
    anim.save("synced_convergence.mp4", writer="ffmpeg")

if __name__=="__main__":
  # _test_odePC()
  # _test_odeEU()
  # _test_odePC_animated()
  # _polygonal_vector_single()
  # _polygonal_vector_multiple()
  # _mod_polygonal_vector_multiple()
  multiple_cycles_polygonal_vector()