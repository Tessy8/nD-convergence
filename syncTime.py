import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

lower_boundary = 0.3
upper_boundary = 0.7


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
            assert th == t[-1] + h0
            t.append(th)
            h = min(h * 1.5, dt)

        if withDy:
            return (np.asarray(t, dtype=float),
                    np.asarray(y, dtype=float),
                    np.asarray(dy0, dtype=float))
        else:
            return (np.asarray(t, dtype=float),
                    np.asarray(y, dtype=float))


def polygonal_field(t, x, pars=None):
    a, b = x
    x0 = a%1
    y0 = b%1
    speed = 0.6
    if x0 < lower_boundary and y0 < lower_boundary:
        base = np.array([1, 1])
    elif lower_boundary <= x0 <= upper_boundary and y0 < lower_boundary:
        base = np.array([1, 1.5])
    elif x0 < lower_boundary and lower_boundary <= y0 <= upper_boundary:
        base = np.array([1.5, 1])
    elif y0 == x0:
        base = np.array([1, 1])
    elif (y0 < -x0 + 1 + upper_boundary):
        base = np.array([1, 1])
        speed /= 1+a/5
    else:
        base = np.array([1, 1])
    return speed * base


def create_continuous_trajectories():
    ode = OdePC(polygonal_field)
    initials = [
        np.array([0.0, 0.1]),
        np.array([0.2, 0.0]),
        np.array([0.3, 0.0]),
        np.array([0.0, 0.2])
    ]
    n_agents = len(initials)
    
    continuous_trajs = []
    continuous_times = []
    
    for agent_idx in range(n_agents):
        current_pos = initials[agent_idx]
        global_time = 0.0
        
        t_c, y_c, dy_c = ode(current_pos, t0=global_time, t1=global_time+15.0, dt=0.001, tTol=1e-6, withDy=True)
        
        continuous_times.append(np.array(t_c))
        continuous_trajs.append(np.array(y_c))
    
    return continuous_trajs, continuous_times


def animate_continuous_agents():
    trajs, times = create_continuous_trajectories()
    n_agents = len(trajs)
    
    total_time = max(times[i][-1] for i in range(n_agents))
    
    fps = 100
    n_frames = int(total_time * fps)
    uniform_times = np.linspace(0, total_time, n_frames)
    
    interpolated_trajs = []
    for i in range(n_agents):
        raw_traj = trajs[i]       
        raw_times = times[i]      

        interp_x = interp1d(raw_times, raw_traj[:, 0],
                            kind='linear',
                            bounds_error=False,
                            fill_value='extrapolate')
        interp_y = interp1d(raw_times, raw_traj[:, 1],
                            kind='linear',
                            bounds_error=False,
                            fill_value='extrapolate')

        x_uniform = interp_x(uniform_times)
        y_uniform = interp_y(uniform_times)

        wrapped = np.column_stack([
            np.mod(x_uniform, 1.0),
            np.mod(y_uniform, 1.0)
        ])

        interpolated_trajs.append(wrapped)
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1.0) 
    ax.set_ylim(0, 1.0)  
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Draw boundaries
    ax.axvline(lower_boundary, color='black', linewidth=2, alpha=0.7)
    ax.axhline(lower_boundary, color='black', linewidth=2, alpha=0.7)
    ax.plot([upper_boundary, 1], [1, upper_boundary], color='black', linewidth=2) 
    # ax.axhline(0, color='red', linewidth=1, alpha=0.5, linestyle='--', label='y=0 (reset line)')
    
    X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    epsilon = 0.3

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]
            in_bottom_right_triangle = (
                (x >= upper_boundary) and (y <= lower_boundary) and (y < x - upper_boundary)
            )
            in_top_left_triangle = (
                (x <= lower_boundary) and (y >= upper_boundary) and (y > x + upper_boundary)
            )
            on_diagonal = (
                abs(x - y) < epsilon
            )

            if in_bottom_right_triangle or in_top_left_triangle or on_diagonal:
                vec = polygonal_field(0, [x, y])
                U[i, j] = vec[0]
                V[i, j] = vec[1]
            else:
                U[i, j] = np.nan
                V[i, j] = np.nan

    ax.quiver(X, Y, U, V, color='gray', alpha=0.4, scale=25) #, headwidth=3, headlength=5, headaxislength=4)

    # Create plot
    colors = ['blue', 'red', 'green', 'orange']
    trails = [ax.plot([], [], '-', color=colors[i], alpha=0.6, linewidth=2)[0] 
              for i in range(n_agents)]
    points = [ax.plot([], [], 'o', color=colors[i], markersize=8)[0] 
              for i in range(n_agents)]
    
    ax.set_title("Continuous Agent Trajectories")
    
    def init():
        for trail, point in zip(trails, points):
            trail.set_data([], [])
            point.set_data([], [])
        return trails + points


    def update(frame):
        trail_length = 0.2

        def toroidal_diff(a):
            d = np.diff(a)
            return ((d + 0.5) % 1.0) - 0.5

        for i in range(n_agents):
            x_all = interpolated_trajs[i][:frame+1, 0]
            y_all = interpolated_trajs[i][:frame+1, 1]
            valid = ~(np.isnan(x_all) | np.isnan(y_all))
            if np.sum(valid) > 1:
                xv = x_all[valid]
                yv = y_all[valid]

                dx = toroidal_diff(xv)
                dy = toroidal_diff(yv)
                dists = np.hypot(dx, dy)
                cumd = np.concatenate(([0], np.cumsum(dists)))
                total = cumd[-1]
                start = max(0, total - trail_length)
                idxs = np.where(cumd >= start)[0]

                x_trail = xv[idxs]
                y_trail = yv[idxs]

                ddx = np.diff(x_trail)
                ddy = np.diff(y_trail)
                wrap_spots = np.where(
                    (ddx >  0.5) | (ddx < -0.5) |
                    (ddy >  0.5) | (ddy < -0.5)
                )[0]

                for j in wrap_spots[::-1]:
                    x_trail = np.insert(x_trail, j+1, np.nan)
                    y_trail = np.insert(y_trail, j+1, np.nan)

            else:
                x_trail = x_all
                y_trail = y_all

            trails[i].set_data(x_trail, y_trail)
            xc, yc = interpolated_trajs[i][frame]
            points[i].set_data([xc], [yc])

        return trails + points
    
    anim = FuncAnimation(fig, update, frames=len(uniform_times), init_func=init,
                         interval=1000/fps, blit=True, repeat=True)
    
    anim.save("agents_animation.mp4", writer="ffmpeg", fps=fps)
    # plt.show()
    
    return anim


if __name__ == "__main__":
    animate_continuous_agents()