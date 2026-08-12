import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class OdePC:
    def __init__(self, fun):
        self._fun = fun

    def __call__(self, y0, t0, t1=None, dt=None, tTol=1e-6, pars=None, allclose=np.allclose, withDy=False):
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
                h = h / 2.0
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
            y.append(y[-1] + dy0[-1] * h0)
            assert th == t[-1] + h0
            t.append(th)
            h = min(h * 1.5, dt)
            continue

        if withDy:
            return np.asarray(t, dtype=float), np.asarray(y, dtype=float), np.asarray(dy0, dtype=float)
        else:
            return np.asarray(t, dtype=float), np.asarray(y, dtype=float)

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

def simulate_converging_agents():
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
    t_total = 0

    for _ in range(cycles):
        new_positions = []
        for i, y0 in enumerate(curr_positions):
            t, y, dy = ode(y0, t0=0, t1=2, dt=0.1, tTol=1e-3, withDy=True)
            all_trajectories[i].append(y)

            last_dy = dy[-1]
            if np.allclose(last_dy, 0):
                for past_dy in reversed(dy):
                    if not np.allclose(past_dy, 0):
                        last_dy = past_dy
                        break

            new_start = project_back(y[-1], last_dy)
            new_positions.append(new_start)

        curr_positions = new_positions #.append

    return all_trajectories

def animate_converging_agents():
    trajectories = simulate_converging_agents()
    n_agents = len(trajectories)
    all_trajs = [np.vstack(agent_trajs) for agent_trajs in trajectories]
    max_len = max(len(traj) for traj in all_trajs)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-2, 4)
    ax.set_ylim(0, 3)
    ax.grid(True)
    ax.set_title("Converging Agents")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    points = [ax.plot([], [], 'o')[0] for _ in range(n_agents)]
    trails = [ax.plot([], [], '--')[0] for _ in range(n_agents)]

    def init():
        for pt, tr in zip(points, trails):
            pt.set_data([], [])
            tr.set_data([], [])
        return points + trails

    def update(frame):
        for i, traj in enumerate(all_trajs):
            if frame < len(traj):
                points[i].set_data([traj[frame][0]], [traj[frame][1]])
                trails[i].set_data(traj[:frame+1, 0], traj[:frame+1, 1])
        return points + trails

    anim = FuncAnimation(fig, update, frames=max_len, init_func=init, interval=50, blit=True)
    anim.save("trail.mp4", writer="ffmpeg")

if __name__ == "__main__":
    animate_converging_agents()
