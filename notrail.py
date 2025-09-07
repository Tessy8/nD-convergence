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

def simulate_converging_agents_cycles():
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

        curr_positions = new_positions

    return all_trajectories

def animate_converging_agents_cycles():
    trajectories = simulate_converging_agents_cycles()
    n_agents = len(trajectories)
    cycles = len(trajectories[0])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-2, 4)
    ax.set_ylim(0, 3)
    ax.set_title("Converging Agents per Cycle")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect('equal')

    all_frames = []
    for cycle in range(cycles):
        max_len = max(len(traj[cycle]) for traj in trajectories)
        for step in range(max_len):
            all_frames.append((cycle, step))
        all_frames.extend([(cycle, max_len - 1)] * 5)

    def init():
        return []

    def update(frame):
        cycle, step = frame
        ax.clear()
        ax.set_xlim(-2, 4)
        ax.set_ylim(0, 3)
        ax.set_title(f"Converging Agents - Cycle {cycle + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.set_aspect('equal')

        for i in range(n_agents):
            traj = trajectories[i][cycle]
            if step < len(traj):
                ax.plot(traj[:step+1, 0], traj[:step+1, 1], '-', lw=1.5)
                ax.plot(traj[step, 0], traj[step, 1], 'o')
            else:
                ax.plot(traj[:, 0], traj[:, 1], '-', lw=1.5)
                ax.plot(traj[-1, 0], traj[-1, 1], 'o')

        return ax.patches + ax.lines

    anim = FuncAnimation(fig, update, frames=all_frames, init_func=init, interval=50, blit=True)
    anim.save("no_trail.mp4", writer="ffmpeg")

if __name__ == "__main__":
    animate_converging_agents_cycles()
