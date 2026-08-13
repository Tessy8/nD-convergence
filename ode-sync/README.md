# ODE sync work

This folder is a short copy of my first synchronization work using the piecewise constant ODE setup Professor Revzen gave me.

Files:
- `odepc.py` - the piecewise constant ODE integrator.
- `syncTime.py` - the 2D synchronization simulation.
- `syncTime4D.py` - the 4D version.
- `syncTime4DWithNoise.py` - the 4D version with noise.
- `syncTime7D.py` - the 7D version.

Run the simulation you want directly, for example:

```bash
python syncTime.py
python syncTime4D.py
```

Some of the later files use extra packages for noise/plotting, so those may need the same environment I used when I wrote them.
