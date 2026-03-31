from hybrid_tools import PI, integrate_pc
import numpy as np

# Sample points that do NOT converge at c=0
# and check if any converge at c=0.25
vals = np.linspace(-PI + 0.4, PI - 0.4, 8)
g = np.meshgrid(vals, vals, vals, indexing='ij')
X = np.stack([g[0].ravel(), g[1].ravel(), g[2].ravel()], axis=1)

conv_c0   = []
conv_c025 = []
for x0 in X:
    r0   = integrate_pc(x0, 0.0,  0.5, 60.0, 0.05, 1e-5, keep_trajectory=False)
    r025 = integrate_pc(x0, 0.25, 0.5, 60.0, 0.05, 1e-5, keep_trajectory=False)
    conv_c0.append(r0["converged"])
    conv_c025.append(r025["converged"])

conv_c0   = np.array(conv_c0)
conv_c025 = np.array(conv_c025)

# Points that gain convergence when guard shifts
gained = X[~conv_c0 & conv_c025]
lost   = X[conv_c0 & ~conv_c025]
print(f"Gained convergence at c=0.25: {len(gained)}")
print(f"Lost convergence at c=0.25:   {len(lost)}")
if len(gained): print("Sample gained:", gained[:3].round(3))
if len(lost):   print("Sample lost:  ", lost[:3].round(3))