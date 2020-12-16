import cvxpy as cp
import numpy as np
from PIL import Image

# Problem data
y = np.load("y.npy").squeeze()
C = np.load("C.npy")
n = C.shape[1]
print(y.shape, C.shape, n)

# Construct the problem
s = cp.Variable(shape=n)
objective = cp.Minimize(cp.norm(s, 1))
# norm2(y - Cs) = 0 is equivalent to y = Cs
constraints = [(y - (C @ s)) == 0]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve(solver='OSQP', eps_abs=1.0e-08, eps_rel=1.0e-08, verbose=True)
print(s.value.shape)

# Load A_inv
A_inv = np.load("A_inv.npy")
print(A_inv.shape)

# Since A is orthogonal, A = transpose(A_inv)
x = A_inv.T @ s.value

# Clip x values to range 0-255, reshape x and save the image
clip_x = x.copy()
clip_x = np.clip(clip_x, 0, 255).astype(np.uint8)
clip_im = Image.fromarray(clip_x.reshape(100, -1).T)
clip_im.save("reconstructed.png")
