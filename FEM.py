from dolfin import *
from mshr import Box, generate_mesh
import numpy as np

# Room dimensions (m)
Lx, Ly, Lz = 3.0, 2.5, 2.2

# Speed of sound
c = 343.0  # m/s

# 1) Create mesh
domain = Box(Point(0.0, 0.0, 0.0), Point(Lx, Ly, Lz))
# Increase resolution for better accuracy (at cost of CPU/memory)
mesh = generate_mesh(domain, 32)

# 2) Function space (P1 Lagrange)
V = FunctionSpace(mesh, "Lagrange", 1)
u = TrialFunction(V)
v = TestFunction(V)

# 3) Bilinear forms: stiffness A and mass B
a = inner(grad(u), grad(v))*dx
m = u*v*dx

# 4) “Pin” one corner to remove zero eigenvalue (constant mode)
corner = "near(x[0], 0.0) && near(x[1], 0.0) && near(x[2], 0.0)"
bc = DirichletBC(V, Constant(0.0), corner)

# 5) Assemble matrices
A = PETScMatrix()
B = PETScMatrix()
assemble(a, tensor=A)
assemble(m, tensor=B)
bc.apply(A)
bc.apply(B)

# 6) Set up SLEPc eigenproblem solver
eigensolver = SLEPcEigenSolver(A, B)
eigensolver.parameters["spectrum"] = "smallest real"
eigensolver.parameters["tolerance"] = 1e-8
eigensolver.solve(10)  # compute first 10 eigenpairs

# 7) Extract and convert to frequencies
modes = []
for i in range(1, 10):  # skip i=0 if it’s near-zero
    r, _, _, _ = eigensolver.get_eigenpair(i)
    if r < DOLFIN_EPS:  # skip any spurious zero
        continue
    fi = c/(2*np.pi)*np.sqrt(r)
    modes.append((i, r, fi))

# 8) Print results
print(f"{'Mode #':>6s}   {'λ (1/m²)':>10s}   {'f (Hz)':>8s}")
print("-"*32)
for i, lam, freq in modes:
    print(f"{i:6d}   {lam:10.4e}   {freq:8.2f}")

# 9) (Optional) Save first mode for visualization
# from dolfin import Function, File
# idx = modes[0][0]
# r0, _, vec0, _ = eigensolver.get_eigenpair(idx)
# mode0 = Function(V)
# mode0.vector()[:] = vec0
# File("mode1_3D.pvd") << mode0
