from dolfin import *
import numpy as np
from dolfin import Function, File
import matplotlib.pyplot as plt


def FEM_solver_display(num_eigenmodes, mode_shape_idx):
    """Calcula el problema de autovalores y autovectores para la geometría generada. Y guarda
    archivos para visualizar.

    Args:
        num_eigenmodes (int):  Numero de soluciones (autovaloes) que calcula
        mode_shape_idx (int): Index de la forma modal que se guarda para graficar
    """
    # Speed of sound
    c = 343.0  # m/s

    # 1) Read mesh from 3D geometry generator
    mesh = Mesh()
    with XDMFFile("room.xdmf") as xf:
        xf.read(mesh)

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
    eigensolver.solve(num_eigenmodes)  # compute first x eigenpairs

    # 7) Extract and convert to frequencies
    modes = []
    for i in range(1, num_eigenmodes):  # skip i=0 if it’s near-zero
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

    r0, _, vec0, _ = eigensolver.get_eigenpair(mode_shape_idx)
    mode1 = Function(V)
    mode1.vector()[:] = vec0

    # 10) Save files for visualization with PyVista

    # Write out the mesh (so meshio can read it later too)
    with XDMFFile("room_mesh.xdmf") as mesh_file:
        mesh_file.write(mesh)

    # Write out the eigenfunction
    with XDMFFile("mode1.xdmf") as mode_file:
        mode_file.write(mode1)

def FEM_solver(num_eigenmodes):
    """Calcula el problema de autovalores y autovectores para la geometría generada.

    Args:
        num_eigenmodes (int):  Numero de soluciones (autovaloes) que calcula
        mode_shape_idx (int): Index de la forma modal que se guarda para graficar
    Returns:
        (np.array): Frecuencias de donde caen los autovalores
    
    """
    # Speed of sound
    c = 343.0  # m/s

    # 1) Read mesh from 3D geometry generator
    mesh = Mesh()
    with XDMFFile("room.xdmf") as xf:
        xf.read(mesh)

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
    eigensolver.solve(num_eigenmodes)  # compute first 10 eigenpairs

    # 7) Extract and convert to frequencies
    modes = []
    for i in range(1, num_eigenmodes):  # skip i=0 if it’s near-zero
        try:
            r, _, _, _ = eigensolver.get_eigenpair(i)
        except:
            print("Algo paso con la cantidad de autovalores")
            continue
        if r < DOLFIN_EPS:  # skip any spurious zero
            continue

        fi = c/(2*np.pi)*np.sqrt(r)
        modes.append(fi)
    
    return np.array(modes)
