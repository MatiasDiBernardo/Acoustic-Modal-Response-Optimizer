from mpi4py import MPI
import numpy as np
import traceback
from dolfinx.io import gmshio
from dolfinx.fem import Function, Constant, form
from dolfinx import geometry, fem
from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix, assemble_vector as petsc_assemble_vector
from ufl import TrialFunction, TestFunction, inner, grad, dx, Measure
from petsc4py import PETSc
from basix.ufl import element

def from_position_to_grid(rec_loc, space_grid):
    center = np.array(rec_loc, dtype=float).reshape(3)
    dx_vec = np.array([space_grid, 0.0, 0.0])
    dy_vec = np.array([0.0, space_grid, 0.0])
    dz_vec = np.array([0.0, 0.0, space_grid])
    return [
        center,
        center + dx_vec, center - dx_vec,
        center + dy_vec, center - dy_vec,
        center + dz_vec, center - dz_vec
    ]

def FEM_Source_Solver_Average(frequency, mesh_filename, rec_loc, verbose=False):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    omega = 2 * np.pi * np.array(frequency)
    c0 = 343.0
    rho0 = 1.225
    U_normal_sphere = 0.01
    sphere_tag = 7

    # 1) Read and partition mesh
    try:
        mesh, cell_tags, facet_tags = gmshio.read_from_msh(
            mesh_filename, comm=comm, rank=0, gdim=3
        )
        # Global tag check
        local_has = facet_tags is not None and sphere_tag in np.unique(facet_tags.values)
        global_has = comm.allreduce(local_has, op=MPI.LOR)
        if rank == 0 and not global_has:
            raise RuntimeError(f"Missing sphere tag {sphere_tag} in entire mesh")
        if not global_has:
            comm.Barrier()
            return None
        # Connectivity
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim-1, tdim)
        mesh.topology.create_connectivity(tdim, tdim-1)
    except Exception as e:
        if rank == 0:
            print(f"ERROR loading mesh: {e}")
            traceback.print_exc()
        return None

    # 2) Complex function space
    try:
        # Create complex function space
        V = fem.functionspace(mesh, ("Lagrange", 1))
    except Exception as e:
        if rank == 0:
            print(f"Error creating complex function space: {e}")
            traceback.print_exc()
        return None

    # 3) Receiver points and ownership
    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    pos_list = from_position_to_grid(rec_loc, space_grid=0.1)
    idx_list = []
    for pos in pos_list:
        pts = np.array([pos], dtype=np.float64)
        coll = geometry.compute_collisions_points(tree, pts)
        cells = geometry.compute_colliding_cells(mesh, coll, pts)
        idx_list.append(cells.links(0)[0] if cells.links(0).size > 0 else -1)

    # Precompute ownership information
    tdim = mesh.topology.dim
    imap = mesh.topology.index_map(tdim)
    rstart, rend = imap.local_range
    owner_mask = []
    for cell in idx_list:
        if cell >= 0:
            global_index = imap.local_to_global([cell])[0]
            owner_mask.append(rstart <= global_index < rend)
        else:
            owner_mask.append(False)

    if verbose and rank == 0:
        print("Positions:", pos_list)
        print("Cell idx:", idx_list)
        print("Owner mask:", owner_mask)

    # 4) Result placeholder on root
    if rank == 0:
        magnitude = np.zeros((len(pos_list), len(omega)))

    # 5) Frequency solve loop
    for i, w in enumerate(omega):
        k_wave = w / c0
        neumann_val = -1j * w * rho0 * U_normal_sphere
        gN = Constant(mesh, PETSc.ScalarType(neumann_val))

        # Trial/Test
        p = TrialFunction(V)
        v = TestFunction(V)

        # Forms
        a_form = inner(grad(p), grad(v)) * dx - k_wave**2 * inner(p, v) * dx
        ds_s = Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=sphere_tag)
        L_form = -inner(gN, v) * ds_s

        # Assemble to PETSc objects
        A = petsc_assemble_matrix(form(a_form), bcs=[])
        A.assemble()  # Critical fix: Assemble the matrix
        b = petsc_assemble_vector(form(L_form))

        # Solver
        x = Function(V)
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        ksp.solve(b, x.x.petsc_vec)

        # Gather values with owner-only evaluation
        local_vals = np.zeros(len(pos_list), dtype=np.complex128)
        for j, (cell, is_owner) in enumerate(zip(idx_list, owner_mask)):
            if cell >= 0 and is_owner:
                local_vals[j] = x.eval(np.array(pos_list[j]), [cell])[0]

        global_vals = np.zeros_like(local_vals)
        comm.Reduce([local_vals, MPI.COMPLEX], [global_vals, MPI.COMPLEX], op=MPI.SUM, root=0)
        if rank == 0:
            magnitude[:, i] = np.abs(global_vals)

    if rank == 0:
        return 20 * np.log10(magnitude)
    return None
