# -*- coding: utf-8 -*-
# !!! NOTA IMPORTANTE SOBRE PETSC Y NÚMEROS COMPLEJOS !!!
# Esta simulación requiere PETSc compilado con soporte para escalares complejos.
# ------------------------------------------------------------------------------------

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, mesh, cpp, geometry
from dolfinx.fem import form, Function, Constant
from dolfinx.fem.petsc import LinearProblem, assemble_vector, apply_lifting, set_bc, assemble_matrix # Import assemble_matrix
from dolfinx.io import VTKFile
from scipy.spatial import cKDTree
from ufl import (TestFunction, TrialFunction, dx, ds, grad, inner,
                 Measure)
import os
import traceback # Import traceback for better error reporting

def FEM_Source_Solver(frequency, mesh_filename, rec_loc):
    """Solver the Hemholtz equation for given room geometry

    Args:
        frequency (np.array): Array of frequencies to evaluate
        mesh_filename (str): Path to the geometry to evaluate
        rec_loc (tuple[float]): Tuple with (X, Y, Z) location of the receptor

    Returns:
        np.array: Magnitude of the pressure
    """

    # Parámetros físicos y de la simulación
    omega = 2 * np.pi * frequency # Frecuencia angular (rad/s)
    c0 = 343.0          # Velocidad del sonido en el aire (m/s)
    rho0 = 1.225        # Densidad del aire (kg/m^3)

    # Amplitud de la velocidad normal en la superficie de la esfera interna (m/s)
    U_normal_sphere = 0.01 # Ejemplo: 1 cm/s

    # Marcadores de faceta (esto es la medida de la esfera??)
    sphere_facet_marker = 7

    # Cargar malla (.msh)
    try:
        # Leer la malla y los tags de celdas y facetas
        msh, cell_tags, facet_tags = io.gmshio.read_from_msh(
            mesh_filename, MPI.COMM_WORLD, rank=0, gdim=3
        )

        # Verificación de facet_tags
        if facet_tags is None or facet_tags.values is None or facet_tags.values.size == 0:
            print("ERROR CRÍTICO: facet_tags no se cargó correctamente o está vacío.")
            exit()
        
        unique_tags_found = np.unique(facet_tags.values)
        # print(f"Tags de faceta únicos encontrados en la malla: {unique_tags_found}")
        # Verificamos solo el tag de la esfera ahora
        required_tags = {sphere_facet_marker}
        if not required_tags.issubset(unique_tags_found):
            print(f"ERROR CRÍTICO: Faltan tags de faceta requeridos. Encontrados: {unique_tags_found}, Requeridos: {required_tags}")
            exit()
        # print(f"Marcador de la esfera (tag {sphere_facet_marker}) encontrado en la malla.")

        # Crear conectividad necesaria
        msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim) # Facets to cells
        msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1) # Cells to facets
        
        # print(f"Malla cargada: {msh.topology.dim}D, {msh.topology.index_map(msh.topology.dim).size_local} celdas locales (proceso {MPI.COMM_WORLD.rank})")

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de malla '{mesh_filename}'.")
        exit()
    except Exception as e:
        print(f"ERROR: Ocurrió un problema al cargar la malla: {e}")
        traceback.print_exc()
        exit()

    # Espacio de funciones
    degree = 1
    V = fem.functionspace(msh, ("Lagrange", degree))

    # Definición del problema variacional (UFL)
    p_trial = TrialFunction(V)
    v_test = TestFunction(V)

    # Definir medida para la esfera
    ds_sphere = Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=sphere_facet_marker)

    c_type = PETSc.ScalarType
    if not np.issubdtype(c_type, np.complexfloating) and MPI.COMM_WORLD.rank == 0:
        print("ADVERTENCIA: PETSc ScalarType no es complejo. Los resultados pueden ser incorrectos.")
    
    # # Defino los puntos a evaluar en el espacio
    receiver_coords = np.array([rec_loc[0], rec_loc[1], rec_loc[2]])
    receiver_point_eval = np.array([receiver_coords]) # Para eval, necesita forma (1,3)

    tree = geometry.bb_tree(msh, msh.topology.dim) 
    candidate_cells_adj = geometry.compute_collisions_points(tree, receiver_point_eval.astype(np.float64))
    receptor_cell_idx = -1
    if candidate_cells_adj.num_nodes > 0: 
        colliding_cells_adj = geometry.compute_colliding_cells(msh, candidate_cells_adj, receiver_point_eval.astype(np.float64))
        if colliding_cells_adj.num_nodes > 0: 
            cells_found_for_point_0 = colliding_cells_adj.links(0) 
            if len(cells_found_for_point_0) > 0:
                receptor_cell_idx = cells_found_for_point_0[0] 
    
    print("La locación buscada es: ", receiver_coords)
    print("Se ingrea al evaluador: ", receiver_point_eval)
    print("Con shape: ", receiver_point_eval.shape)
    print("El indice es: ", receptor_cell_idx)
    print("Con shape: ", receptor_cell_idx.shape)


    # Itero en frecuencia
    magnitude = []
    PETSc.Options().clear()
    for om in omega:
        k_wave = om / c0      # Número de onda (rad/m)
        
        # Condición de Neumann en la esfera: dp/dn = g_N
        neumann_value = -1j * om * rho0 * U_normal_sphere
        neumann_term = fem.Constant(msh, c_type(neumann_value)) # Esto es g_N
        # print(f"Valor del término de Neumann (g_N) en la esfera: {neumann_value:.2e}")

        a_form_ufl = inner(grad(p_trial), grad(v_test)) * dx \
                    - k_wave**2 * inner(p_trial, v_test) * dx

        # Forma lineal (lado derecho de la ecuación débil)
        # Sigue siendo solo el término de Neumann en la esfera
        L_form_ufl = -inner(neumann_term, v_test) * ds_sphere # SIGNO CORREGIDO

        bcs = [] # Sin condiciones de Dirichlet

        A_form = form(a_form_ufl, dtype=PETSc.ScalarType)
        b_form = form(L_form_ufl, dtype=PETSc.ScalarType)

        # 2) Assemble system
        A = assemble_matrix(A_form, bcs=bcs)
        A.assemble()

        b = assemble_vector(b_form)
        for bc in bcs:
            bc.apply(b)
        
        x_petsc = A.createVecLeft()       # this is a petsc4py.PETSc.Vec

        # 3) Build your custom KSP/PC
        ksp = PETSc.KSP().create(msh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")

        # 4) Solve into x_petsc
        ksp.solve(b, x_petsc)

        # 5) Copy the data back into your Function
        p_solution = Function(V, dtype=PETSc.ScalarType)
        # DOLFINx Vector (p_solution.x) has an .array property you can overwrite:
        p_solution.x.array[:] = x_petsc.getArray()

        # Guarda el resultado en los puntos del espacio elegidos
        if receptor_cell_idx != -1:
            print(f"Punto receptor encontrado en la celda local {receptor_cell_idx} en el proceso {msh.comm.rank}")
            solucion_compleja = p_solution.eval(receiver_point_eval, [receptor_cell_idx])
            magnitude.append(np.abs(solucion_compleja))
            print(solucion_compleja)
        else:
            if msh.comm.size > 1:
                _found_locally = 1 if receptor_cell_idx != -1 else 0 
                found_globally = msh.comm.allreduce(_found_locally, op=MPI.SUM)
                if found_globally == 0 and msh.comm.rank == 0 : 
                    print(f"ADVERTENCIA: Punto receptor {receiver_coords} no encontrado en ninguna celda en ningún proceso.")
            elif msh.comm.rank == 0 : # Serial y no encontrado
                print(f"ADVERTENCIA: Punto receptor {receiver_coords} no encontrado en ninguna celda.")
    
    return np.array(magnitude)

def FEM_Source_Solver_Spatial_Average(frequency, mesh_filename, rec_loc):
    """Solver the Hemholtz equation for given room geometry

    Args:
        frequency (np.array): Array of frequencies to evaluate
        mesh_filename (str): Path to the geometry to evaluate
        rec_loc (tuple[float]): Tuple with (X, Y, Z) location of the receptor

    Returns:
        np.array: Magnitude of the pressure
    """

    # Parámetros físicos y de la simulación
    omega = 2 * np.pi * frequency # Frecuencia angular (rad/s)
    c0 = 343.0          # Velocidad del sonido en el aire (m/s)
    rho0 = 1.225        # Densidad del aire (kg/m^3)

    # Amplitud de la velocidad normal en la superficie de la esfera interna (m/s)
    U_normal_sphere = 0.01 # Ejemplo: 1 cm/s

    # Marcadores de faceta (esto es la medida de la esfera??)
    sphere_facet_marker = 7

    # Cargar malla (.msh)
    try:
        # Leer la malla y los tags de celdas y facetas
        msh, cell_tags, facet_tags = io.gmshio.read_from_msh(
            mesh_filename, MPI.COMM_WORLD, rank=0, gdim=3
        )

        # Verificación de facet_tags
        if facet_tags is None or facet_tags.values is None or facet_tags.values.size == 0:
            print("ERROR CRÍTICO: facet_tags no se cargó correctamente o está vacío.")
            exit()
        
        unique_tags_found = np.unique(facet_tags.values)
        # print(f"Tags de faceta únicos encontrados en la malla: {unique_tags_found}")
        # Verificamos solo el tag de la esfera ahora
        required_tags = {sphere_facet_marker}
        if not required_tags.issubset(unique_tags_found):
            print(f"ERROR CRÍTICO: Faltan tags de faceta requeridos. Encontrados: {unique_tags_found}, Requeridos: {required_tags}")
            exit()
        # print(f"Marcador de la esfera (tag {sphere_facet_marker}) encontrado en la malla.")

        # Crear conectividad necesaria
        msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim) # Facets to cells
        msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1) # Cells to facets
        
        # print(f"Malla cargada: {msh.topology.dim}D, {msh.topology.index_map(msh.topology.dim).size_local} celdas locales (proceso {MPI.COMM_WORLD.rank})")

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de malla '{mesh_filename}'.")
        exit()
    except Exception as e:
        print(f"ERROR: Ocurrió un problema al cargar la malla: {e}")
        traceback.print_exc()
        exit()

    # Espacio de funciones
    degree = 1
    V = fem.functionspace(msh, ("Lagrange", degree))

    # Definición del problema variacional (UFL)
    p_trial = TrialFunction(V)
    v_test = TestFunction(V)

    # Definir medida para la esfera
    ds_sphere = Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=sphere_facet_marker)

    c_type = PETSc.ScalarType
    if not np.issubdtype(c_type, np.complexfloating) and MPI.COMM_WORLD.rank == 0:
        print("ADVERTENCIA: PETSc ScalarType no es complejo. Los resultados pueden ser incorrectos.")
    
    # Defino los puntos a evaluar en el espacio
    space_points = 6
    receiver_coords = np.array([rec_loc[0], rec_loc[1], rec_loc[2]])
    receiver_coords = np.array(rec_loc, dtype=np.float64).reshape(3,)

    # 2) grab all vertex coordinates from the mesh
    #    msh.geometry.x is of shape (num_vertices, 3)
    all_coords = msh.geometry.x

    # 3) build a global k-d tree on the vertices
    tree = cKDTree(all_coords)

    # 4) query the 5 nearest neighbors
    #    `idxs` is length-5 array of indices, `dists` their distances
    dists, idxs = tree.query(receiver_coords, k=space_points)

    # 5) collect their positions
    new_pos = all_coords[idxs]   # shape (5,3)
    new_idx = idxs.reshape(-1, 1)
    print("Locación objetivo: ", receiver_coords)
    print("Solucion con cKDTree", new_pos)
    # print("Shape", new_pos.shape)
    print("Idx", new_idx)
    # print("Shape", new_idx.shape)
    print("Dist", dists)

    # Itero en frecuencia
    magnitude_matriz = np.zeros((space_points, len(omega)))
    PETSc.Options().clear()
    for i in range(len(omega)):
        k_wave = omega[i] / c0      # Número de onda (rad/m)
        
        # Condición de Neumann en la esfera: dp/dn = g_N
        neumann_value = -1j * omega[i] * rho0 * U_normal_sphere
        neumann_term = fem.Constant(msh, c_type(neumann_value)) # Esto es g_N
        # print(f"Valor del término de Neumann (g_N) en la esfera: {neumann_value:.2e}")

        a_form_ufl = inner(grad(p_trial), grad(v_test)) * dx \
                    - k_wave**2 * inner(p_trial, v_test) * dx

        # Forma lineal (lado derecho de la ecuación débil)
        # Sigue siendo solo el término de Neumann en la esfera
        L_form_ufl = -inner(neumann_term, v_test) * ds_sphere # SIGNO CORREGIDO

        bcs = [] # Sin condiciones de Dirichlet

        A_form = form(a_form_ufl, dtype=PETSc.ScalarType)
        b_form = form(L_form_ufl, dtype=PETSc.ScalarType)

        # 2) Assemble system
        A = assemble_matrix(A_form, bcs=bcs)
        A.assemble()

        b = assemble_vector(b_form)
        for bc in bcs:
            bc.apply(b)
        
        x_petsc = A.createVecLeft()       # this is a petsc4py.PETSc.Vec

        # 3) Build your custom KSP/PC
        ksp = PETSc.KSP().create(msh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")

        # 4) Solve into x_petsc
        ksp.solve(b, x_petsc)

        # 5) Copy the data back into your Function
        p_solution = Function(V, dtype=PETSc.ScalarType)
        # DOLFINx Vector (p_solution.x) has an .array property you can overwrite:
        p_solution.x.array[:] = x_petsc.getArray()

        # Guarda el resultado en los puntos del espacio elegidos (asume que se encontraron 5 puntos)
        for j in range(new_pos.shape[0]):
            pos_real = np.array([new_pos[j,:]])
            idx_mesh = new_idx[j, :][0]
            solucion_compleja = p_solution.eval(pos_real, [idx_mesh])
            val = np.abs(solucion_compleja)
            magnitude_matriz[j, i] = val
        
    return magnitude_matriz
