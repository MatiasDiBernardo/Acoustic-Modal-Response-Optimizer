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
import dolfinx.log
from scipy.spatial import cKDTree
from ufl import (TestFunction, TrialFunction, dx, ds, grad, inner,
                 Measure)
import os
import traceback # Import traceback for better error reporting
dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)

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

    # Con esto sabe donde esta la esfera
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

def from_position_to_grid(pos, dx):
    new_pos = [
               np.array([pos[0], pos[1], pos[2]]),
               np.array([pos[0] + dx, pos[1], pos[2]]),
               np.array([pos[0] - dx, pos[1], pos[2]]),
               np.array([pos[0], pos[1] + dx, pos[2]]),
               np.array([pos[0], pos[1] - dx, pos[2]]),
               np.array([pos[0], pos[1], pos[2] + dx]),
               np.array([pos[0], pos[1], pos[2] - dx]),
             ]
    return new_pos


def respuesta_esfera_pulsante_velocidad(freqs, radio_esfera, distancia_receptor, rho0=1.21, c0=343.0):
    """
    Devuelve la magnitud de la respuesta en frecuencia (presión) en campo lejano
    para una esfera pulsante con velocidad constante en su superficie.
    """
    freqs = np.atleast_1d(freqs)
    omega = 2 * np.pi * freqs
    k = omega / c0
    ka = k * radio_esfera
    kr = k * distancia_receptor

    # Numerador y denominador en magnitud
    numerador = np.abs(ka * np.exp(-1j * kr))
    denominador = np.abs(1 - 1j * ka)

    respuesta = rho0 * (radio_esfera / distancia_receptor) * (numerador / denominador)
    return respuesta


def FEM_Source_Solver_Average(frequency, mesh_filename, rec_loc, verbose=False, degree = 1):
    """Solver the Hemholtz equation for given room geometry

    Args:
        frequency (np.array): Array of frequencies to evaluate
        mesh_filename (str): Path to the geometry to evaluate
        rec_loc (tuple[float]): Tuple with (X, Y, Z) location of the receptor. It generates 6 nearest points

    Returns:
        np.array: Magnitude of the pressure
    """

    # Parámetros físicos y de la simulación
    omega = 2 * np.pi * frequency # Frecuencia angular (rad/s)
    c0 = 343.0          # Velocidad del sonido en el aire (m/s)
    rho0 = 1.225        # Densidad del aire (kg/m^3)

    # Amplitud de la velocidad normal en la superficie de la esfera interna (m/s)
    U_normal_sphere = 0.01 # Ejemplo: 1 cm/s

    # Con esto sabe donde esta la esfera
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

    V = fem.functionspace(msh, ("Lagrange", degree))

    # Definición del problema variacional (UFL)
    p_trial = TrialFunction(V)
    v_test = TestFunction(V)

    # Definir medida para la esfera
    ds_sphere = Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=sphere_facet_marker)

    # Se calcula el área de la superficie de la fuente integrando 1 sobre sus facetas.
    # Para ello, se define una constante 1.0 sobre la malla para la integración.
    one = fem.Constant(msh, PETSc.ScalarType(1.0))
    superficie_esfera = fem.assemble_scalar(fem.form(one * ds_sphere))
    
    # Se despeja el radio de la fórmula del área de una esfera: A = 4*pi*r^2  => r = sqrt(A / (4*pi))
    radio_esfera_calculado = np.sqrt(superficie_esfera / (4 * np.pi))
    
    # print(f"Radio de la esfera fuente detectado automáticamente: {radio_esfera_calculado:.4f} m")


    c_type = PETSc.ScalarType
    if not np.issubdtype(c_type, np.complexfloating) and MPI.COMM_WORLD.rank == 0:
        print("ADVERTENCIA: PETSc ScalarType no es complejo. Los resultados pueden ser incorrectos.")
    
    # Defino los puntos a evaluar en el espacio
    tree = geometry.bb_tree(msh, msh.topology.dim) 
    space_spatial_grid = 0.1  # Espaciado para el promedidado espacial del SV
    pos_list = from_position_to_grid(rec_loc, space_spatial_grid)
    idx_list = []
    for pos in pos_list:
        receiver_point_eval = np.array([pos]) # Para eval, necesita forma (1,3)

        candidate_cells_adj = geometry.compute_collisions_points(tree, receiver_point_eval.astype(np.float64))
        receptor_cell_idx = -1
        if candidate_cells_adj.num_nodes > 0: 
            colliding_cells_adj = geometry.compute_colliding_cells(msh, candidate_cells_adj, receiver_point_eval.astype(np.float64))
            if colliding_cells_adj.num_nodes > 0: 
                cells_found_for_point_0 = colliding_cells_adj.links(0) 
                if len(cells_found_for_point_0) > 0:
                    receptor_cell_idx = cells_found_for_point_0[0] 
                    idx_list.append(receptor_cell_idx)
    
    assert len(pos_list) == len(idx_list), "Un elemento de la grilla de puntos espaciales no se le pudo asignar malla"

    if verbose:
        print("La lista de posiciones es: ", pos_list)
        print("La lista de indices es: ", idx_list)

    # Itero en frecuencia

    space_points = len(pos_list)  # Real seran 7
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

        for j in range(space_points):
            # Guarda el resultado en los puntos del espacio elegidos
            if receptor_cell_idx != -1:
                solucion_compleja = p_solution.eval(np.array(pos_list[j]), [idx_list[j]])
                magnitude_matriz[j, i] = np.abs(solucion_compleja)
            else:
                if msh.comm.size > 1:
                    _found_locally = 1 if receptor_cell_idx != -1 else 0 
                    found_globally = msh.comm.allreduce(_found_locally, op=MPI.SUM)
                    if found_globally == 0 and msh.comm.rank == 0 : 
                        print(f"ADVERTENCIA: Punto receptor {pos_list} no encontrado en ninguna celda en ningún proceso.")
                elif msh.comm.rank == 0 : # Serial y no encontrado
                    print(f"ADVERTENCIA: Punto receptor {pos_list} no encontrado en ninguna celda.")

    distancia = np.linalg.norm(np.array(rec_loc))  # Asumiendo fuente en origen

   
    R_f = respuesta_esfera_pulsante_velocidad(frequency, radio_esfera_calculado, distancia)
    # Dividís cada respuesta espectral por la magnitud ideal de la fuente
    # Usás broadcasting para aplicar a cada fila (posición)
    magnitude_matriz_corr = magnitude_matriz / (R_f[np.newaxis, :] + 1e-20)

    # Luego pasás a dB
    return 20 * np.log10(magnitude_matriz_corr + 1e-12)



def _solve_for_degree(msh, facet_tags, pos_list, idx_list, frequency, degree):
    """
    Función auxiliar interna que resuelve el problema FD-FEM para un grado
    y un rango de frecuencias específicos. Devuelve la magnitud de presión cruda.
    """
    omega = 2 * np.pi * frequency
    c0 = 343.0; rho0 = 1.225; U_normal_sphere = 1.0; sphere_facet_marker = 7 # U=1 para que la compensación sea directa
    
    V = fem.functionspace(msh, ("Lagrange", degree))
    p_trial, v_test = TrialFunction(V), TestFunction(V)
    ds_sphere = Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=sphere_facet_marker)
    c_type = PETSc.ScalarType

    space_points = len(pos_list)
    magnitude_matriz_parcial = np.zeros((space_points, len(omega)))
    
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType("preonly"); pc = ksp.getPC(); pc.setType("lu"); pc.setFactorSolverType("mumps")

    for i in range(len(omega)):
        k_wave = omega[i] / c0
        neumann_value = -1j * omega[i] * rho0 * U_normal_sphere
        neumann_term = fem.Constant(msh, c_type(neumann_value))
        
        a_form_ufl = inner(grad(p_trial), grad(v_test)) * dx - k_wave**2 * inner(p_trial, v_test) * dx
        L_form_ufl = -inner(neumann_term, v_test) * ds_sphere
        
        A_form = fem.form(a_form_ufl, dtype=PETSc.ScalarType)
        b_form = fem.form(L_form_ufl, dtype=PETSc.ScalarType)
        A = assemble_matrix(A_form); A.assemble()
        b = assemble_vector(b_form)
        
        ksp.setOperators(A)
        x_petsc = A.createVecLeft()
        ksp.solve(b, x_petsc)
        
        p_solution = fem.Function(V, dtype=PETSc.ScalarType)
        p_solution.x.array[:] = x_petsc.getArray()
        
        for j in range(space_points):
            solucion_compleja = p_solution.eval(np.array(pos_list[j]), [idx_list[j]])
            magnitude_matriz_parcial[j, i] = np.abs(solucion_compleja)
            
    return magnitude_matriz_parcial

# --- FUNCIÓN PRINCIPAL ADAPTATIVA Y CON COMPENSACIÓN ---

def FEM_Source_Solver_Adaptive(frequency, mesh_filename, rec_loc, adaptive=True, verbose=False):
    """
    Función principal que dirige la solución FD-FEM y aplica compensación.
    Si adaptive=True, usa un esquema de dos pasadas con diferente orden.
    Si adaptive=False, usa un único orden (grado 2) para todo el rango.
    """
    # --- 1. Preparativos Comunes ---
    print(f"--- Iniciando solver. Modo adaptativo: {adaptive} ---")
    try:
        msh, cell_tags, facet_tags = io.gmshio.read_from_msh(mesh_filename, MPI.COMM_WORLD, rank=0, gdim=3)
        msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    except Exception as e:
        print(f"ERROR al cargar la malla: {e}"); traceback.print_exc(); exit()

    # Cálculo automático del radio de la esfera
    one = fem.Constant(msh, PETSc.ScalarType(1.0))
    ds_sphere = Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=7)
    superficie_esfera = fem.assemble_scalar(fem.form(one * ds_sphere))
    radio_esfera_calculado = np.sqrt(superficie_esfera / (4 * np.pi))
    print(f"Radio de la esfera fuente detectado: {radio_esfera_calculado:.4f} m")

    tree = geometry.bb_tree(msh, msh.topology.dim)
    pos_list = from_position_to_grid(rec_loc, 0.1)
    idx_list = []
    for pos in pos_list:
        # ... (código para localizar receptores) ...
        receiver_point_eval = np.array([pos])
        candidate_cells_adj = geometry.compute_collisions_points(tree, receiver_point_eval.astype(np.float64))
        if candidate_cells_adj.num_nodes > 0: 
            colliding_cells_adj = geometry.compute_colliding_cells(msh, candidate_cells_adj, receiver_point_eval.astype(np.float64))
            if colliding_cells_adj.num_nodes > 0 and len(colliding_cells_adj.links(0)) > 0:
                idx_list.append(colliding_cells_adj.links(0)[0])

    magnitude_matriz_total = np.zeros((len(pos_list), len(frequency)))
    
    # --- 2. Lógica Adaptativa o Fija ---
    if not adaptive:
        print("Ejecutando con orden fijo (grado 2) para todo el rango...")
        resultado = _solve_for_degree(msh, facet_tags, pos_list, idx_list, frequency, degree=2)
        magnitude_matriz_total = resultado
    else:
        f_crossover = np.median(frequency)
        print(f"Frecuencia de cambio de orden: {f_crossover:.1f} Hz")
        
        mask_bajas = frequency <= f_crossover
        freqs_bajas = frequency[mask_bajas]; freqs_altas = frequency[~mask_bajas]
        
        if len(freqs_bajas) > 0:
            print(f"Ejecutando {len(freqs_bajas)} frecuencias con grado 1...")
            resultado_bajas = _solve_for_degree(msh, facet_tags, pos_list, idx_list, freqs_bajas, degree=1)
            magnitude_matriz_total[:, mask_bajas] = resultado_bajas

        if len(freqs_altas) > 0:
            print(f"Ejecutando {len(freqs_altas)} frecuencias con grado 2...")
            resultado_altas = _solve_for_degree(msh, facet_tags, pos_list, idx_list, freqs_altas, degree=2)
            magnitude_matriz_total[:, ~mask_bajas] = resultado_altas

    # --- 3. Compensación y Retorno ---
    print("\n--- Aplicando compensación por esfera pulsante ---")
    distancia = np.linalg.norm(np.array(rec_loc))
    R_f = respuesta_esfera_pulsante_velocidad(frequency, radio_esfera_calculado, distancia)
    
    # El broadcasting de NumPy se encarga de dividir cada fila de la matriz por el vector R_f
    magnitude_matriz_corr = magnitude_matriz_total / (R_f[np.newaxis, :] + 1e-20)
    
    print("--- Simulación Finalizada ---")
    return 20 * np.log10(magnitude_matriz_corr + 1e-12)






'''
def FEM_Source_Solver_Average(frequency, mesh_filename, rec_loc, verbose=False, degree = 1):
    """Solver the Hemholtz equation for given room geometry

    Args:
        frequency (np.array): Array of frequencies to evaluate
        mesh_filename (str): Path to the geometry to evaluate
        rec_loc (tuple[float]): Tuple with (X, Y, Z) location of the receptor. It generates 6 nearest points

    Returns:
        np.array: Magnitude of the pressure
    """

    # Parámetros físicos y de la simulación
    omega = 2 * np.pi * frequency # Frecuencia angular (rad/s)
    c0 = 343.0          # Velocidad del sonido en el aire (m/s)
    rho0 = 1.225        # Densidad del aire (kg/m^3)

    # Amplitud de la velocidad normal en la superficie de la esfera interna (m/s)
    U_normal_sphere = 0.01 # Ejemplo: 1 cm/s

    # Con esto sabe donde esta la esfera
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

    V = fem.functionspace(msh, ("Lagrange", degree))

    # Definición del problema variacional (UFL)
    p_trial = TrialFunction(V)
    v_test = TestFunction(V)

    # Definir medida para la esfera
    ds_sphere = Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=sphere_facet_marker)

    # Se calcula el área de la superficie de la fuente integrando 1 sobre sus facetas.
    # Para ello, se define una constante 1.0 sobre la malla para la integración.
    one = fem.Constant(msh, PETSc.ScalarType(1.0))
    superficie_esfera = fem.assemble_scalar(fem.form(one * ds_sphere))
    
    # Se despeja el radio de la fórmula del área de una esfera: A = 4*pi*r^2  => r = sqrt(A / (4*pi))
    radio_esfera_calculado = np.sqrt(superficie_esfera / (4 * np.pi))
    
    print(f"Radio de la esfera fuente detectado automáticamente: {radio_esfera_calculado:.4f} m")


    c_type = PETSc.ScalarType
    if not np.issubdtype(c_type, np.complexfloating) and MPI.COMM_WORLD.rank == 0:
        print("ADVERTENCIA: PETSc ScalarType no es complejo. Los resultados pueden ser incorrectos.")
    
    # Defino los puntos a evaluar en el espacio
    tree = geometry.bb_tree(msh, msh.topology.dim) 
    space_spatial_grid = 0.1  # Espaciado para el promedidado espacial del SV
    pos_list = from_position_to_grid(rec_loc, space_spatial_grid)
    idx_list = []
    for pos in pos_list:
        receiver_point_eval = np.array([pos]) # Para eval, necesita forma (1,3)

        candidate_cells_adj = geometry.compute_collisions_points(tree, receiver_point_eval.astype(np.float64))
        receptor_cell_idx = -1
        if candidate_cells_adj.num_nodes > 0: 
            colliding_cells_adj = geometry.compute_colliding_cells(msh, candidate_cells_adj, receiver_point_eval.astype(np.float64))
            if colliding_cells_adj.num_nodes > 0: 
                cells_found_for_point_0 = colliding_cells_adj.links(0) 
                if len(cells_found_for_point_0) > 0:
                    receptor_cell_idx = cells_found_for_point_0[0] 
                    idx_list.append(receptor_cell_idx)
    
    assert len(pos_list) == len(idx_list), "Un elemento de la grilla de puntos espaciales no se le pudo asignar malla"

    if verbose:
        print("La lista de posiciones es: ", pos_list)
        print("La lista de indices es: ", idx_list)

    # Itero en frecuencia

    space_points = len(pos_list)  # Real seran 7
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

        for j in range(space_points):
            # Guarda el resultado en los puntos del espacio elegidos
            if receptor_cell_idx != -1:
                solucion_compleja = p_solution.eval(np.array(pos_list[j]), [idx_list[j]])
                magnitude_matriz[j, i] = np.abs(solucion_compleja)
            else:
                if msh.comm.size > 1:
                    _found_locally = 1 if receptor_cell_idx != -1 else 0 
                    found_globally = msh.comm.allreduce(_found_locally, op=MPI.SUM)
                    if found_globally == 0 and msh.comm.rank == 0 : 
                        print(f"ADVERTENCIA: Punto receptor {pos_list} no encontrado en ninguna celda en ningún proceso.")
                elif msh.comm.rank == 0 : # Serial y no encontrado
                    print(f"ADVERTENCIA: Punto receptor {pos_list} no encontrado en ninguna celda.")

    distancia = np.linalg.norm(np.array(rec_loc))  # Asumiendo fuente en origen

   
    R_f = respuesta_esfera_pulsante_velocidad(frequency, radio_esfera_calculado, distancia)
    # Dividís cada respuesta espectral por la magnitud ideal de la fuente
    # Usás broadcasting para aplicar a cada fila (posición)
    magnitude_matriz_corr = magnitude_matriz / (R_f[np.newaxis, :] + 1e-20)

    # Luego pasás a dB
    return 20 * np.log10(magnitude_matriz_corr + 1e-12)
'''