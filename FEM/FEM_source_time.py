# --- Asegúrate de tener estas importaciones al principio de tu archivo ---
import numpy as np  
from mpi4py import MPI
from petsc4py import PETSc
from scipy.optimize import brentq
from scipy.signal import windows
from numpy import fft


from scipy.integrate import cumulative_trapezoid
# --- Asegúrate de tener estas importaciones al principio de tu archivo ---
import numpy as np
import scipy.fft as fft
from scipy.signal import windows
from scipy.integrate import cumulative_trapezoid
from mpi4py import MPI
from petsc4py import PETSc

# --- Importaciones de DOLFINx ---
import dolfinx
from dolfinx import cpp
from dolfinx import fem, io, mesh, geometry
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from ufl import (Measure, TrialFunction, TestFunction, grad, inner, dx, ds)


def optimizar_fuente_para_velocidad(f_min, f_max, amplitude=1.0, delay_factor=6.0):
    # ... (Esta función ya era correcta y no necesita cambios) ...
    if f_min <= 0 or f_max <= f_min:
        raise ValueError("Las frecuencias deben ser positivas y f_max > f_min.")
    def get_db_drop_velocidad(f, sigma):
        if sigma < 1e-12: return -np.inf
        fp = 1.0 / (2 * np.pi * sigma)
        R = f / fp
        if R < 1e-9: return -np.inf
        magnitude_ratio = R * np.exp(0.5 - (R**2 / 2.0))
        if magnitude_ratio < 1e-100: return -2000.0
        return 20 * np.log10(magnitude_ratio)
    def error_function(sigma):
        drop_at_fmin = get_db_drop_velocidad(f_min, sigma)
        drop_at_fmax = get_db_drop_velocidad(f_max, sigma)
        return drop_at_fmin - drop_at_fmax
    try:
        fp_alto = f_max * 5; sigma_min = 1.0 / (2 * np.pi * fp_alto)
        fp_bajo = f_min / 5; sigma_max = 1.0 / (2 * np.pi * fp_bajo)
        sigma_optimo = brentq(error_function, sigma_min, sigma_max)
    except ValueError:
        raise RuntimeError("No se pudo encontrar un sigma que equilibre las atenuaciones.")
    A = amplitude; sigma = sigma_optimo; delay = delay_factor * sigma
    fp_resultante_velocidad = 1.0 / (2 * np.pi * sigma)
    db_drop_resultante = get_db_drop_velocidad(f_max, sigma)
    return A, sigma, delay, fp_resultante_velocidad, db_drop_resultante


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

def respuesta_esfera_pulsante(freqs, radio_esfera, distancia_receptor, rho0=1.21, c=343.0):
    """
    Calcula la magnitud de la función de transferencia de radiación |R(ω)|
    para una esfera pulsante en campo libre, donde p(ω) = R(ω) · v_s(ω).
    
    Args:
        freqs (np.ndarray): Array de frecuencias en Hz.
        radio_esfera (float): Radio 'a' de la esfera en metros.
        distancia_receptor (float): Distancia 'r' del centro de la esfera al receptor en metros.
        rho0 (float): Densidad del medio.
        c (float): Velocidad del sonido.

    Returns:
        np.ndarray: Magnitud de la función de transferencia |R(ω)|.
    """
    # Evitar división por cero en f=0
    freqs[freqs == 0] = 1e-12 
    
    omega = 2 * np.pi * freqs
    k = omega / c
    ka = k * radio_esfera

    # Fórmula correcta de la magnitud de la función de transferencia de radiación
    magnitud_R = (omega * rho0 * radio_esfera**2) / (distancia_receptor * np.sqrt(1 + ka**2))
    
    return magnitud_R

def respuesta_esfera_pulsante_velocidad(freqs, radio_esfera, distancia_receptor, rho0=1.21, c=343.0):
    """
    Calcula la magnitud de la función de transferencia de presión para una esfera pulsante con velocidad constante.
    
    Retorna |R(ω)| tal que: p(ω) = R(ω) · v_s(ω)
    """
    omega = 2 * np.pi * freqs
    k = omega / c
    ka = k * radio_esfera
    kr = k * distancia_receptor

    numerador = np.abs(np.exp(1j * kr) - np.exp(-1j * ka))
    denominador = np.abs(1 - 1j * ka)

    respuesta = rho0 * (radio_esfera / distancia_receptor) * (numerador / denominador)
    return respuesta
def FEM_time_grid_v2(path_mesh, receptor_pos, f_min, f_max, target_courant=1.0, deconvolve=True, process_fft=True, save_data=False):
    """
    Calcula la respuesta al impulso y opcionalmente guarda un fragmento de la
    simulación 3D para visualización en ParaView.
    El paso de tiempo se determina a partir de un número de Courant objetivo.
    """
    # --- 1. PARÁMETROS FUNDAMENTALES ---
    c0 = 343.0
    rho0 = 1.21
    sphere_facet_marker = 7

    # --- 2. CARGA DE MALLA Y CÁLCULO DE h_min ---
    print(f"\n--- Cargando malla desde: {path_mesh} ---")
    msh, cell_tags, facet_tags = io.gmshio.read_from_msh(path_mesh, MPI.COMM_WORLD, rank=0, gdim=3)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    print("--- Calculando tamaño mínimo del elemento (h_min) de la malla ---")
    tdim = msh.topology.dim
    entities = np.arange(msh.topology.index_map(tdim).size_local, dtype=np.int32)
    h_values = cpp.mesh.h(msh._cpp_object, tdim, entities)
    h_min_local = np.min(h_values)
    h_min = msh.comm.allreduce(h_min_local, op=MPI.MIN)
    if msh.comm.rank == 0:
        print(f"h_min global detectado: {h_min:.6f} m")

    # --- 3. DETERMINACIÓN DE PARÁMETROS TEMPORALES ---
    print(f"\n--- Determinando paso de tiempo para C={target_courant} ---")
    dt = target_courant * h_min / c0
    fs_effective = 1.0 / dt
    nyquist_oversampling_factor = fs_effective / f_max
    if msh.comm.rank == 0:
        print(f"Frecuencia de muestreo efectiva: {fs_effective:.1f} Hz (Factor de sobremuestreo: {nyquist_oversampling_factor:.1f}x f_max)")
        if nyquist_oversampling_factor < 5.0:
            print("\n!!! ADVERTENCIA: El paso de tiempo resultante es grande para la f_max especificada.")
            print(f"    El factor de sobremuestreo ({nyquist_oversampling_factor:.1f}x) es bajo, puede haber pérdida de fidelidad.")

    T_final = (1 / f_min) * 20
    num_steps = int(np.ceil(T_final / dt))
    if msh.comm.rank == 0:
        print(f"Parámetros de simulación: {num_steps} pasos, T_final={T_final:.3f}s, dt={dt:.2e}s)")

    # --- 4. OPTIMIZACIÓN DE LA FUENTE ---
    print("\n--- Optimizando fuente para espectro de presión plano ---")
    source_amplitude, source_width, source_delay, fp, db_drop = optimizar_fuente_para_velocidad(f_min, f_max, amplitude=1e6, delay_factor=6.0)
    print(f"Fuente diseñada: Caída de {db_drop:.1f} dB en los extremos. Pico de velocidad en {fp:.1f} Hz.")

    # --- 5. DEFINICIÓN DEL PROBLEMA FEM ---
    print("\n--- Definiendo el problema de Elementos Finitos ---")
    degree = 2
    V = fem.functionspace(msh, ("Lagrange", degree))
    V_out = fem.functionspace(msh, ("Lagrange", 1)) # Para la salida
    p_out = fem.Function(V_out, name="PresionAcustica")

    p_new, p_now, p_old = fem.Function(V), fem.Function(V), fem.Function(V)
    u, v_test = TrialFunction(V), TestFunction(V)
    ds_sphere = Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=sphere_facet_marker)

    def source_acceleration(t_eval):
        t_shifted = t_eval - source_delay
        term_squared = (t_shifted / source_width)**2
        return source_amplitude * (1.0 - term_squared) * np.exp(-term_squared / 2.0)

    source_accel_const = fem.Constant(msh, PETSc.ScalarType(0.0))
    dt_ = fem.Constant(msh, PETSc.ScalarType(dt))
    c0_ = fem.Constant(msh, PETSc.ScalarType(c0))
    rho0_ = fem.Constant(msh, PETSc.ScalarType(rho0))
    beta = 0.25
    a_form_ufl = (inner(u, v_test) * dx + beta * c0_**2 * dt_**2 * inner(grad(u), grad(v_test)) * dx)
    L_form_ufl = (inner(2 * p_now - p_old, v_test) * dx -
                  c0_**2 * dt_**2 * inner(grad((1 - 2 * beta) * p_now + beta * p_old), grad(v_test)) * dx -
                  c0_**2 * dt_**2 * rho0_ * inner(source_accel_const, v_test) * ds_sphere)

    # --- 6. PREPARACIÓN DEL SOLVER, RECEPTORES Y SALIDA ---
    print("\n--- Preparando Simulación Implícita y Receptores ---")
    a_form_compiled = fem.form(a_form_ufl)
    A = assemble_matrix(a_form_compiled); A.assemble()
    L_form_compiled = fem.form(L_form_ufl)
    b = create_vector(L_form_compiled)
    solver = PETSc.KSP().create(msh.comm); solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY); solver.getPC().setType(PETSc.PC.Type.LU)
    
    # === NUEVO: Medición programática del radio de la esfera ===
    print("--- Midiendo radio de la esfera fuente desde la malla ---")
    facet_dim = msh.topology.dim - 1
    sphere_facets = facet_tags.indices[facet_tags.values == sphere_facet_marker]
    sphere_dofs = fem.locate_dofs_topological(V, facet_dim, sphere_facets)
    sphere_nodes_coords = V.tabulate_dof_coordinates()[sphere_dofs]
    
    # Calcular centro y radio (compatible con MPI)
    center_local = np.sum(sphere_nodes_coords, axis=0)
    num_nodes_local = len(sphere_nodes_coords)
    global_center_sum = msh.comm.allreduce(center_local, op=MPI.SUM)
    global_num_nodes = msh.comm.allreduce(num_nodes_local, op=MPI.SUM)
    sphere_center = global_center_sum / global_num_nodes
    
    distances = np.linalg.norm(sphere_nodes_coords - sphere_center, axis=1)
    radius_local_sum = np.sum(distances)
    global_radius_sum = msh.comm.allreduce(radius_local_sum, op=MPI.SUM)
    radio_esfera_medido = global_radius_sum / global_num_nodes
    if msh.comm.rank == 0:
        print(f"Radio medido: {radio_esfera_medido:.6f} m, Centro: {sphere_center}")
    # === FIN DE LA MEDICIÓN ===

    tree = geometry.bb_tree(msh, msh.topology.dim)
    receptor_pos_list = from_position_to_grid(receptor_pos, 0.1)
    # ... (resto del setup de receptores) ...
    receptor_cell_idx_list = []
    for pos in receptor_pos_list:
        point_to_eval = np.array([pos])
        candidate_cells = geometry.compute_collisions_points(tree, point_to_eval)
        colliding_cells = geometry.compute_colliding_cells(msh, candidate_cells, point_to_eval)
        if colliding_cells.num_nodes > 0:
            receptor_cell_idx_list.append(colliding_cells.links(0)[0])

    if save_data and msh.comm.rank == 0:
        print("--- Preparando archivo de salida 'campo_presion_3d.xdmf' para ParaView ---")
        xdmf_file = io.XDMFFile(msh.comm, "campo_presion_3d.xdmf", "w")
        xdmf_file.write_mesh(msh)
        pulse_start_time = source_delay - 4 * source_width
        save_end_time = pulse_start_time + 0.1
        print(f"--- Se guardarán los datos 3D entre t={pulse_start_time:.3f}s y t={save_end_time:.3f}s ---")

    num_points = len(receptor_pos_list)
    pressure_at_receiver_matrix = np.zeros((num_points, num_steps + 1))
    source_signal_full = np.zeros(num_steps + 1)

    # --- 7. BUCLE TEMPORAL ---
    print("\n--- Iniciando Bucle Temporal ---")
    # ... (el bucle temporal es correcto y no necesita cambios) ...
    t = 0.0
    for n in range(num_steps):
        t += dt
        source_accel_const.value = source_acceleration(t)
        
        with b.localForm() as loc_b: loc_b.zeroEntries()
        assemble_vector(b, L_form_compiled)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        solver.solve(b, p_new.x.petsc_vec)
        p_new.x.scatter_forward()
        p_old.x.array[:], p_now.x.array[:] = p_now.x.array, p_new.x.array
        
        if msh.comm.rank == 0:
            source_signal_full[n + 1] = source_accel_const.value
            for i in range(num_points):
                point = np.array([receptor_pos_list[i]])
                cell_idx = receptor_cell_idx_list[i]
                pressure_at_receiver_matrix[i, n + 1] = p_now.eval(point, [cell_idx])[0]

            if save_data and (t >= pulse_start_time and t <= save_end_time):
                p_out.interpolate(p_now)
                xdmf_file.write_function(p_out, t)
        
        if (n + 1) % (num_steps // 10) == 0: print(f"  Progreso: {int(100 * (n + 1) / num_steps)}%")
    print("--- Bucle Temporal Finalizado ---")

    if save_data and msh.comm.rank == 0:
        xdmf_file.close()
        print("--- Archivo 'campo_presion_3d.xdmf' cerrado. ---")

    # --- 8. POST-PROCESAMIENTO ---
    if process_fft:
        if msh.comm.rank == 0:
            print("\n--- Simulación Finalizada. Procesando resultados... ---")
            source_velocity_full = cumulative_trapezoid(source_signal_full, dx=dt, initial=0)
            S_f_velocity = fft.rfft(source_velocity_full)
            freqs = fft.rfftfreq(len(source_signal_full), dt)
            f_response_matrix  = np.zeros((num_points, len(freqs)))

            # Usar el centro medido para calcular la distancia al receptor
            distancia_receptor_real = np.linalg.norm(receptor_pos_list[0] - sphere_center)
            
            # Calcular la respuesta de radiación usando el radio medido
            R_f_magnitude = respuesta_esfera_pulsante(freqs, radio_esfera_medido, distancia_receptor_real)

            if deconvolve:
                for i in range(num_points):
                    P_f = fft.rfft(pressure_at_receiver_matrix[i,:])
                    epsilon = (1e-9 * np.max(np.abs(S_f_velocity)))**2
                    
                    H_combinada_mag = np.abs(P_f * np.conj(S_f_velocity)) / (np.abs(S_f_velocity)**2 + epsilon)
                    H_room_mag = H_combinada_mag / (R_f_magnitude + 1e-20)
                    
                    f_response_matrix[i,:] = 20 * np.log10(H_room_mag + 1e-12)
            else:
                for i in range(num_points):
                    P_f = fft.rfft(pressure_at_receiver_matrix[i,:])
                    f_response_matrix[i,:] = 20 * np.log10(np.abs(P_f) + 1e-12)

            return f_response_matrix, freqs
        else:
            return None, None
    else:
        return pressure_at_receiver_matrix, None




def FEM_time_grid(path_mesh, receptor_pos, f_min, f_max, target_courant=1.0, deconvolve=True, process_fft=True, save_data=False):
    """
    Calcula la respuesta al impulso y opcionalmente guarda un fragmento de la 
    simulación 3D para visualización en ParaView.
    El paso de tiempo se determina a partir de un número de Courant objetivo.
    """
    # --- 1. Parámetros Fundamentales ---
    c0 = 343.0
    rho0 = 1.21

    # --- 2. Carga de malla y cálculo de h_min ---
    print(f"\n--- Cargando malla desde: {path_mesh} ---")
    msh, cell_tags, facet_tags = io.gmshio.read_from_msh(path_mesh, MPI.COMM_WORLD, rank=0, gdim=3)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    
    print("--- Calculando tamaño mínimo del elemento (h_min) de la malla ---")
    tdim = msh.topology.dim
    entities = np.arange(msh.topology.index_map(tdim).size_local, dtype=np.int32)
    
    # Calcula el diámetro de la celda para todas las celdas en el proceso local
    # USANDO LA RUTA CORRECTA: dolfinx.cpp.mesh.h
    h_values = cpp.mesh.h(msh._cpp_object, tdim, entities)
    h_min_local = np.min(h_values)
    h_min = msh.comm.allreduce(h_min_local, op=MPI.MIN)
    
    if msh.comm.rank == 0:
        print(f"h_min global detectado: {h_min:.6f} m")

    # --- 3. Determinación de Parámetros Temporales ---
    print(f"\n--- Determinando paso de tiempo para C={target_courant} ---")
    
    # Calcular dt a partir del número de Courant deseado
    dt = target_courant * h_min / c0
    fs_effective = 1.0 / dt
    
    # Chequeo de seguridad: Verificar que cumplimos un criterio Nyquist holgado
    nyquist_oversampling_factor = fs_effective / f_max
    if msh.comm.rank == 0:
        print(f"Frecuencia de muestreo efectiva: {fs_effective:.1f} Hz (Factor de sobremuestreo: {nyquist_oversampling_factor:.1f}x f_max)")
        # Se recomienda un factor de al menos 5x para buena fidelidad, 2x es el mínimo teórico.
        if nyquist_oversampling_factor < 5.0:
            print("\n!!! ADVERTENCIA: El paso de tiempo resultante es grande para la f_max especificada.")
            print(f"    El factor de sobremuestreo ({nyquist_oversampling_factor:.1f}x) es bajo, puede haber pérdida de fidelidad.")
            print("    Considere usar un 'target_courant' más bajo o una malla más gruesa para esta f_max.")

    # El tiempo final sigue dependiendo de f_min para capturar ciclos completos de la onda más larga
    T_final = (1 / f_min) * 20 # Aumenté A=20 como discutimos, puedes ajustarlo
    num_steps = int(np.ceil(T_final / dt))
    
    if msh.comm.rank == 0:
        print(f"Parámetros de simulación: {num_steps} pasos, T_final={T_final:.3f}s, dt={dt:.2e}s)")


    print("\n--- Optimizando fuente para espectro de VELOCIDAD plano ---")
    source_amplitude, source_width, source_delay, fp, db_drop = optimizar_fuente_para_velocidad(f_min, f_max, amplitude=1e6, delay_factor=6.0)
    print(f"Fuente diseñada: Caída de {db_drop:.1f} dB en los extremos. Pico de velocidad en {fp:.1f} Hz.")

    sphere_facet_marker = 7

    # --- 2. Carga de malla y definición del problema FEM ---
    print(f"\n--- Cargando malla desde: {path_mesh} ---")
    msh, cell_tags, facet_tags = io.gmshio.read_from_msh(path_mesh, MPI.COMM_WORLD, rank=0, gdim=3)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    
    # Espacio de funciones para el CÁLCULO (alta precisión)
    degree = 2
    V = fem.functionspace(msh, ("Lagrange", degree))
    
    # --- MODIFICACIÓN: Crear Espacio de Funciones para la SALIDA ---
    # Se crea un espacio de grado 1, compatible con la malla lineal, para guardar en el archivo.
    V_out = fem.functionspace(msh, ("Lagrange", 1))
    p_out = fem.Function(V_out)
    p_out.name = "PresionAcustica" # Nombre que aparecerá en ParaView
    
    p_new, p_now, p_old = fem.Function(V), fem.Function(V), fem.Function(V)
    u, v_test = TrialFunction(V), TestFunction(V)
    ds_sphere = Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=sphere_facet_marker)

    def source_acceleration(t_eval):
        t_shifted = t_eval - source_delay
        term_squared = (t_shifted / source_width)**2
        return source_amplitude * (1.0 - term_squared) * np.exp(-term_squared / 2.0)
    
    source_accel_const = fem.Constant(msh, PETSc.ScalarType(0.0))
    dt_ = fem.Constant(msh, PETSc.ScalarType(dt)); c0_ = fem.Constant(msh, PETSc.ScalarType(c0))
    rho0_ = fem.Constant(msh, PETSc.ScalarType(rho0)); beta = 0.25
    a_form_ufl = (inner(u, v_test) * dx + beta * c0_**2 * dt_**2 * inner(grad(u), grad(v_test)) * dx)
    L_form_ufl = (inner(2 * p_now - p_old, v_test) * dx -
                  c0_**2 * dt_**2 * inner(grad((1 - 2 * beta) * p_now + beta * p_old), grad(v_test)) * dx -
                  c0_**2 * dt_**2 * rho0_ * inner(source_accel_const, v_test) * ds_sphere)
    
    # --- 3. Preparación del Solver, Receptores y Archivo de Salida ---
    print("\n--- Preparando Simulación Implícita y Receptores ---")
    a_form_compiled = fem.form(a_form_ufl)
    A = assemble_matrix(a_form_compiled); A.assemble()
    L_form_compiled = fem.form(L_form_ufl)
    b = create_vector(L_form_compiled)
    solver = PETSc.KSP().create(msh.comm); solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY); solver.getPC().setType(PETSc.PC.Type.LU)
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    receptor_pos_list = from_position_to_grid(receptor_pos, 0.1)
    receptor_cell_idx_list = []
    for pos in receptor_pos_list:
        point_to_eval = np.array([pos])
        candidate_cells = geometry.compute_collisions_points(tree, point_to_eval)
        colliding_cells = geometry.compute_colliding_cells(msh, candidate_cells, point_to_eval)
        if colliding_cells.num_nodes > 0:
            receptor_cell_idx_list.append(colliding_cells.links(0)[0])
    
    if save_data and msh.comm.rank == 0:
        print("--- Preparando archivo de salida 'campo_presion_3d.xdmf' para ParaView ---")
        xdmf_file = io.XDMFFile(msh.comm, "campo_presion_3d.xdmf", "w")
        xdmf_file.write_mesh(msh)
        
        pulse_start_time = source_delay - 4 * source_width 
        save_end_time = pulse_start_time + 0.1
        print(f"--- Se guardarán los datos 3D entre t={pulse_start_time:.3f}s y t={save_end_time:.3f}s ---")

    num_points = len(receptor_pos_list)
    pressure_at_receiver_matrix = np.zeros((num_points, num_steps + 1))
    source_signal_full = np.zeros(num_steps + 1)
    
    # --- 4. Bucle Temporal con Guardado Condicional ---
    print("\n--- Iniciando Bucle Temporal ---")
    t = 0.0
    for n in range(num_steps):
        t += dt
        source_accel_const.value = source_acceleration(t)
        
        with b.localForm() as loc_b: loc_b.zeroEntries()
        assemble_vector(b, L_form_compiled)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        solver.solve(b, p_new.x.petsc_vec)
        p_new.x.scatter_forward()
        p_old.x.array[:], p_now.x.array[:] = p_now.x.array, p_new.x.array
        
        if msh.comm.rank == 0:
            source_signal_full[n + 1] = source_accel_const.value
            for i in range(num_points):
                point = np.array([receptor_pos_list[i]])
                cell_idx = receptor_cell_idx_list[i]
                pressure_at_receiver_matrix[i, n + 1] = p_now.eval(point, [cell_idx])[0]

            if save_data and (t >= pulse_start_time and t <= save_end_time):
                p_out.interpolate(p_now)
                xdmf_file.write_function(p_out, t)
        
        if (n + 1) % (num_steps // 10) == 0: print(f"  Progreso: {int(100 * (n + 1) / num_steps)}%")
    print("--- Bucle Temporal Finalizado ---")

    if save_data and msh.comm.rank == 0:
        xdmf_file.close()
        print("--- Archivo 'campo_presion_3d.xdmf' cerrado. ---")

    # --- 5. Post-Procesamiento ---
    if process_fft:
        if msh.comm.rank == 0:
            print("\n--- Simulación Finalizada. Procesando resultados... ---")
            source_velocity_full = cumulative_trapezoid(source_signal_full, dx=dt, initial=0)
            S_f_velocity = fft.rfft(source_velocity_full)
            freqs = fft.rfftfreq(len(source_signal_full), dt)
            f_response_matrix  = np.zeros((num_points, len(freqs)))
            
            if deconvolve:
                for i in range(num_points):
                    P_f = fft.rfft(pressure_at_receiver_matrix[i,:])
                    epsilon = (1e-9 * np.max(np.abs(S_f_velocity)))**2
                    H_f = (P_f * np.conj(S_f_velocity)) / (np.abs(S_f_velocity)**2 + epsilon)
                    f_response_matrix[i,:] = 20 * np.log10(np.abs(H_f) + 1e-12)
            else:
                for i in range(num_points):
                    P_f = fft.rfft(pressure_at_receiver_matrix[i,:])
                    f_response_matrix[i,:] = 20 * np.log10(np.abs(P_f) + 1e-12)

            return f_response_matrix, freqs 
        else:
            return None, None
    else:
        return pressure_at_receiver_matrix, None

'''

def diseñar_pulso_velocidad_fir(num_taps, f_min, f_max, fs):
    """Diseña un pulso de velocidad FIR y devuelve el pulso y su longitud real."""
    if num_taps % 2 == 0:
        num_taps += 1
    freqs = fft.rfftfreq(num_taps, 1/fs)
    ideal_response = np.zeros_like(freqs)
    pass_band_mask = (freqs >= f_min) & (freqs <= f_max)
    ideal_response[pass_band_mask] = 1.0
    impulse_response_ideal = fft.irfft(ideal_response, n=num_taps)
    window = windows.hann(num_taps)
    fir_pulse = np.fft.fftshift(impulse_response_ideal) * window
    return fir_pulse, num_taps

def derivar_pulso(pulso, dt):
    """Calcula la derivada de un pulso de tiempo."""
    return np.gradient(pulso, dt)

def from_position_to_grid(pos, dx):
    """Crea una grilla de 7 puntos alrededor de una posición central."""
    return [
        np.array([pos[0], pos[1], pos[2]]),
        np.array([pos[0] + dx, pos[1], pos[2]]),
        np.array([pos[0] - dx, pos[1], pos[2]]),
        np.array([pos[0], pos[1] + dx, pos[2]]),
        np.array([pos[0], pos[1] - dx, pos[2]]),
        np.array([pos[0], pos[1], pos[2] + dx]),
        np.array([pos[0], pos[1], pos[2] - dx]),
    ]



def FEM_time_grid(path_mesh, receptor_pos, f_min, f_max, deconvolve=True):
    """
    Calcula la respuesta al impulso en una grilla de puntos usando una fuente FIR
    diseñada a medida. La deconvolución se realiza Presión/Velocidad.
    """
    # --- 1. Parámetros de simulación ---
    c0 = 343.0
    rho0 = 1.21
    fs = 20 * f_max  # Frecuencia de muestreo
    dt = 1 / fs
    T_final = (1 / f_min) * 20  
    num_steps = int(np.ceil(T_final / dt))
    print(f"Parámetros de simulación: {num_steps} pasos, T_final={T_final:.3f}s, dt={dt:.2e}s")

    # --- 2. Diseño de la fuente FIR ---
    print("\n--- Diseñando fuente de excitación con filtro FIR ---")
    transition_width = 4.0  # Ancho de la banda de transición en Hz
    num_taps_deseado = int((4 * fs) / transition_width)
    
    velo_t_corto, num_taps_real = diseñar_pulso_velocidad_fir(num_taps_deseado, f_min, f_max, fs)
    accel_t_corto = derivar_pulso(velo_t_corto, dt)
    print(f"Fuente FIR diseñada con {num_taps_real} puntos.")

    # --- 3. Preparación de las señales de tiempo completas (con zero-padding) ---
    # La señal de excitación para la simulación debe tener la longitud total.
    source_acceleration_full = np.zeros(num_steps + 1)
    # Se guarda también la señal de velocidad para la deconvolución posterior.
    source_velocity_full = np.zeros(num_steps + 1)
    
    # Se coloca el pulso corto en el centro de la ventana de simulación para hacerlo causal.
    offset = (num_steps + 1 - num_taps_real) // 2
    source_acceleration_full[offset : offset + num_taps_real] = accel_t_corto
    source_velocity_full[offset : offset + num_taps_real] = velo_t_corto
    
    # --- 4. Carga de malla y definición del problema FEM (sin cambios) ---
    sphere_facet_marker = 7
    print(f"\n--- Cargando malla desde: {path_mesh} ---")
    msh, cell_tags, facet_tags = io.gmshio.read_from_msh(path_mesh, MPI.COMM_WORLD, rank=0, gdim=3)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    
    degree = 2
    V = fem.functionspace(msh, ("Lagrange", degree))
    p_new, p_now, p_old = fem.Function(V), fem.Function(V), fem.Function(V)
    u, v_test = TrialFunction(V), TestFunction(V)
    ds_sphere = Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=sphere_facet_marker)

    source_accel_const = fem.Constant(msh, PETSc.ScalarType(0.0))
    dt_ = fem.Constant(msh, PETSc.ScalarType(dt))
    c0_ = fem.Constant(msh, PETSc.ScalarType(c0))
    rho0_ = fem.Constant(msh, PETSc.ScalarType(rho0))
    beta = 0.25
    a_form_ufl = (inner(u, v_test) * dx + beta * c0_**2 * dt_**2 * inner(grad(u), grad(v_test)) * dx)
    L_form_ufl = (inner(2 * p_now - p_old, v_test) * dx -
                  c0_**2 * dt_**2 * inner(grad((1 - 2 * beta) * p_now + beta * p_old), grad(v_test)) * dx -
                  c0_**2 * dt_**2 * rho0_ * inner(source_accel_const, v_test) * ds_sphere)
    
    # --- 5. Preparación del Solver y Receptores (sin cambios) ---
    # ... (código para ensamblar A, b, KSP y localizar receptores) ...
    print("\n--- Preparando Simulación Implícita y Receptores ---")
    a_form_compiled = fem.form(a_form_ufl)
    A = assemble_matrix(a_form_compiled); A.assemble()
    L_form_compiled = fem.form(L_form_ufl)
    b = create_vector(L_form_compiled)
    solver = PETSc.KSP().create(msh.comm); solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY); solver.getPC().setType(PETSc.PC.Type.LU)
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    receptor_pos_list = from_position_to_grid(receptor_pos, 0.1)
    receptor_cell_idx_list = []
    for pos in receptor_pos_list:
        # ... (código para encontrar celdas) ...
        point_to_eval = np.array([pos])
        candidate_cells = geometry.compute_collisions_points(tree, point_to_eval)
        colliding_cells = geometry.compute_colliding_cells(msh, candidate_cells, point_to_eval)
        if colliding_cells.num_nodes > 0:
            receptor_cell_idx_list.append(colliding_cells.links(0)[0])
    
    num_points = len(receptor_pos_list)
    pressure_at_receiver_matrix = np.zeros((num_points, num_steps + 1), dtype=np.float64)

    # --- 6. Bucle Temporal ---
    print("\n--- Iniciando Bucle Temporal ---")
    for n in range(num_steps):
        # La fuente ahora se toma del array pre-calculado
        source_accel_const.value = source_acceleration_full[n]
        
        with b.localForm() as loc_b: loc_b.zeroEntries()
        assemble_vector(b, L_form_compiled)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        solver.solve(b, p_new.x.petsc_vec)
        p_new.x.scatter_forward()
        p_old.x.array[:], p_now.x.array[:] = p_now.x.array, p_new.x.array
        
        if msh.comm.rank == 0:
            for i in range(num_points):
                point = np.array([receptor_pos_list[i]])
                cell_idx = receptor_cell_idx_list[i]
                pressure_at_receiver_matrix[i, n] = p_now.eval(point, [cell_idx])[0]

        if (n + 1) % (num_steps // 10) == 0: print(f"  Progreso: {int(100 * (n + 1) / num_steps)}%")
    print("--- Bucle Temporal Finalizado ---")

    # --- 7. Post-Procesamiento con Deconvolución Correcta --- 
    if msh.comm.rank == 0:
        print("\n--- Simulación Finalizada. Procesando resultados... ---")

        # a) FFT de la VELOCIDAD de la fuente (la entrada física correcta)
        S_f_velocity = fft.rfft(source_velocity_full)
        # b) Eje de frecuencias
        freqs = fft.rfftfreq(len(source_velocity_full), dt)
        
        # c) Preparar la matriz de resultados y deconvolucionar
        f_response_matrix  = np.zeros((num_points, len(freqs)))
        if deconvolve:
            for i in range(num_points):
                P_f = fft.rfft(pressure_at_receiver_matrix[i, :])
                epsilon = (1e-12 * np.max(np.abs(S_f_velocity)))**2
                H_f = (P_f * np.conj(S_f_velocity)) / (np.abs(S_f_velocity)**2 + epsilon)
                f_response_matrix[i, :] = 20 * np.log10(np.abs(H_f) + 1e-12)
        else:
            for i in range(num_points):
                P_f = fft.rfft(pressure_at_receiver_matrix[i, :])
                f_response_matrix[i, :] = 20 * np.log10(np.abs(P_f) + 1e-12)

        return f_response_matrix, freqs 
    else:
        return None, None

def FEM_time_optimal_gaussian_impulse(path_mesh, receptor_pos, f_min, f_max, h_min, use_spatial_averaging=True ):
    """
    Calcula la respuesta al impulso usando el método implícito de Newmark-beta.
    Permite opcionalmente promediar espacialmente la medición en el receptor.
    """
    # --- 1. Parámetros físicos y de la simulación ---
    c0 = 343.0
    rho0 = 1.21

    # Lógica del paso de tiempo basada en precisión (no en CFL)
    fs_accuracy = 20 * f_max
    dt = 1 / fs_accuracy
    T_final = (1 / f_min) * 20
    num_steps = int(np.ceil(T_final / dt))
    print(f"Método Implícito (Newmark-beta). Pasos: {num_steps} (T_final={T_final:.3f}s, dt={dt:.2e}s)")

    source_amplitude, source_width, source_delay, fp, db_drop = ricker_wavlet_parameters(f_min, f_max, amplitude=1e6, delay_factor=6.0)
    print(f"Caída de {db_drop:.1f} dB en los extremos. Pico en {fp:.1f} Hz.")

    sphere_facet_marker = 7

    # --- 2. Cargar malla ---
    print(f"\n--- Cargando malla desde: {path_mesh} ---")
    msh, cell_tags, facet_tags = io.gmshio.read_from_msh(
        path_mesh, MPI.COMM_WORLD, rank=0, gdim=3
    )
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    # --- 3. Espacio de funciones ---
    degree = 2
    V = fem.functionspace(msh, ("Lagrange", degree))

    p_new, p_now, p_old = fem.Function(V), fem.Function(V), fem.Function(V)
    p_now.x.array[:], p_old.x.array[:] = 0.0, 0.0
    
    u, v_test = TrialFunction(V), TestFunction(V)
    ds_sphere = Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=sphere_facet_marker)

    # --- 4. Definición de la fuente temporal ---
    source_accel_const = fem.Constant(msh, PETSc.ScalarType(0.0))
    def source_acceleration(t_eval):
        t_shifted = t_eval - source_delay
        term = (t_shifted / source_width)**2
        return source_amplitude * (1.0 - 2.0 * term) * np.exp(-term)

    # --- 5. Formulación Variacional de Newmark-beta ---
    dt_ = fem.Constant(msh, PETSc.ScalarType(dt))
    c0_ = fem.Constant(msh, PETSc.ScalarType(c0))
    rho0_ = fem.Constant(msh, PETSc.ScalarType(rho0))
    beta = 0.25

    a_form_ufl = (inner(u, v_test) * dx +
                  beta * c0_**2 * dt_**2 * inner(grad(u), grad(v_test)) * dx)
    L_form_ufl = (inner(2 * p_now - p_old, v_test) * dx -
                  c0_**2 * dt_**2 * inner(grad((1 - 2 * beta) * p_now + beta * p_old), grad(v_test)) * dx -
                  c0_**2 * dt_**2 * rho0_ * inner(source_accel_const, v_test) * ds_sphere)
    
    # --- 6. Preparación de la Simulación y Receptores ---
    print("\n--- Preparando Simulación Implícita ---")
    a_form_compiled = fem.form(a_form_ufl)
    A = assemble_matrix(a_form_compiled)
    A.assemble()

    L_form_compiled = fem.form(L_form_ufl)
    b = create_vector(L_form_compiled)

    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    if use_spatial_averaging:
        # --- SI SE USA PROMEDIACIÓN ---
        print("\n--- Preparando grilla de receptores para promediación espacial ---")
        tree = geometry.bb_tree(msh, msh.topology.dim)
        space_spatial_grid = 0.1 
        receptor_pos_list = from_position_to_grid(receptor_pos, space_spatial_grid)
        receptor_cell_idx_list = []
        points_found = 0
        for pos in receptor_pos_list:
            point_to_eval = np.array([pos])
            candidate_cells = geometry.compute_collisions_points(tree, point_to_eval)
            colliding_cells = geometry.compute_colliding_cells(msh, candidate_cells, point_to_eval)
            if colliding_cells.num_nodes > 0:
                receptor_cell_idx = colliding_cells.links(0)[0]
                receptor_cell_idx_list.append(receptor_cell_idx)
                points_found += 1
        assert len(receptor_pos_list) == len(receptor_cell_idx_list), \
            f"Error: {len(receptor_pos_list) - points_found} puntos de la grilla están fuera de la malla."
        print(f"Grilla de {len(receptor_pos_list)} receptores localizada exitosamente.")
    else:
        # --- SI NO SE USA PROMEDIACIÓN (UN SOLO PUNTO) ---
        print("\n--- Preparando receptor en un único punto (sin promediación) ---")
        tree = geometry.bb_tree(msh, msh.topology.dim)
        receptor_point_eval = np.array([receptor_pos])
        candidate_cells = geometry.compute_collisions_points(tree, receptor_point_eval)
        colliding_cells = geometry.compute_colliding_cells(msh, candidate_cells, receptor_point_eval)
        receptor_cell_idx = colliding_cells.links(0)[0] if colliding_cells.num_nodes > 0 else -1
        if receptor_cell_idx == -1:
            print("ADVERTENCIA: El punto receptor no fue encontrado en la malla.")
        else:
            print("Receptor único localizado exitosamente.")

    pressure_at_receiver_global = np.zeros(num_steps + 1, dtype=np.float64)
    source_signal_full = np.zeros(num_steps + 1, dtype=np.float64)

    # --- 7. Bucle Temporal ---
    print("\n--- Iniciando Bucle Temporal ---")
    t = 0.0
    for n in range(num_steps):
        t += dt
        source_accel_const.value = source_acceleration(t)

        with b.localForm() as loc_b: loc_b.zeroEntries()
        assemble_vector(b, L_form_compiled)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        solver.solve(b, p_new.x.petsc_vec)
        p_new.x.scatter_forward()

        p_old.x.array[:], p_now.x.array[:] = p_now.x.array, p_new.x.array

        if msh.comm.rank == 0:
            source_signal_full[n + 1] = source_accel_const.value
            
            if use_spatial_averaging:
                # --- LÓGICA DE PROMEDIACIÓN ---
                pressure_sum = 0.0
                num_points = len(receptor_pos_list)
                if num_points > 0:
                    for i in range(num_points):
                        point = np.array([receptor_pos_list[i]])
                        cell_idx = receptor_cell_idx_list[i]
                        pressure_sum += p_now.eval(point, [cell_idx])[0]
                    pressure_at_receiver_global[n + 1] = pressure_sum / num_points
            else:
                # --- LÓGICA DE PUNTO ÚNICO ---
                if receptor_cell_idx != -1:
                    pressure_at_receiver_global[n + 1] = p_now.eval(receptor_point_eval, [receptor_cell_idx])[0]

        if (n + 1) % (num_steps // 10) == 0: print(f"  Progreso: {int(100 * (n + 1) / num_steps)}%")
    print("--- Bucle Temporal Finalizado ---")

    # --- 8. Post-proceso con Deconvolución ---
    if msh.comm.rank == 0:
        print("\n--- Realizando Post-Proceso con Deconvolución ---")

        P_f = np.fft.fft(pressure_at_receiver_global)  
        S_f = np.fft.fft(source_signal_full)
        freqs = np.fft.fftfreq(len(pressure_at_receiver_global), dt)

        S_f_conj = np.conj(S_f)
        S_f_mag_sq = np.abs(S_f)**2
        epsilon = (1e-8 * np.max(np.abs(S_f)))**2
        H_f = (P_f * S_f_conj) / (S_f_mag_sq + epsilon)

        mask = (freqs >= 20.0) & (freqs <= 200.0)
        f_plot = freqs[mask]
        
        mag_H = np.abs(H_f[mask])
        mag_S = np.abs(S_f[mask])
        mag_P = np.abs(P_f[mask])

        print("--- Post-Proceso Finalizado ---")
        return np.real(f_plot), mag_H, mag_S, mag_P
    else:
        return None, None, None, None

def FEM_time_grid(path_mesh, receptor_pos, f_min, f_max, deconvolve=True):
    """
    Calcula la respuesta al impulso en una grilla de puntos alrededor del receptor
    y devuelve una matriz con las series de tiempo de presión para cada punto.
    """
    # --- 1. Parámetros físicos y de la simulación ---
    c0 = 343.0
    rho0 = 1.21

    # Lógica del paso de tiempo basada en precisión (no en CFL)
    fs_accuracy = 20 * f_max
    dt = 1 / fs_accuracy
    T_final = (1 / f_min) * 20  
    num_steps = int(np.ceil(T_final / dt))
    print(f"Método Implícito (Newmark-beta). Pasos: {num_steps} (T_final={T_final:.3f}s, dt={dt:.2e}s)")

    source_amplitude, source_width, source_delay, fp, db_drop = ricker_wavlet_parameters(f_min, f_max, amplitude=1e6, delay_factor=6.0)
    print(f"Caída de {db_drop:.1f} dB en los extremos. Pico en {fp:.1f} Hz.")

    sphere_facet_marker = 7

    # --- 2. Cargar malla ---
    print(f"\n--- Cargando malla desde: {path_mesh} ---")
    msh, cell_tags, facet_tags = io.gmshio.read_from_msh(
        path_mesh, MPI.COMM_WORLD, rank=0, gdim=3
    )
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    # --- 3. Espacio de funciones ---
    degree = 2
    V = fem.functionspace(msh, ("Lagrange", degree))

    p_new, p_now, p_old = fem.Function(V), fem.Function(V), fem.Function(V)
    p_now.x.array[:], p_old.x.array[:] = 0.0, 0.0
    
    u, v_test = TrialFunction(V), TestFunction(V)
    ds_sphere = Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=sphere_facet_marker)

    # --- 4. Definición de la fuente temporal ---
    source_accel_const = fem.Constant(msh, PETSc.ScalarType(0.0))
    def source_acceleration(t_eval):
        t_shifted = t_eval - source_delay
        term = (t_shifted / source_width)**2
        return source_amplitude * (1.0 - 2.0 * term) * np.exp(-term)

    # --- 5. Formulación Variacional de Newmark-beta ---
    dt_ = fem.Constant(msh, PETSc.ScalarType(dt))
    c0_ = fem.Constant(msh, PETSc.ScalarType(c0))
    rho0_ = fem.Constant(msh, PETSc.ScalarType(rho0))
    beta = 0.25

    a_form_ufl = (inner(u, v_test) * dx +
                  beta * c0_**2 * dt_**2 * inner(grad(u), grad(v_test)) * dx)
    L_form_ufl = (inner(2 * p_now - p_old, v_test) * dx -
                  c0_**2 * dt_**2 * inner(grad((1 - 2 * beta) * p_now + beta * p_old), grad(v_test)) * dx -
                  c0_**2 * dt_**2 * rho0_ * inner(source_accel_const, v_test) * ds_sphere)
    
    # --- 6. Preparación de la Simulación y Receptores ---
    print("\n--- Preparando Simulación Implícita ---")
    a_form_compiled = fem.form(a_form_ufl)
    A = assemble_matrix(a_form_compiled)
    A.assemble()

    L_form_compiled = fem.form(L_form_ufl)
    b = create_vector(L_form_compiled)

    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    print("\n--- Preparando grilla de receptores ---")
    tree = geometry.bb_tree(msh, msh.topology.dim)
    space_spatial_grid = 0.1 
    receptor_pos_list = from_position_to_grid(receptor_pos, space_spatial_grid)
    receptor_cell_idx_list = []
    for pos in receptor_pos_list:
        point_to_eval = np.array([pos])
        candidate_cells = geometry.compute_collisions_points(tree, point_to_eval)
        colliding_cells = geometry.compute_colliding_cells(msh, candidate_cells, point_to_eval)
        if colliding_cells.num_nodes > 0:
            receptor_cell_idx_list.append(colliding_cells.links(0)[0])
    assert len(receptor_pos_list) == len(receptor_cell_idx_list), \
        "Error: Uno o más puntos de la grilla están fuera de la malla."
    print(f"Grilla de {len(receptor_pos_list)} receptores localizada exitosamente.")
    
    # Se crea una matriz para guardar los resultados: (puntos, tiempo)
    num_points = len(receptor_pos_list)
    pressure_at_receiver_matrix = np.zeros((num_points, num_steps + 1), dtype=np.float64)
    source_signal_full = np.zeros(num_steps + 1, dtype=np.float64)

    # --- 7. Bucle Temporal ---
    print("\n--- Iniciando Bucle Temporal ---")
    t = 0.0
    for n in range(num_steps):
        t += dt
        source_accel_const.value = source_acceleration(t)

        with b.localForm() as loc_b: loc_b.zeroEntries()
        assemble_vector(b, L_form_compiled)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        solver.solve(b, p_new.x.petsc_vec)
        p_new.x.scatter_forward()

        p_old.x.array[:], p_now.x.array[:] = p_now.x.array, p_new.x.array

        if msh.comm.rank == 0:
            source_signal_full[n + 1] = source_accel_const.value
            
            # Evaluar la presión en cada punto de la grilla y guardarla en la matriz
            for i in range(num_points):
                point = np.array([receptor_pos_list[i]])
                cell_idx = receptor_cell_idx_list[i]
                pressure_at_receiver_matrix[i, n + 1] = p_now.eval(point, [cell_idx])[0]

        if (n + 1) % (num_steps // 10) == 0: print(f"  Progreso: {int(100 * (n + 1) / num_steps)}%")
    print("--- Bucle Temporal Finalizado ---")

    # --- 8. Pospro Opcional --- 
    source_f_responese = fft.rfft(source_signal_full)
    freqs = fft.rfftfreq(len(source_signal_full), dt)
    f_response_matrix  = np.zeros((num_points, len(freqs),)) 


    if deconvolve == True:
        for i in range(num_points):
            rir = pressure_at_receiver_matrix[i,:]
            rta_f = fft.rfft(rir)
            epsilon = (1e-9 * np.max(np.abs(source_f_responese)))**2
            rta_f = (rta_f * np.conj(source_f_responese)) / (np.abs(source_f_responese)**2 + epsilon)
            rta_f = 20 * np.log10(np.abs(rta_f))
            f_response_matrix[i,:] = rta_f
             

    if msh.comm.rank == 0:
        print("\n--- Simulación Finalizada. Devolviendo resultados ---")
        return f_response_matrix, freqs 
    else:
        return None, None
'''