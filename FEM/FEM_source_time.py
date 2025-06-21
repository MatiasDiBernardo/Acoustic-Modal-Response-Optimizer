# --- Asegúrate de tener estas importaciones al principio de tu archivo ---
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from scipy.optimize import brentq
from scipy.signal import windows
# --- Importaciones de DOLFINx ---
from dolfinx import fem, io, mesh, plot, log, geometry

from dolfinx import fem, io, mesh, geometry
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from ufl import (Measure, TrialFunction, TestFunction, grad, inner, dx, ds)

# (También necesitas la función 'ricker_wavlet_parameters' que ya tienes)
# --------------------------------------------------------------------------

def ricker_wavlet_parameters(f_min, f_max, amplitude=1.0, delay_factor=6.0):
    """
    Calcula los parámetros óptimos para una ondícula de Ricker que equilibra
    la atenuación en las frecuencias mínima y máxima de una banda.
    """
    if f_min <= 0 or f_max <= f_min:
        raise ValueError("Las frecuencias deben ser positivas y f_max > f_min.")
    
    def get_db_drop(f, sigma):
        if sigma < 1e-12: return -np.inf
        fp = np.sqrt(2) / (2 * np.pi * sigma)
        R = f / fp
        if R < 1e-9: return -np.inf
        magnitude_ratio = R**2 * np.exp(1 - R**2)
        if magnitude_ratio < 1e-100: return -2000.0
        return 20 * np.log10(magnitude_ratio)

    def error_function(sigma):
        drop_at_fmin = get_db_drop(f_min, sigma)
        drop_at_fmax = get_db_drop(f_max, sigma)
        return drop_at_fmin - drop_at_fmax
    
    try:
        fp_alto = f_max * 5
        sigma_min = np.sqrt(2) / (2 * np.pi * fp_alto)
        fp_bajo = f_min / 5
        sigma_max = np.sqrt(2) / (2 * np.pi * fp_bajo)
        sigma_optimo = brentq(error_function, sigma_min, sigma_max)
    except ValueError:
        raise RuntimeError("No se pudo encontrar un sigma que equilibre las atenuaciones.")
        
    A = amplitude
    sigma = sigma_optimo
    delay = delay_factor * sigma
    fp_resultante = np.sqrt(2) / (2 * np.pi * sigma)
    db_drop_resultante = get_db_drop(f_max, sigma)
    return A, sigma, delay, fp_resultante, db_drop_resultante



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




def FEM_time_optimal_gaussian_impulse(path_mesh, receptor_pos, f_min, f_max, h_min, use_spatial_averaging=True):
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

