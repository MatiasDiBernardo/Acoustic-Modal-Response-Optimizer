# --- Asegúrate de tener estas importaciones al principio de tu archivo ---
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from scipy.optimize import brentq

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

def FEM_time_optimal_gaussian_impulse(path_mesh, receptor_pos, f_min, f_max):
    """
    Calcula la respuesta al impulso usando el método implícito de Newmark-beta,
    que es incondicionalmente estable y no introduce amortiguación numérica.
    """
    # --- 1. Parámetros físicos y de la simulación ---
    c0 = 343.0
    rho0 = 1.21
    x_r, y_r, z_r = receptor_pos[0], receptor_pos[1], receptor_pos[2]

    # Lógica del paso de tiempo basada en precisión (no en CFL)
    fs_accuracy = 12 * f_max
    dt = 1 / fs_accuracy
    T_final = (1 / f_min) * 12
    num_steps = int(np.ceil(T_final / dt))
    print(f"Método Implícito (Newmark-beta). Pasos: {num_steps} (T_final={T_final:.3f}s, dt={dt:.2e}s)")

    source_amplitude, source_width, source_delay, fp, db_drop = ricker_wavlet_parameters(f_min, f_max, amplitude=1e6, delay_factor=6.0)
    print(f"Caída de {db_drop:.1f} dB en los extremos. Pico en {fp:.1f} Hz.")

    sphere_facet_marker = 7
    receiver_point_eval = np.array([[x_r, y_r, z_r]])

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
        # Ondícula de Ricker (2da derivada de una Gaussiana)
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
    
    # --- 6. Preparación de la Simulación ---
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

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidate_cells = geometry.compute_collisions_points(tree, receiver_point_eval)
    colliding_cells = geometry.compute_colliding_cells(msh, candidate_cells, receiver_point_eval)
    receptor_cell_idx = colliding_cells.links(0)[0] if colliding_cells.num_nodes > 0 else -1

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
            if receptor_cell_idx != -1:
                pressure_at_receiver_global[n + 1] = p_now.eval(receiver_point_eval, [receptor_cell_idx])[0]

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