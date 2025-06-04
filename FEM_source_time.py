import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
# Importar geometry y io.XDMFFile
from dolfinx import fem, io, mesh, cpp, plot, geometry 
from dolfinx.fem.petsc import LinearProblem, assemble_vector, assemble_matrix, apply_lifting, set_bc, create_matrix, create_vector
from dolfinx.io import VTKFile
from ufl import (TestFunction, TrialFunction, dx, ds, grad, inner,
                 Measure, as_vector, FacetNormal)
import os
import traceback 
import matplotlib.pyplot as plt  # Asegurarse que está importado

# IMPORTANTE: Este script requiere FEniCS X Scalar (no imaginario)

def FEM_time(path_mesh, receptor_pos):
    """Calcula rta en frec de 20 a 200 Hz (lineal no dB)

    Args:
        path_mesh (str): Path al mallado
        receptor_pos ([x, y, z]): Lista de floats con la posición del receptor

    Returns:
        array, array: freqs, magnitude
    """
    # Puntos receptor
    x_r = receptor_pos[0]
    y_r = receptor_pos[1]
    z_r = receptor_pos[2]

    # --- 1. Parámetros físicos y de la simulación ---
    c0 = 343.0          # Velocidad del sonido (m/s)
    rho0 = 1.225        # Densidad del aire (kg/m^3)

    # Parámetros de la fuente impulsiva
    source_amplitude = 0.1  
    source_width = 5e-4     # Ancho del pulso Gaussiano (s)
    source_delay = 6 * source_width  # Retardo del pulso (0.003s)

    # Frecuencia mínima de interés (Hz)
    f_min_deseada = 20.0    # en lugar de 50 Hz

    # Número de ciclos de f_min a capturar (puedes ajustar a 5 ciclos si quieres T_final=0.25s)
    num_ciclos_min = 25     
    T_final = num_ciclos_min / f_min_deseada  # => 10 ciclos de 20 Hz => 0.5 s

    # Frecuencia máxima de resolución temporal (Hz)
    # Antes era 2000 Hz, ahora basta con 500 Hz para cubrir hasta 200 Hz con holgura
    f_max_resolucion_temporal = 500.0  # Hz
    puntos_por_periodo_dt = 50.0       # Puntos por período de la frecuencia de Nyquist
    dt = 1.0 / (puntos_por_periodo_dt * f_max_resolucion_temporal)  # ~1e-4 s

    num_steps = int(np.ceil(T_final / dt))  # redondeamos hacia arriba
    print(f"Número de pasos de tiempo: {num_steps} (T_final={T_final:.3f}s, dt={dt:.2e}s)")

    # Marcadores de faceta
    sphere_facet_marker = 7
    # Paredes rígidas (Neumann homogéneo implícito)

    # Punto receptor
    receiver_coords = np.array([x_r, y_r, z_r])
    receiver_point_eval = np.array([receiver_coords])  # forma (1,3)

    print(f"Tiempo final: {T_final:.4f} s") 
    print(f"Paso de tiempo dt: {dt:.2e} s (basado en f_max_resolucion_temporal={f_max_resolucion_temporal} Hz)")
    print(f"Retardo de la fuente (source_delay): {source_delay:.4f} s")
    print(f"Marcador para la superficie de la esfera (fuente): {sphere_facet_marker}")
    print(f"Punto receptor: {receiver_coords}")

    # --- 2. Cargar malla ---
    mesh_filename = path_mesh 
    print(f"\n--- Cargando malla desde: {mesh_filename} ---")
    try:
        msh, cell_tags, facet_tags = io.gmshio.read_from_msh(
            mesh_filename, MPI.COMM_WORLD, rank=0, gdim=3
        )

        if facet_tags is None or facet_tags.values is None or facet_tags.values.size == 0:
            print("ERROR CRÍTICO: facet_tags no se cargó correctamente o está vacío.")
            exit()
        
        unique_tags_found = np.unique(facet_tags.values)
        print(f"Tags de faceta únicos encontrados en la malla: {unique_tags_found}")
        required_tags = {sphere_facet_marker}
        wall_tags_exist = set([1, 2, 3, 4, 5, 6]).issubset(unique_tags_found)
        if not required_tags.issubset(unique_tags_found) or not wall_tags_exist:
            print(f"ERROR CRÍTICO: Faltan tags de faceta requeridos.")
            exit()
        print(f"Marcador de la esfera (tag {sphere_facet_marker}) encontrado en la malla.")

        msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
        msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
        
        print(f"Malla cargada: {msh.topology.dim}D, {msh.topology.index_map(msh.topology.dim).size_local} celdas locales (proceso {MPI.COMM_WORLD.rank})")

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de malla '{mesh_filename}'.")
        print("Asegúrate de generar la malla con el script de Gmsh correspondiente.")
        exit()
    except Exception as e:
        print(f"ERROR: Ocurrió un problema al cargar la malla: {e}")
        traceback.print_exc()
        exit()

    # --- 3. Espacio de funciones y funciones ---
    degree = 1  # P1 para eficiencia
    V = fem.functionspace(msh, ("Lagrange", degree))
    print(f"\n--- Espacio de Funciones (V) ---")
    print(f"Tipo: Lagrange, Grado: {degree}")
    print(f"Número de grados de libertad globales: {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")

    # Funciones para la presión en diferentes pasos de tiempo
    scalar_type_real = np.float64 
    p_new = fem.Function(V, name="pressure_new", dtype=scalar_type_real) 
    p_now = fem.Function(V, name="pressure", dtype=scalar_type_real)      
    p_old = fem.Function(V, dtype=scalar_type_real)      
    print(f"Funciones p_new, p_now, p_old creadas con dtype: {p_now.x.array.dtype}")

    u = TrialFunction(V) 
    v_test = TestFunction(V)

    # Medida para la superficie de la esfera
    ds_sphere = Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=sphere_facet_marker)

    # --- 4. Definición de la fuente temporal ---
    source_accel_const = fem.Constant(msh, scalar_type_real(0.0)) 

    def source_acceleration(t_eval):
        t_shifted = t_eval - source_delay
        amplitude_factor = source_amplitude / (source_width**2) 
        accel = amplitude_factor * (-2.0 * t_shifted / source_width**2) * np.exp(-(t_shifted / source_width)**2)
        return accel

    # --- 5. Formulación Variacional Discretizada en Tiempo ---
    dt_ = fem.Constant(msh, scalar_type_real(dt))
    c0_ = fem.Constant(msh, scalar_type_real(c0))
    rho0_ = fem.Constant(msh, scalar_type_real(rho0))

    a_form_ufl = inner(u, v_test) * dx \
                + c0_**2 * dt_**2 * inner(grad(u), grad(v_test)) * dx
    L_form_ufl = inner(2.0 * p_now - p_old, v_test) * dx \
                - c0_**2 * dt_**2 * rho0_ * inner(source_accel_const, v_test) * ds_sphere

    # --- 6. Preparación de la Simulación ---
    print("\n--- Preparando Simulación Temporal ---")
    print("Compilando forma bilineal (matriz A)...")
    a_form_compiled = fem.form(a_form_ufl, dtype=scalar_type_real) 
    A = fem.petsc.assemble_matrix(a_form_compiled)
    A.assemble()
    print("Matriz A ensamblada.")
    print("Compilando forma lineal (vector b)...")
    L_form_compiled = fem.form(L_form_ufl, dtype=scalar_type_real) 
    b = fem.petsc.create_vector(L_form_compiled) 
    print("Configurando solver (LU)...")
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY) 
    solver.getPC().setType(PETSc.PC.Type.LU) 
    # solver.setFromOptions() # Comentado para evitar advertencias si no hay opciones de línea de comando

    # --- Preparar registro de datos en el receptor ---
    print("Localizando punto receptor...")
    tree = geometry.bb_tree(msh, msh.topology.dim) 
    candidate_cells_adj = geometry.compute_collisions_points(tree, receiver_point_eval.astype(np.float64))
    receptor_cell_idx = -1
    if candidate_cells_adj.num_nodes > 0: 
        colliding_cells_adj = geometry.compute_colliding_cells(msh, candidate_cells_adj, receiver_point_eval.astype(np.float64))
        if colliding_cells_adj.num_nodes > 0: 
            cells_found_for_point_0 = colliding_cells_adj.links(0) 
            if len(cells_found_for_point_0) > 0:
                receptor_cell_idx = cells_found_for_point_0[0] 

    if receptor_cell_idx != -1:
        print(f"Punto receptor encontrado en la celda local {receptor_cell_idx} en el proceso {msh.comm.rank}")
    else:
        if msh.comm.size > 1:
            _found_locally = 1 if receptor_cell_idx != -1 else 0 
            found_globally = msh.comm.allreduce(_found_locally, op=MPI.SUM)
            if found_globally == 0 and msh.comm.rank == 0 : 
                print(f"ADVERTENCIA: Punto receptor {receiver_coords} no encontrado en ninguna celda en ningún proceso.")
        elif msh.comm.rank == 0 : # Serial y no encontrado
            print(f"ADVERTENCIA: Punto receptor {receiver_coords} no encontrado en ninguna celda.")

    times = np.zeros(num_steps + 1)
    pressure_at_receiver_global = None
    if msh.comm.rank == 0:
        pressure_at_receiver_global = np.zeros(num_steps + 1, dtype=float)

    # --- Preparar archivo VTK para guardar resultados (CAMPO COMPLETO) ---
    output_vtk_dir = "tdfem_ir_analisis_vtk_fmax500_dt_adjusted" 
    if msh.comm.rank == 0:
        if not os.path.exists(output_vtk_dir):
            os.makedirs(output_vtk_dir)
    vtk_filename = os.path.join(output_vtk_dir, "pressure_evolution.pvd")
    vtk_file = VTKFile(msh.comm, vtk_filename, "w")
    p_now.name = "pressure" 
    # Frecuencia de guardado VTK ajustada
    save_every_n_steps = num_steps // 100 if num_steps >= 100 else 1  # Guardar ~100 fotogramas
    save_every_n_steps = max(1, save_every_n_steps)  # Asegurar al menos 1
    print(f"Guardando campo de presión cada {save_every_n_steps} paso(s) en '{vtk_filename}' (Total fotogramas: {num_steps // save_every_n_steps})")

    # --- 7. Bucle Temporal ---
    print("\n--- Iniciando Bucle Temporal ---")
    print(f"Se realizarán {num_steps} pasos de tiempo. Esto puede tardar.") 
    t = 0.0
    p_now.x.array[:] = 0.0
    p_old.x.array[:] = 0.0
    vtk_file.write_function(p_now, t) 
    if msh.comm.rank == 0 and receptor_cell_idx != -1: 
        try:
            pressure_at_receiver_global[0] = np.real(p_now.eval(receiver_point_eval, [receptor_cell_idx])[0])
        except Exception as e_eval_init_loop: 
            print(f"ADVERTENCIA: Falló eval inicial en receptor: {e_eval_init_loop}")
            pressure_at_receiver_global[0] = np.nan

    for n in range(num_steps):
        t += dt
        current_accel = source_acceleration(t)
        source_accel_const.value = current_accel
        
        with b.localForm() as loc_b: loc_b.zeroEntries()
        fem.petsc.assemble_vector(b, L_form_compiled)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        try:
            p_new.x.array[:] = 0.0 
            solver.solve(b, p_new.x.petsc_vec)
            p_new.x.scatter_forward() 
        except Exception as e_solve:
            print(f"ERROR: Falló la solución en el paso {n+1} (t={t:.4f}): {e_solve}"); traceback.print_exc(); break 

        p_old.x.array[:] = p_now.x.array
        p_now.x.array[:] = p_new.x.array 
        
        if msh.comm.rank == 0 and receptor_cell_idx != -1:
            try:
                pressure_val = p_now.eval(receiver_point_eval, [receptor_cell_idx])[0]
                pressure_at_receiver_global[n+1] = np.real(pressure_val)
            except Exception as e_eval:
                pressure_at_receiver_global[n+1] = np.nan 
        elif msh.comm.rank == 0: 
            pressure_at_receiver_global[n+1] = np.nan
        
        if (n + 1) % (num_steps // 20 if num_steps > 200 else 50) == 0: 
            norm_p_now_numpy = np.linalg.norm(p_now.x.array) 
            print(f"  Progreso: Paso {n+1}/{num_steps} (t = {t:.4f} s), Norma p_now: {norm_p_now_numpy:.3e}, Fuente: {current_accel:.3e}")

        if (n + 1) % save_every_n_steps == 0: 
            vtk_file.write_function(p_now, t) 

    if (num_steps) % save_every_n_steps != 0:
        vtk_file.write_function(p_now, T_final)
        print(f"  Guardado VTK final en t = {T_final:.4f} s (Paso {num_steps})")
    vtk_file.close() 
    print("--- Bucle Temporal Finalizado ---")

    # --- 8. Post-proceso: FFT y Plots ---
    tiempo = np.arange(0, (num_steps + 1)*dt, dt)

    # FFT de la señal en el receptor
    rta_en_f = np.fft.fft(pressure_at_receiver_global)
    frecuencia = np.fft.fftfreq(len(pressure_at_receiver_global), dt)

    # Shift para centrar las frecuencias y magnitudes
    shifted_frecuencia = np.fft.fftshift(frecuencia)
    shifted_magnitudes = np.abs(np.fft.fftshift(rta_en_f))

    # Filtrar solo las frecuencias positivas entre 20 y 200 Hz
    positive_indices = (shifted_frecuencia > 0)
    f_pos = shifted_frecuencia[positive_indices]
    mag_pos = shifted_magnitudes[positive_indices]

    # Índices correspondientes a 20–200 Hz
    f_min_plot = 20.0
    f_max_plot = 200.0
    mask_plot = np.logical_and(f_pos >= f_min_plot, f_pos <= f_max_plot)
    f_plot = f_pos[mask_plot]
    mag_plot = mag_pos[mask_plot]

    return f_plot, mag_plot
