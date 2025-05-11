# -*- coding: utf-8 -*-
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, mesh, cpp, plot, geometry 
from dolfinx.fem.petsc import LinearProblem, assemble_vector, assemble_matrix, apply_lifting, set_bc, create_matrix, create_vector
from dolfinx.io import VTKFile
from ufl import (TestFunction, TrialFunction, dx, ds, grad, inner,
                 Measure, as_vector, FacetNormal)
import os
import traceback 
# *** DESCOMENTADO: Importar matplotlib.pyplot para graficar ***
import matplotlib.pyplot as plt

# --- 0. Definiciones Geométricas (para consistencia con Gmsh y receptor) ---
Lx = 2.0
Ly = 1.5
Lz = 1.0
# Posición de la fuente (debe coincidir con el script de Gmsh)
x_esfera_src = 0.3 
y_esfera_src = 0.3
z_esfera_src = 0.3


# --- 1. Parámetros físicos y de la simulación ---
c0 = 343.0          # Velocidad del sonido (m/s)
rho0 = 1.225        # Densidad del aire (kg/m^3)

# Parámetros de la fuente impulsiva
source_amplitude = 0.1  
source_width = 5e-4     # Ancho del pulso Gaussiano (s)
source_delay = 6 * source_width # Retardo del pulso (0.003s)

# Parámetros de la simulación temporal
f_min_deseada = 50.0    # Hz, frecuencia mínima a resolver bien en FFT
num_ciclos_min = 10     # Número de ciclos de f_min a capturar
T_final = num_ciclos_min / f_min_deseada # Tiempo final de simulación (s) -> 0.2s

# dt definido por f_max_resolucion_temporal
f_max_resolucion_temporal = 2000.0 # Hz, frecuencia máxima a resolver bien por el esquema temporal
puntos_por_periodo_dt = 20.0       # Puntos por período de f_max_resolucion_temporal
dt = 1.0 / (puntos_por_periodo_dt * f_max_resolucion_temporal) # -> 2.5e-5 s
num_steps = int(T_final / dt)
print(f"Número de pasos de tiempo: {num_steps} (T_final={T_final:.3f}s, dt={dt:.2e}s)")


# Marcadores de faceta
sphere_facet_marker = 7
# Paredes rígidas (Neumann homogéneo implícito)

# Punto receptor
receiver_coords = np.array([Lx/2, Ly/2, Lz/2])
receiver_point_eval = np.array([receiver_coords]) # Para eval, necesita forma (1,3)

print("--- Parámetros de Simulación (TD-FEM para Respuesta al Impulso) ---")
print(f"Dimensiones de la sala: Lx={Lx}, Ly={Ly}, Lz={Lz}")
print(f"Velocidad del sonido: {c0:.1f} m/s")
print(f"Densidad del aire: {rho0:.3f} kg/m^3")
print(f"Tiempo final: {T_final:.4f} s") 
print(f"Paso de tiempo dt: {dt:.2e} s (basado en f_max_resolucion_temporal={f_max_resolucion_temporal} Hz)")
print(f"Retardo de la fuente (source_delay): {source_delay:.4f} s")
print(f"Marcador para la superficie de la esfera (fuente): {sphere_facet_marker}")
print(f"Punto receptor: {receiver_coords}")


# --- 2. Cargar malla ---
# *** Nombre de archivo de malla para f_max 400Hz ***
mesh_filename = "mallado/sala_impulso_fmax400_v2.msh" 
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
degree = 1 # Usamos P1 para eficiencia
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
output_vtk_dir = "tdfem_ir_analisis_vtk_fmax400_dt_adjusted" 
if msh.comm.rank == 0:
    if not os.path.exists(output_vtk_dir):
        os.makedirs(output_vtk_dir)
vtk_filename = os.path.join(output_vtk_dir, "pressure_evolution.pvd")
vtk_file = VTKFile(msh.comm, vtk_filename, "w")
p_now.name = "pressure" 
# Frecuencia de guardado VTK ajustada
save_every_n_steps = num_steps // 100 if num_steps >= 100 else 1 # Guardar ~100 fotogramas
save_every_n_steps = max(1, save_every_n_steps) # Asegurar al menos 1
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

# --- 8. Información y Ploteo para Respuesta al Impulso (Solo en proceso 0) ---
if msh.comm.rank == 0:
    print("\n--- Procesando Respuesta al Impulso del Receptor ---")
    if pressure_at_receiver_global is not None:
        
        pressure_signal_to_plot = np.nan_to_num(pressure_at_receiver_global) # Reemplazar NaNs con 0 para ploteo
        
        print(f"  Estadísticas de la señal temporal del receptor (después de nan_to_num):")
        print(f"    Min: {np.min(pressure_signal_to_plot):.3e}")
        print(f"    Max: {np.max(pressure_signal_to_plot):.3e}")
        print(f"    Media: {np.mean(pressure_signal_to_plot):.3e}")
        max_abs_val_signal = np.max(np.abs(pressure_signal_to_plot))
        print(f"    Max Abs: {max_abs_val_signal:.3e}")
        
        # Guardar los datos en un archivo de texto
        output_data_filename = os.path.join(output_vtk_dir, "receptor_signal_tdfem.txt")
        try:
            np.savetxt(output_data_filename, np.vstack((times, pressure_signal_to_plot)).T, 
                       header="Time(s) Pressure(Pa)", fmt="%.6e")
            print(f"Datos del receptor guardados en: {output_data_filename}")
        except Exception as e_save:
            print(f"ERROR guardando datos del receptor: {e_save}")

        # Graficar la respuesta temporal del receptor
        try:
            plt.figure(figsize=(10, 4))
            plt.plot(times, pressure_signal_to_plot)
            plt.title(f"Respuesta Temporal en el Receptor ({receiver_coords})")
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Presión (Pa)")
            plt.grid(True)
            # Ajustar límites del eje X para ver el inicio y el pulso
            plot_time_end_view = min(T_final, source_delay + 20 * source_width) 
            plot_time_end_view = max(plot_time_end_view, 0.01) # Mostrar al menos 10ms
            plt.xlim(0, plot_time_end_view)
            
            # Ajustar límites del eje Y si la señal no es plana
            if max_abs_val_signal > 1e-9: # Solo si hay señal significativa
                plt.ylim(-max_abs_val_signal * 1.2, max_abs_val_signal * 1.2)
            
            plot_filename_temp = os.path.join(output_vtk_dir, "receptor_IR_temporal.png")
            plt.savefig(plot_filename_temp)
            print(f"Gráfico de respuesta temporal guardado en: {plot_filename_temp}")
            # plt.show() # Descomentar para mostrar interactivamente
            plt.close() # Cerrar la figura para liberar memoria
        except Exception as e_plot:
            print(f"ERROR al generar el gráfico temporal: {e_plot}")
            traceback.print_exc()
            
    else:
        print("No hay datos del receptor para graficar (probablemente no es rank 0 o el punto no se encontró).")
    
    sampling_rate = 1.0 / dt
    print(f"\nTasa de muestreo (Sample Rate) para la señal del receptor: {sampling_rate:.1f} Hz (1/dt)")
    print("Puedes usar el archivo 'receptor_signal_tdfem.txt' para análisis y ploteo externo.")

print(f"\nResultados VTK del campo completo guardados en la carpeta '{output_vtk_dir}'.")
print("\nScript finalizado.")
