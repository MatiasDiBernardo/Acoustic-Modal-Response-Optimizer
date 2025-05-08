# -*- coding: utf-8 -*-
# !!! NOTA IMPORTANTE SOBRE PETSC Y NÚMEROS COMPLEJOS !!!
# Esta simulación requiere PETSc compilado con soporte para escalares complejos.
# ------------------------------------------------------------------------------------

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, mesh, cpp
from dolfinx.fem.petsc import LinearProblem, assemble_vector, apply_lifting, set_bc, assemble_matrix # Import assemble_matrix
from dolfinx.io import VTKFile
from ufl import (TestFunction, TrialFunction, dx, ds, grad, inner,
                 Measure)
import os
import traceback # Import traceback for better error reporting

# --- 1. Parámetros físicos y de la simulación ---
frequency = 222.0  # Hz (Frecuencia baja)
omega = 2 * np.pi * frequency # Frecuencia angular (rad/s)
c0 = 343.0          # Velocidad del sonido en el aire (m/s)
rho0 = 1.225        # Densidad del aire (kg/m^3)
k_wave = omega / c0      # Número de onda (rad/m)

# Amplitud de la velocidad normal en la superficie de la esfera interna (m/s)
U_normal_sphere = 0.01 # Ejemplo: 1 cm/s

# Marcadores de faceta
sphere_facet_marker = 7
wall_markers_robin = [1, 2, 3, 4, 5, 6] # Tags para paredes con Robin BC

print("--- Parámetros de Simulación ---")
print(f"Frecuencia: {frequency:.1f} Hz")
print(f"Velocidad del sonido: {c0:.1f} m/s")
print(f"Densidad del aire: {rho0:.3f} kg/m^3")
print(f"Número de onda k: {k_wave:.2f} rad/m (Longitud de onda: {c0/frequency:.2f} m)")
print(f"Velocidad normal en la esfera: {U_normal_sphere:.2e} m/s")
print(f"Marcador para la superficie de la esfera (fuente): {sphere_facet_marker}")
print(f"Marcadores para paredes con Robin BC: {wall_markers_robin}")


# --- 2. Cargar malla ---
# *** Usamos la malla del dominio pequeño ***
mesh_filename = "mallado/esfera_en_paralelepipedo_268Hz_mode222.msh"
print(f"\n--- Cargando malla desde: {mesh_filename} ---")
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
    print(f"Tags de faceta únicos encontrados en la malla: {unique_tags_found}")
    # Verificamos todos los tags usados
    required_tags = set(wall_markers_robin + [sphere_facet_marker])
    if not required_tags.issubset(unique_tags_found):
         print(f"ERROR CRÍTICO: Faltan tags de faceta requeridos. Encontrados: {unique_tags_found}, Requeridos: {required_tags}")
         exit()
    print(f"Todos los marcadores de faceta requeridos encontrados en la malla.")

    # Crear conectividad necesaria
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim) # Facets to cells
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1) # Cells to facets
    
    print(f"Malla cargada: {msh.topology.dim}D, {msh.topology.index_map(msh.topology.dim).size_local} celdas locales (proceso {MPI.COMM_WORLD.rank})")

except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo de malla '{mesh_filename}'.")
    print("Asegúrate de generar la malla con el script de Gmsh para dominio reducido.")
    exit()
except Exception as e:
     print(f"ERROR: Ocurrió un problema al cargar la malla: {e}")
     traceback.print_exc()
     exit()

# --- 3. Espacio de funciones ---
# Usamos Grado 1 para reducir coste computacional
degree = 1
V = fem.functionspace(msh, ("Lagrange", degree))
print(f"\n--- Espacio de Funciones (V) ---")
print(f"Tipo: Lagrange, Grado: {degree}")
print(f"Número de grados de libertad locales: {V.dofmap.index_map.size_local * V.dofmap.index_map_bs}")
print(f"Número de grados de libertad globales: {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")


# --- 4. Definición del problema variacional (UFL) ---
p_trial = TrialFunction(V)
v_test = TestFunction(V)

# Definir medida para la esfera
ds_sphere = Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=sphere_facet_marker)
# Definir una medida general 'ds' que usa los facet_tags
ds = Measure("ds", domain=msh, subdomain_data=facet_tags)

c_type = PETSc.ScalarType
if not np.issubdtype(c_type, np.complexfloating) and MPI.COMM_WORLD.rank == 0:
    print("ADVERTENCIA: PETSc ScalarType no es complejo. Los resultados pueden ser incorrectos.")
print(f"Usando PETSc ScalarType: {c_type}")

# Condición de Neumann en la esfera: dp/dn = g_N
neumann_value = -1j * omega * rho0 * U_normal_sphere
neumann_term = fem.Constant(msh, c_type(neumann_value)) # Esto es g_N
print(f"Valor del término de Neumann (g_N) en la esfera: {neumann_value:.2e}")

# Condición de Robin en las paredes exteriores: dp/dn + gamma*p = 0
alpha_robin = 0.1
gamma = (alpha_robin + 1j) * k_wave
gamma_const = fem.Constant(msh, c_type(gamma))
print(f"Valor del coeficiente Robin (gamma = (alpha+j)k con alpha={alpha_robin}) en paredes {wall_markers_robin}: {gamma:.2e}")


# Forma bilineal (lado izquierdo de la ecuación débil)
a_form_ufl = inner(grad(p_trial), grad(v_test)) * dx \
             - k_wave**2 * inner(p_trial, v_test) * dx
# Añadir término de Robin sumando sobre las paredes designadas
for marker in wall_markers_robin:
    a_form_ufl += inner(gamma_const * p_trial, v_test) * ds(marker)

# Forma lineal (lado derecho de la ecuación débil)
# Término de contorno es - Integral(dp/dn * v) ds = - Integral(g_N * v) ds_sphere
L_form_ufl = -inner(neumann_term, v_test) * ds_sphere # SIGNO CORREGIDO

bcs = [] # Sin condiciones de Dirichlet


# --- 5. Compilar formas y resolver ---
print("\n--- Resolución del Problema Lineal ---")
print("Compilando formas variacionales...")
try:
    a_form_compiled = fem.form(a_form_ufl, dtype=c_type)
    L_form_compiled = fem.form(L_form_ufl, dtype=c_type)

    p_solution = fem.Function(V, dtype=c_type)
    p_solution.name = "pressure_complex_P1" # Actualizado a P1
    print(f"Función solución p_solution creada con dtype: {p_solution.x.array.dtype}")

    # *** Usamos LinearProblem con solver directo LU/MUMPS ***
    print("Creando el problema lineal...")
    problem = LinearProblem(a_form_compiled, L_form_compiled, bcs=bcs, u=p_solution,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", 
                                           "pc_factor_mat_solver_type": "mumps"})
    
    print("Resolviendo el problema lineal...")
    problem.solve()
    print("¡Solución completada!")

except RuntimeError as e:
    if "PETSC_DIR" in str(e) or "PETSC_ARCH" in str(e) or "MPI" in str(e):
        print("ERROR DE RUNTIME: Problema con PETSc/MPI.")
    if "UMFPACK" in str(e) or "MUMPS" in str(e):
         print("ERROR DE RUNTIME: Problema con el solver directo (UMFPACK/MUMPS).")
    print(f"Detalles del error: {e}")
    traceback.print_exc()
    exit()
except Exception as e:
    print(f"ERROR inesperado durante la compilación o solución: {e}")
    traceback.print_exc()
    exit()

# --- 6. Post-procesamiento y guardar resultados ---
print("\n--- Post-procesamiento y Guardado de Resultados ---")

output_dir = "helmholtz_pulsating_sphere_results_30Hz_small_Direct" # Nuevo directorio para esta prueba
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
MPI.COMM_WORLD.barrier()

# La solución ya está en P1, usamos directamente
p_solution_P1c_array = p_solution.x.array 

# --- Verificación de Valores de la Solución (Magnitud P1) ---
print("Extrayendo componentes de la solución P1 compleja como arrays NumPy...")
if np.isnan(p_solution_P1c_array).any():
    print("ERROR: Se encontraron valores NaN en la solución p_solution.")
    pressure_real_array = np.nan_to_num(p_solution_P1c_array.real).astype(np.float64)
    pressure_imag_array = np.nan_to_num(p_solution_P1c_array.imag).astype(np.float64)
    pressure_magnitude_array = np.abs(np.nan_to_num(p_solution_P1c_array)).astype(np.float64)
    pressure_phase_array = np.angle(np.nan_to_num(p_solution_P1c_array)).astype(np.float64)
    print("ADVERTENCIA: Se reemplazaron NaNs con 0 para el guardado VTK.")
else:
    pressure_real_array = p_solution_P1c_array.real.astype(np.float64)
    pressure_imag_array = p_solution_P1c_array.imag.astype(np.float64)
    pressure_magnitude_array = np.abs(p_solution_P1c_array).astype(np.float64)
    pressure_phase_array = np.angle(p_solution_P1c_array).astype(np.float64)

print("\n--- Verificación de Valores de la Solución (Magnitud P1) ---")
min_mag = np.min(pressure_magnitude_array)
max_mag = np.max(pressure_magnitude_array)
mean_mag = np.mean(pressure_magnitude_array)
print(f"Min magnitud presión P1: {min_mag:.4e}")
print(f"Max magnitud presión P1: {max_mag:.4e}")
print(f"Media magnitud presión P1: {mean_mag:.4e}")
if not np.isnan(min_mag) and not np.isnan(max_mag) and np.isclose(min_mag, max_mag):
    print("ADVERTENCIA: La magnitud de la presión P1 parece ser uniforme.")
elif np.isnan(min_mag) or np.isnan(max_mag):
     print("ADVERTENCIA: Se encontraron NaNs en la magnitud de la presión.")


numpy_arrays_to_write = {
    "pressure_real_P1": pressure_real_array,
    "pressure_imag_P1": pressure_imag_array,
    "pressure_magnitude_P1": pressure_magnitude_array,
    "pressure_phase_P1": pressure_phase_array
}

print(f"Guardando campos (P1) en directorio '{output_dir}'...")
# Creamos una función P1 real para guardar los componentes
V_output_P1 = fem.functionspace(msh, ("Lagrange", 1))
p_output_P1_real_component = fem.Function(V_output_P1, dtype=np.float64)

for field_name, data_array in numpy_arrays_to_write.items():
    print(f"--- Procesando campo: {field_name} ---")
    try:
        if data_array.size == p_output_P1_real_component.x.array.size:
             p_output_P1_real_component.x.array[:] = data_array
        else:
             raise ValueError(f"Incompatibilidad de tamaño para {field_name}.")

        p_output_P1_real_component.name = field_name
        vtk_filename = os.path.join(output_dir, f"acoustic_field_{field_name}.pvd")
        
        print(f"Escribiendo '{vtk_filename}'...")
        with VTKFile(msh.comm, vtk_filename, "w") as vtk:
            vtk.write_function(p_output_P1_real_component, 0.0) 
        print(f"'{field_name}' escrito correctamente.")

    except Exception as e_vtk:
         print(f"ERROR inesperado durante el guardado VTK de '{field_name}': {e_vtk}")
         traceback.print_exc()
    print(f"--- Fin procesamiento campo: {field_name} ---")

print(f"\nResultados (P1) guardados en formato VTK en la carpeta '{output_dir}'.")
print("Puedes visualizarlos con ParaView o similar.")

print("\nScript finalizado.")
