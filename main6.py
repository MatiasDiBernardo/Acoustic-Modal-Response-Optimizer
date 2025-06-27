# Evaluacion del TD-Grid
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
import scipy.fft 

# --- 1. IMPORTACIONES DE FUNCIONES LOCALES ---
# Asegúrate de que la ruta de importación sea correcta según tu estructura de carpetas.
from FEM.FEM_source_time import FEM_time_grid
# from mesh.mesh_3D_generator import generate_mesh_parallelepiped

# --- 2. DEFINICIÓN DE PARÁMETROS GEOMÉTRICOS Y DE SIMULACIÓN ---
Lx, Ly, Lz = 2.0, 4.0, 3.0
L = [(Lx, Ly)]
source_position = (0.7, 1.3, 0.8)
receptor_pos = (1.501, 2.998, 1.203)
path_mesh = "mallado/room.msh"

# Parámetros de frecuencia para la simulación
f_min = 20.0
f_max = 200.0

# --- 3. GENERACIÓN DE MALLA (si es necesario) ---
# print("--- Generando/verificando malla ---")
# from mesh.mesh_3D_generator import generate_mesh_parallelepiped
# h_min_global = generate_mesh_parallelepiped([Lx, Ly], Lz, source_position, f_max)
# print(f"h_min usado para la malla: {h_min_global}")

# --- 4. EJECUCIÓN DE LA SIMULACIÓN ---
print("\n--- Iniciando simulación con FEM_time_grid ---")
f_response_matrix, freqs = FEM_time_grid(
    path_mesh=path_mesh,
    receptor_pos=receptor_pos,
    f_min=f_min,
    f_max=f_max,
    # El argumento h_minm no se usaba, se puede quitar si tu función no lo necesita
    # h_minm=0.1, 
    deconvolve=False
)

# --- 5. VISUALIZACIÓN DE RESULTADOS (MODIFICADO) ---
if f_response_matrix is not None and freqs is not None:
    print("\n--- Graficando resultado para la posición central (Punto 0) ---")
    
    # --- MODIFICACIÓN: Seleccionar solo la primera fila (posición cero) ---
    # La matriz tiene forma (7, N_freqs), seleccionamos la primera curva.
    mag_db = f_response_matrix[0, :]

    # Preparar el plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)

    # --- MODIFICACIÓN: Graficar una sola curva ---
    ax.plot(freqs, mag_db, lw=2, label='Respuesta en Receptor Central')
        
    # --- MODIFICACIÓN: Cálculo de límites Y simplificado para una curva ---
    plot_freq_range_mask = (freqs >= f_min) & (freqs <= f_max)
    visible_db_values = mag_db[plot_freq_range_mask]
    
    if visible_db_values.size > 0:
        min_db = np.min(visible_db_values)
        max_db = np.max(visible_db_values)
        # Se añade un padding o margen del 5% del rango dinámico visible
        padding = (max_db - min_db) * 0.05 + 2 # +2 para un margen mínimo
        ax.set_ylim(min_db - padding, max_db + padding)

    # Configuración final del gráfico
    ax.set_title(f'Respuesta en Frecuencia del Receptor Central (FEM TD-Grid)\nRecinto: {Lx}x{Ly}x{Lz} m')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Magnitud (dB)')
    ax.set_xlim(f_min, f_max)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

else:
    print("La simulación no devolvió resultados (probablemente por ejecución en paralelo).")