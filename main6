# Evaluacion del TD-Grid
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
import scipy.fft 

# 1. IMPORTACIONES DE FUNCIONES LOCALES
# Asegúrate de que la ruta de importación sea correcta según tu estructura de carpetas.
# Se asume que FEM_time_grid está en el archivo FEM_source_time.py
from FEM.FEM_source_time import FEM_time_grid
# from mesh.mesh_3D_generator import generate_mesh_parallelepiped

# --- 2. DEFINICIÓN DE PARÁMETROS GEOMÉTRICOS Y DE SIMULACIÓN ---
Lx, Ly, Lz = 2.0, 4.0, 2.0
L = [(Lx, Ly)]
source_position = (0.7, 1.3, 0.8)
receptor_pos = (1.501, 2.998, 1.203)
path_mesh = "mallado/room.msh"

# Parámetros de frecuencia para la simulación
f_min = 20.0
f_max = 200.0

# --- 3. GENERACIÓN DE MALLA (si es necesario) ---
# Descomentar si necesitas generar la malla cada vez.
# print("--- Generando/verificando malla ---")
# from mesh.mesh_3D_generator import generate_mesh_parallelepiped
# h_min_global = generate_mesh_parallelepiped([Lx, Ly], Lz, source_position, f_max)
# print(f"h_min usado para la malla: {h_min_global}")

# --- 4. EJECUCIÓN DE LA SIMULACIÓN ---
print("\n--- Iniciando simulación con FEM_time_grid ---")
# La función ahora devuelve la matriz de respuesta en frecuencia (en dB) y el eje de frecuencias.
f_response_matrix, freqs = FEM_time_grid(
    path_mesh=path_mesh,
    receptor_pos=receptor_pos,
    f_min=f_min,
    f_max=f_max,
    h_minm=0.1, # El nombre del parámetro era h_minm en tu función
    deconvolve=True
)

# --- 5. VISUALIZACIÓN DE RESULTADOS ---
# El post-procesamiento ya no es necesario aquí. Solo se grafica.
if f_response_matrix is not None and freqs is not None:
    print("\n--- Graficando resultados ---")
    
    # Preparar el plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Máscara para el rango de frecuencias que se va a graficar
    plot_freq_range_mask = (freqs >= f_min) & (freqs <= f_max)

    # Iterar sobre cada fila de la matriz de resultados (que ya está en dB)
    for i in range(f_response_matrix.shape[0]):
        # La fila 'i' ya contiene los valores en dB para el punto 'i'
        mag_db = f_response_matrix[i, :]
        
        # Graficar directamente. No se necesita máscara porque rfft/rfftfreq ya manejan las freqs positivas.
        ax.plot(freqs, mag_db, lw=1.5, alpha=0.8, label=f'Punto {i+1}')
        
    # --- Cálculo de límites Y dinámicos ---
    # Se extraen los valores de dB solo del rango de frecuencias visible
    visible_db_values = [f_response_matrix[i, plot_freq_range_mask] for i in range(f_response_matrix.shape[0])]
    if visible_db_values:
        all_visible_db = np.concatenate(visible_db_values)
        min_db = np.min(all_visible_db)
        max_db = np.max(all_visible_db)
        padding = 5
        ax.set_ylim(min_db - padding, max_db + padding)

    # Configuración final del gráfico
    ax.set_title(f'Respuesta en Frecuencia en la Grilla de Receptores (Deconvolucionada)\nRecinto: {Lx}x{Ly}x{Lz} m')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Magnitud (dB)')
    ax.set_xlim(f_min, f_max)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

else:
    print("La simulación no devolvió resultados (probablemente por ejecución en paralelo).")