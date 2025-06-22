# Evaluacion del TD-Grid
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
import scipy.fft

# 1. IMPORTACIONES DE FUNCIONES LOCALES
# Asegúrate de que la ruta de importación sea correcta según tu estructura de carpetas
from FEM.FEM_source_time import FEM_time_grid
# from FEM.mode_sumation import compute_modal_transfer
from mesh.mesh_3D_generator import generate_mesh_parallelepiped
# from Source_image import source_image_cuboid

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
print("--- Generando/verificando malla ---")
h_min_global = generate_mesh_parallelepiped([Lx, Ly], Lz, source_position, f_max)
print(f"h_min usado para la malla: {h_min_global}")

# --- 4. EJECUCIÓN DE LA SIMULACIÓN TD-GRID ---
print("\n--- Iniciando simulación con FEM_time_grid ---")
pressure_matrix, source_signal = FEM_time_grid(
    path_mesh=path_mesh,
    receptor_pos=receptor_pos,
    f_min=f_min,
    f_max=f_max,
    h_min=0.1
)

# --- 5. POST-PROCESAMIENTO Y VISUALIZACIÓN ---
if pressure_matrix is not None and source_signal is not None:
    print("\n--- Procesando resultados para graficar ---")

    # Re-calcular parámetros de tiempo para el eje de frecuencia
    num_samples = pressure_matrix.shape[1]
    fs_accuracy = 20 * f_max
    dt = 1 / fs_accuracy
    
    # Calcular el eje de frecuencias completo
    freqs = scipy.fft.fftfreq(num_samples, dt)
    
    # Calcular la FFT de la señal de la fuente una sola vez
    S_f = scipy.fft.fft(source_signal)
    
    # Máscara para seleccionar solo las frecuencias positivas
    positive_freq_mask = freqs >= 0
    
    # Preparar el plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- MODIFICACIÓN: Lista para almacenar los valores de dB en el rango visible ---
    visible_db_values = []
    
    # Máscara para el rango de frecuencias que se va a graficar (f_min a f_max)
    plot_freq_range_mask = (freqs >= f_min) & (freqs <= f_max)

    # Iterar sobre cada respuesta al impulso en la matriz
    for i in range(pressure_matrix.shape[0]):
        rir_t = pressure_matrix[i, :]
        P_f = scipy.fft.fft(rir_t)
        
        # Deconvolución para obtener la Función de Transferencia H(f)
        epsilon = (1e-8 * np.max(np.abs(S_f)))**2
        H_f = (P_f * np.conj(S_f)) / (np.abs(S_f)**2 + epsilon)
        
        # Convertir a decibelios (dB)
        mag_db = 20 * np.log10(np.abs(H_f))
        
        # Graficar usando solo las frecuencias positivas
        ax.plot(freqs[positive_freq_mask], 
                mag_db[positive_freq_mask], 
                lw=1.5, 
                alpha=0.8, 
                label=f'Punto {i+1}')
        
        # --- MODIFICACIÓN: Guardar los valores de dB del rango visible ---
        # Se asegura de tomar solo los valores válidos (no -inf)
        valid_db_in_range = mag_db[plot_freq_range_mask]
        if valid_db_in_range.size > 0:
            visible_db_values.append(valid_db_in_range)

    # --- MODIFICACIÓN: Cálculo y aplicación de los límites Y dinámicos ---
    if visible_db_values:
        # Concatenar todos los valores de dB en un solo array
        all_visible_db = np.concatenate(visible_db_values)
        
        # Encontrar el mínimo y máximo global
        min_db = np.min(all_visible_db)
        max_db = np.max(all_visible_db)
        
        # Añadir un padding (margen) para que el gráfico no se vea ajustado
        padding = 5  # 5 dB de margen arriba y abajo
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