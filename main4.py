import numpy as np
import matplotlib.pyplot as plt

# 1. IMPORTACIONES DE FUNCIONES LOCALES
from FEM.FEM_source_time import FEM_time_optimal_gaussian_impulse
from FEM.mode_sumation import compute_modal_transfer_complete
from mesh.mesh_3D_generator import generate_mesh_parallelepiped 

# Se asume que los archivos TD_FEM_MUMPS.py existen si se usan 
# from TD_FEM_MUMPS import FEM_time_iterativo, FEM_time_MUMPS

# --- 2. DEFINICIÓN DE PARÁMETROS GEOMÉTRICOS ---
Lx, Ly, Lz = 2.0, 4.0, 2.0
L = (Lx, Ly, Lz)
source_position = (0.7, 1.3, 0.8) 
receptor_pos = (1.501, 2.998, 1.203)
f_max_mallado = 200.0

# --- 3. GENERACIÓN DE MALLA ---
print("--- Generando/verificando malla ---")
h_min_global = generate_mesh_parallelepiped([Lx, Ly], Lz, source_position, f_max_mallado)
print(f"h_min recibido por el script principal: {h_min_global}")

# --- 4. EJECUCIÓN DE LAS SIMULACIONES ---
print("\n--- Iniciando Simulación FEM con Deconvolución ---")
freqs_fem, mag_H_fem, mag_S_fem, mag_P_fem = FEM_time_optimal_gaussian_impulse(
    "mallado/room.msh", 
    receptor_pos, 
    h_min=h_min_global
)

# Cálculo Analítico por Suma Modal
magnitudes_modal_db = None
if freqs_fem is not None:
    print("\n--- Calculando solución analítica por Suma Modal ---")
    magnitudes_modal_db = compute_modal_transfer_complete(
        rs=source_position,
        rr=receptor_pos,
        L=L,
        freqs=freqs_fem,
        eps=(0.0, 0.0, 0.0)
    )

# --- 5. PLOTEO DE RESULTADOS COMPARATIVOS ---
if all(v is not None for v in [freqs_fem, mag_H_fem, mag_S_fem, mag_P_fem, magnitudes_modal_db]):
    print("\n--- Simulación finalizada. Generando gráfico comparativo ---")
    
    # --- PROCESAMIENTO DE DATOS PARA PLOTEO ---
    H_fem_db = 20 * np.log10(np.abs(mag_H_fem) + 1e-12)
    P_fem_db = 20 * np.log10(np.abs(mag_P_fem) + 1e-12)
    S_fem_db = 20 * np.log10(np.abs(mag_S_fem) + 1e-12)
    
    # Normalizar las curvas para comparar su FORMA, haciendo que empiecen en 0 dB
    H_fem_db_normalized = H_fem_db - H_fem_db[0]
    modal_db_normalized = magnitudes_modal_db - magnitudes_modal_db[0]
    P_fem_db_normalized = P_fem_db - P_fem_db[0] # <-- NORMALIZACIÓN AÑADIDA
    S_fem_db_normalized = S_fem_db - S_fem_db[0] # <-- NORMALIZACIÓN AÑADIDA

    # --- GRÁFICO ---
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # --- Curvas Principales a Comparar (Funciones de Transferencia H(f)) ---
    ax.plot(freqs_fem, modal_db_normalized, 
            label='Analítica H(f) (Suma Modal)', 
            color='red', linestyle='--', linewidth=2.5, zorder=10)
    ax.plot(freqs_fem, H_fem_db_normalized, 
            label='Simulación H(f) (FEM Deconvolucionada)', 
            color='blue', linewidth=1.5, alpha=0.9)

    # --- Curvas de Diagnóstico (para entender la simulación) ---
    ax.plot(freqs_fem, P_fem_db_normalized, # <-- USANDO LA VARIABLE NORMALIZADA
            label='Presión P(f) Normalizada (FEM sin deconvolución)', # <-- ETIQUETA ACTUALIZADA
            color='purple', linestyle=':', linewidth=2, alpha=0.6)
    ax.plot(freqs_fem, S_fem_db_normalized, # <-- USANDO LA VARIABLE NORMALIZADA
            label='Espectro de la Fuente S(f) Normalizado', # <-- ETIQUETA ACTUALIZADA
            color='green', linestyle=':', linewidth=2, alpha=0.6)
    
    # --- Configuración Final del Gráfico ---
    ax.set_title("Comparación: Simulación FEM vs. Solución Analítica (Paredes Rígidas)", fontsize=16)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud Normalizada (dB)")
    
    if len(freqs_fem) > 0:
        ax.set_xlim(np.min(freqs_fem), np.max(freqs_fem))
        
    y_min_plot = np.min(modal_db_normalized[np.isfinite(modal_db_normalized)]) - 10
    ax.set_ylim(max(y_min_plot, -90), 40)

    ax.grid(True, which='both', linestyle=':')
    ax.legend()
    plt.tight_layout()
    plt.show()

else:
    print("\nNo se recibieron datos de la simulación para graficar (proceso no principal o error).")