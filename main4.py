import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows # Necesitas esta importación
import scipy.fft # Importar el módulo de FFT

# 1. IMPORTACIONES DE FUNCIONES LOCALES
from FEM.FEM_source_time import FEM_time_optimal_gaussian_impulse
from FEM.mode_sumation import compute_modal_transfer
from mesh.mesh_3D_generator import generate_mesh_parallelepiped
from Source_image import source_image_cuboid

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
    f_min =20,
    f_max=f_max_mallado,
    h_min=h_min_global
)

# Inicializar variables para los resultados de los otros métodos
magnitudes_modal_db = None
H_ism_db_normalized = None

# -----------------------------------------------------------------------------
# INICIO DEL BLOQUE LÓGICO PRINCIPAL
# Todas las simulaciones dependientes se ejecutan solo si el FEM fue exitoso
# -----------------------------------------------------------------------------
if freqs_fem is not None:
    # --- Cálculo Analítico por Suma Modal ---
    print("\n--- Calculando solución analítica por Suma Modal ---")
    magnitudes_modal_db = compute_modal_transfer(
        rs=source_position,
        rr=receptor_pos,
        L=L,
        freqs=freqs_fem,
        eta=1e-9
    )

    # ##########################################################################
    # INICIO DE BLOQUE MOVIDO Y CORREGIDO
    # Este bloque ahora está DENTRO del 'if freqs_fem is not None:'
    # ##########################################################################
    
    # --- Cálculo por Fuente Imagen (ISM) para Comparación ---
    # 1. Parámetros para la simulación ISM
    fs_ism = 8000  # Usar un fs estándar y suficientemente alto (como entero)
    order_ism = 40 # Un orden de reflexión razonable y rápido

    print(f"\n--- Calculando por Fuente Imagen con fs={fs_ism} Hz y orden={order_ism} ---")

    # 2. Ejecutar la simulación ISM para obtener la RIR
    rir_image = source_image_cuboid(L, source_position, receptor_pos, order_ism, fs_ism)

    # 3. Procesar el resultado de ISM para hacerlo comparable
    if rir_image is not None and len(rir_image) > 0:
                # Crear una ventana (ej. Hanning) del mismo tamaño que la RIR
        hanning_window = windows.hann(len(rir_image))
        # Multiplicar la RIR por la ventana para suavizar los bordes
        rir_image_windowed = rir_image * hanning_window
        # Calcular la FFT de la RIR
        H_ism_complex = scipy.fft.rfft(rir_image)
        # Calcular el vector de frecuencias correspondiente
        freqs_ism = scipy.fft.rfftfreq(len(rir_image), 1 / fs_ism)
        
        # 4. Interpolar el resultado de ISM a las frecuencias del FEM
        mag_H_ism_interpolated = np.interp(freqs_fem, freqs_ism, np.abs(H_ism_complex))

        # 5. Convertir a dB y normalizar
        H_ism_db = 20 * np.log10(mag_H_ism_interpolated + 1e-12)
        H_ism_db_normalized = H_ism_db - np.mean(H_ism_db)
    
    # ##########################################################################
    # FIN DE BLOQUE MOVIDO Y CORREGIDO
    # ##########################################################################


# --- 5. PLOTEO DE RESULTADOS COMPARATIVOS ---
# Se verifica que todos los resultados (incluyendo el de ISM) se hayan calculado
if all(v is not None for v in [freqs_fem, mag_H_fem, mag_S_fem, mag_P_fem, magnitudes_modal_db, H_ism_db_normalized]):
    print("\n--- Simulación finalizada. Generando gráfico comparativo ---")

    # --- PROCESAMIENTO DE DATOS PARA PLOTEO ---
    H_fem_db = 20 * np.log10(np.abs(mag_H_fem) + 1e-12)
    P_fem_db = 20 * np.log10(np.abs(mag_P_fem) + 1e-12)
    S_fem_db = 20 * np.log10(np.abs(mag_S_fem) + 1e-12)

    # Normalizar las curvas para comparar su FORMA
    H_fem_db_normalized = H_fem_db - np.mean(H_fem_db)
    modal_db_normalized = magnitudes_modal_db - np.mean(magnitudes_modal_db)
    P_fem_db_normalized = P_fem_db - np.mean(P_fem_db)
    S_fem_db_normalized = S_fem_db - np.mean(S_fem_db)


    # --- GRÁFICO ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # --- Curvas Principales a Comparar (Funciones de Transferencia H(f)) ---
    ax.plot(freqs_fem, modal_db_normalized,
            label='Analítica H(f) (Suma Modal)',
            color='red', linestyle='--', linewidth=2.5, zorder=10)
            
    ax.plot(freqs_fem, H_fem_db_normalized,
            label='Simulación H(f) (FEM Deconvolucionada)',
            color='blue', linewidth=1.5, alpha=0.9)

    ax.plot(freqs_fem, H_ism_db_normalized,
            label='Fuente Imagen H(f) (Interpolada)',
            color='darkorange', linestyle='-.', linewidth=2.0, zorder=9)

    # --- Curvas de Diagnóstico ---
    ax.plot(freqs_fem, P_fem_db_normalized,
            label='Presión P(f) Normalizada (FEM)',
            color='purple', linestyle=':', linewidth=2, alpha=0.6)
    ax.plot(freqs_fem, S_fem_db_normalized,
            label='Espectro Fuente S(f) Normalizado',
            color='green', linestyle=':', linewidth=2, alpha=0.6)

    # --- Configuración Final del Gráfico ---
    ax.set_title("Comparación: Simulación FEM vs. Solución Analítica vs. Fuente Imagen (Paredes Rígidas)", fontsize=16)
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