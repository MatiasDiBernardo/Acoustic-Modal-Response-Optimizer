import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.signal import savgol_filter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FEM.FEM_source_time import FEM_time_optimal_gaussian_impulse
from FEM.FEM_source import FEM_Source_Solver_Average
from FEM.mode_sumation import compute_modal_transfer
from mesh.mesh_3D_simple import create_simple_mesh
from mesh.mesh_3D_generator import generate_mesh_parallelepiped

# Parámetros iniciales
L = (3.0, 4.0, 2.2)          # Room dimensions [m]
rs = [1.0, 1.5, 1.2]        # Source position [m]
rr = [2.9, 2.0, 1.8]        # Receiver position [m]

f_max = 150
f_min = 30

freqs = np.arange(f_min, f_max, 2)
mesh = "room_test"  # Crear malla con el script correspondiente

# Crea la malla FEM Source
create_simple_mesh(L[0], L[1], L[2], rs, f_max, mesh)

# Crea malla FEM Time
generate_mesh_parallelepiped([L[0], L[1]], L[2], rs, f_max)

# Modal Sum (resolución analítica sumando auto funciones)
rta = compute_modal_transfer(rs, rr, L, freqs)
rta -= rta[0]

# FEM Source común (Calculo por frecuencia de la ecuación de Hemholtz)
time_start_source = time.time()
name_mesh = f"mallado/{mesh}.msh"
res_fem_s = FEM_Source_Solver_Average(freqs, name_mesh, rr)
res_tot_prom_s = np.sum(res_fem_s, axis=0) / 7
res_tot_prom_s -= res_tot_prom_s[0]
source_smooth = savgol_filter(res_tot_prom_s, window_length=10, polyorder=2)
time_end_source = time.time() - time_start_source

# FEM Time (Simula en tiempo una respuesta al impulso y calcula RIR)
time_start_time = time.time()
freqs_fem, mag_H_fem, mag_S_fem, mag_P_fem = FEM_time_optimal_gaussian_impulse(f"mallado/{mesh}.msh", rr, f_min, f_max, 0, True) 
H_fem_db = 20 * np.log10(np.abs(mag_H_fem) + 1e-12)
H_fem_db_normalized = H_fem_db - H_fem_db[0]
time_end_time = time.time() - time_start_time

print(" ")
print("Resultados: ")
print("El tiempo que tardo FEM Source es de: ", time_end_source)
print("El tiempo que tardo FEM Time es de: ", time_end_time)
plt.figure()
plt.plot(freqs, rta, '--', label="Modal Summation", linewidth=2)
plt.plot(freqs, res_tot_prom_s,'--', label="FEM Source")
# plt.plot(freqs, source_smooth, label="FEM Source Smooth")
plt.plot(freqs_fem, H_fem_db_normalized, '--', label="FEM Time", linewidth=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)
plt.show()