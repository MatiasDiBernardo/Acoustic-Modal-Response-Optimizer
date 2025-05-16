import numpy as np
import matplotlib.pyplot as plt

from FEM_source import FEM_Source_Solver, FEM_Source_Solver_Spatial_Average

# Solución para ver funcionamiento del solver por fuente

freqs_eval = np.arange(20, 200, 2)
mesh = "mallado/esfera_en_paralelepipedo_refined.msh"  # Crear malla con el script correspondiente
receptor_position1 = (0.7, 0.7, 0.7)
receptor_position2 = (0.5, 0.5, 0.7)
receptor_position3 = (1, 0.5, 0.7)

# Refomular para que el ciclo sea adentro de la función (ciclo de freqs) y no tiene que crear los objetos en cada pasada
# Además, para evaluar en diferentes posiciones podría usar la función una sola vez
res1 = FEM_Source_Solver_Spatial_Average(freqs_eval, mesh, receptor_position1)

plt.plot(freqs_eval, 20 * np.log10(res1[0]))
plt.plot(freqs_eval, 20 * np.log10(res1[1]))
plt.plot(freqs_eval, 20 * np.log10(res1[2]))
plt.plot(freqs_eval, 20 * np.log10(res1[3]))
plt.plot(freqs_eval, 20 * np.log10(res1[4]))
plt.plot(freqs_eval, 20 * np.log10(res1[5]))
plt.grid()
plt.show()
