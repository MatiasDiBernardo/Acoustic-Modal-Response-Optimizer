import numpy as np
import matplotlib.pyplot as plt

from FEM_source import FEM_Source_Solver, FEM_Source_Solver_Spatial_Average

# Solución para ver funcionamiento del solver por fuente

freqs_eval = np.arange(20, 200, 2)
mesh = "mallado/room.msh"  # Crear malla con el script correspondiente
receptor_position1 = (0.7, 0.7, 0.7)

res1 = FEM_Source_Solver_Spatial_Average(freqs_eval, mesh, receptor_position1)

plt.plot(freqs_eval, res1[0])
plt.plot(freqs_eval, res1[1])
plt.plot(freqs_eval, res1[2])
plt.plot(freqs_eval, res1[3])
plt.plot(freqs_eval, res1[4])
plt.plot(freqs_eval, res1[5])
plt.grid()
plt.show()
