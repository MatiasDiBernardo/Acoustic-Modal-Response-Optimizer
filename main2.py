import numpy as np
import matplotlib.pyplot as plt
import time

from FEM_source import FEM_Source_Solver, FEM_Source_Solver_Spatial_Average

# Solución sin adaptar grilla
freqs_eval = np.arange(20, 200, 2)

mesh = "mallado/room_max200.msh"  # Crear malla con el script correspondiente
space_average = 6  # El agregar promediado espacial casi no afecta la preformance

receptor_position1 = (2, 4, 1.2)

time_base = time.time()
res_base = FEM_Source_Solver_Spatial_Average(freqs_eval, mesh, receptor_position1, space_average)
t1 = time.time()
time1 = t1 - time_base

# Solución adaptando grilla
f1 = np.arange(20, 80, 2)
mesh1 =  "mallado/room_max80.msh"
res1 = FEM_Source_Solver_Spatial_Average(f1, mesh1, receptor_position1, space_average)

f2 = np.arange(83, 140, 2)
mesh2 =  "mallado/room_max140.msh"
res2 = FEM_Source_Solver_Spatial_Average(f2, mesh2, receptor_position1, space_average)

f3 = np.arange(143, 200, 2)
mesh3 =  "mallado/room_max200.msh"
res3 = FEM_Source_Solver_Spatial_Average(f3, mesh3, receptor_position1, space_average)

res_tot = np.hstack([res1, res2, res3])
f_tot =  np.hstack([f1, f2, f3])

t2 = time.time()
time2 = t2 - t1

print("El tiempo de cada ejecución fue: ")
print("Tiempo normal: ", time1)
print("Tiempo grilla adapt: ", time2)

plt.plot(freqs_eval, res_base[0], label="Base")
plt.plot(f_tot, res_tot[0], label="Adapt")
plt.legend()
# plt.plot(freqs_eval, res1[1])
# plt.plot(freqs_eval, res1[2])
# plt.plot(freqs_eval, res1[3])
# plt.plot(freqs_eval, res1[4])
# plt.plot(freqs_eval, res1[5])
plt.grid()
plt.show()
