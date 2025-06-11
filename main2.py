import numpy as np
import matplotlib.pyplot as plt
import time

from FEM.FEM_source import FEM_Source_Solver, FEM_Source_Solver_Average

res = 1  # Resolución del salto de frecuencias
freqs_eval = np.arange(20, 200, res)
mesh = "mallado/room_max200.msh"  # Crear malla con el script correspondiente
receptor_position1 = (2.8, 2.0, 1.8)

# Solución grilla común
time_base = time.time()
res_base = FEM_Source_Solver_Average(freqs_eval, mesh, receptor_position1)
res_base_prom = np.sum(res_base, axis=0)/7
t1 = time.time()
time1 = t1 - time_base

# Solución adaptando grilla
res = 2
f1 = np.arange(20, 80, res)
mesh1 =  "mallado/room_max80.msh"
res1 = FEM_Source_Solver_Average(f1, mesh1, receptor_position1)

f2 = np.arange(80, 140, res)
mesh2 =  "mallado/room_max140.msh"
res2 = FEM_Source_Solver_Average(f2, mesh2, receptor_position1)

f3 = np.arange(140, 200, res)
mesh3 =  "mallado/room_max200.msh"
res3 = FEM_Source_Solver_Average(f3, mesh3, receptor_position1)

res_tot = np.hstack([res1, res2, res3])
res_tot_prom = np.sum(res_tot, axis=0) / 7
f_tot =  np.hstack([f1, f2, f3])

t2 = time.time()
time2 = t2 - t1

# Solución simple
res_simple = FEM_Source_Solver(freqs_eval, mesh, receptor_position1)
res_simple = 20 * np.log10(np.abs(res_simple))
time3 = time.time() - t2

print("El tiempo de cada ejecución fue: ")
print("Tiempo normal sin average: ", time3)
print("Tiempo normal average: ", time1)
print("Tiempo grilla adapt: ", time2)

plt.plot(freqs_eval, res_simple, "--", label="Base Central")
plt.plot(freqs_eval, res_base_prom, "--", label="Average")
plt.plot(f_tot, res_tot_prom, "--", label ="Average Grilla Adapt")
# plt.plot(f_tot, res_base[0], "--")
# plt.plot(f_tot, res_base[1], "--")
# plt.plot(f_tot, res_base[2], "--")
# plt.plot(f_tot, res_base[3], "--")
# plt.plot(f_tot, res_base[4], "--")
# plt.plot(f_tot, res_base[5], "--")
# plt.plot(f_tot, res_base[6], "--")
plt.legend()
plt.grid()
plt.show()

