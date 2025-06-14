import numpy as np
import matplotlib.pyplot as plt
import time

from FEM.FEM_source_parallel import FEM_Source_Solver_Average
from mpi4py import MPI
    
comm = MPI.COMM_WORLD
rank = comm.rank

res = 1  # Resolución del salto de frecuencias
freqs_eval = np.arange(20, 200, res)
mesh = "mallado/room_max200.msh"  # Crear malla con el script correspondiente
receptor_position1 = (2.8, 2.0, 1.8)

# Solución grilla común
time_base = time.time()
res_base = FEM_Source_Solver_Average(freqs_eval, mesh, receptor_position1)
print(res_base)
#res_base_prom = np.sum(res_base, axis=0)/7
t1 = time.time() - time_base
print("El tiempo de ejecución fue de: ", t1)

# Handle results
if rank == 0 and res_base is not None:
    plt.plot(freqs_eval, res_base[0])
    plt.show()
    print("Simulation successful. Results saved.")
elif rank == 0:
    print("Simulation failed. Check error messages.")
