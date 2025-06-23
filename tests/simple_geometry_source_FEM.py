import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from room.geometry_generator import calculation_of_geometry_simple
from mesh.mesh_3D_simple import create_simple_mesh
from FEM.FEM_source import FEM_Source_Solver_Average
from aux.merit_figure import merit_spatial_deviation, merit_magnitude_deviation

import numpy as np
import matplotlib.pyplot as plt

def example_optim_simple():
    # Dimensiones sala (en centímetros)
    Lx = 2.5
    Ly = 3
    Lz = 2.2       
    Dx = 0.5        
    Dy = 0.8       
    Dz = 0.1       
    
    # Posiciones fuente y receptor (en metros)
    source_position = (1.9, 1.0, 1.3)
    receptor_position = (1.25, 1.9, 1.2)

    # Parametros de control
    M = 200        # Cantidad de salas a generar
    res_freq = 1   # Resolución freq
    
    # Almacenar toda la data
    mesh1 = "room_80"
    mesh2 = "room_140"
    mesh3 = "room_200"

    rooms = []
    mag_responses = []
    merit_sv_values = []
    merit_md_values = []
    
    for i in range(M):
        print("Vamos por el room: ", i)

        # Simple version
        Lx_new, Ly_new, Lz_new = calculation_of_geometry_simple(Lx, Ly, Lz, Dx, Dy, Dz)

        dx_room = (Lx - Lx_new)/2
        dy_room = (Ly - Ly_new)/2
        new_source_pos = (source_position[0] - dx_room, source_position[1] - dy_room, source_position[2])
        new_receptor_pos = (receptor_position[0] - dx_room, receptor_position[1] - dy_room, receptor_position[2])

        # Arreglar posición de fuente y receptor para garantizar simetría
        create_simple_mesh(Lx_new, Ly_new, Lz_new, new_source_pos, 80, mesh1)
        create_simple_mesh(Lx_new, Ly_new, Lz_new, new_source_pos, 140, mesh2)
        create_simple_mesh(Lx_new, Ly_new, Lz_new, new_source_pos, 200, mesh3)

        # Evalua la rta en frecuencia para esa sala
        f1 = np.arange(20, 80, res_freq)
        res1 = FEM_Source_Solver_Average(f1, f'mallado/{mesh1}.msh', new_receptor_pos)

        f2 = np.arange(80, 140, res_freq)
        res2 = FEM_Source_Solver_Average(f2, f'mallado/{mesh2}.msh', new_receptor_pos)

        f3 = np.arange(140, 200, res_freq)
        res3 = FEM_Source_Solver_Average(f3, f'mallado/{mesh3}.msh', new_receptor_pos)

        res_tot = np.hstack([res1, res2, res3])
        res_tot_prom = np.sum(res_tot, axis=0) / 7
        f_tot =  np.hstack([f1, f2, f3])

        # Calcula figuras de mérito
        sv_merit = merit_spatial_deviation(res_tot)
        md_merit = merit_magnitude_deviation(res_tot)
        
        merit_sv_values.append(sv_merit)
        merit_md_values.append(md_merit)

        mag_responses.append(res_tot_prom)
        rooms.append((Lx_new, Ly_new, Lz_new))
    
    merit_sv_values = np.array(merit_sv_values)
    merit_md_values = np.array(merit_md_values)
    merit_general = merit_md_values + merit_sv_values

    idx_best_room = np.argmin(merit_general)
    idx_worst_room = np.argmax(merit_general)
    
    print(f"El valor de mértio mínimo es: {merit_general[idx_best_room]} | Con SV: {merit_sv_values[idx_best_room]} | Con MD: {merit_md_values[idx_best_room]}" )
    print("La sala que lo genero es: ", rooms[idx_best_room])
    print("............................")
    print(f"El peor valor de mérito es: {merit_general[idx_worst_room]} | Con SV: {merit_sv_values[idx_worst_room]} | Con MD: {merit_md_values[idx_worst_room]}" )
    print("La sala que lo genero es: ", rooms[idx_worst_room])
    print("............................")
    print("El valor de mértio promedio es: ", np.mean(merit_general))
    
    plt.plot(f_tot, mag_responses[idx_best_room], label="Best Modal Dist")
    plt.plot(f_tot, mag_responses[idx_worst_room], label="Worst Modal Dist")
    plt.plot(f_tot, mag_responses[np.random.randint(0, M - 1)], label="Random Modal Dist")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()

example_optim_simple()
