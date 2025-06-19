import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from room.geometry_generator import calculation_of_geometry
from mesh.mesh_3D_complex import create_complex_mesh
from FEM.FEM_source import FEM_Source_Solver_Average
from aux.merit_figure import merit_magnitude_deviation, merit_spatial_deviation
from plots.graph_room_outline import plot_room_outline

def example_optim():
    # Dimensiones sala (en centímetros)
    Lx = 250       # Largo de la sala en X 
    Ly = 300       # Largo de la sala en Y
    Lz = 220       # Alto de la sala
    Dx = 50        # Delta X
    Dy = 80        # Delta Y
    Dz = 10        # Delta Z
    
    # Posiciones fuente y receptor (en metros)
    source_position = (1.9, 1.0, 1.3)
    receptor_position = (1.25, 1.9, 1.2)
    f_max = 180
    start_time = time.time()
    
    # Parametros de control
    N = 250        # Densidad de la grilla del generador de geometrías
    M = 200       # Cantidad de salas a generar
    n_walls = 2    # Número de cortes en las paredes
    res_freq = 2
    freqs_eval = np.arange(20, f_max, res_freq)  # Frecuencias a evaluar
    
    # Almacenar toda la data
    rooms = calculation_of_geometry(Lx, Ly, Dx, Dy, N, M, n_walls)
    mesh = "room_mesh_complex"  # Crear malla con el script correspondiente
    mesh1 = "room_mesh_complex1"  # Crear malla con el script correspondiente
    mesh2 = "room_mesh_complex2"  # Crear malla con el script correspondiente
    mesh3 = "room_mesh_complex3"  # Crear malla con el script correspondiente
    mag_responses = []
    merit_sv_values = []
    merit_md_values = []

    # Itera y almacena los valores para encontrar el mejor cuarto
    for i in range(M):
        print("Vamos por el room: ", i)

        # Crea la malla de la geometría selecionada
        Z = (Lz - np.random.uniform(0, Dz))/100
        create_complex_mesh(rooms[i], Z, source_position, 80, mesh1)
        create_complex_mesh(rooms[i], Z, source_position, 140, mesh2)
        create_complex_mesh(rooms[i], Z, source_position, 200, mesh3)

        # Evalua la rta en frecuencia para esa sala
        f1 = np.arange(20, 80, res_freq)
        res1 = FEM_Source_Solver_Average(f1, f'mallado/{mesh1}.msh', receptor_position)

        f2 = np.arange(80, 140, res_freq)
        res2 = FEM_Source_Solver_Average(f2, f'mallado/{mesh2}.msh', receptor_position)

        f3 = np.arange(140, 200, res_freq)
        res3 = FEM_Source_Solver_Average(f3, f'mallado/{mesh3}.msh', receptor_position)

        res_tot = np.hstack([res1, res2, res3])
        res_tot_prom = np.sum(res_tot, axis=0) / 7

        # Calcula figuras de mérito
        sv_merit = merit_spatial_deviation(res_tot)
        md_merit = merit_magnitude_deviation(res_tot)
        
        merit_sv_values.append(sv_merit)
        merit_md_values.append(md_merit)
        mag_responses.append(res_tot_prom)
    
    f_tot =  np.hstack([f1, f2, f3])
    merit_sv_values = np.array(merit_sv_values)
    merit_md_values = np.array(merit_md_values)
    merit_general = merit_md_values + merit_sv_values

    idx_best_room = np.argmin(merit_general)
    idx_worst_room = np.argmax(merit_general)

    print(f"El valor de mértio mínimo es: {merit_general[idx_best_room]} | Con SV: {merit_sv_values[idx_best_room]} | Con MD: {merit_md_values[idx_best_room]}" )
    #print("La sala que lo genero es: ", rooms[idx_best_room])
    print("............................")
    print(f"El peor valor de mérito es: {merit_general[idx_worst_room]} | Con SV: {merit_sv_values[idx_worst_room]} | Con MD: {merit_md_values[idx_worst_room]}" )
    #print("La sala que lo genero es: ", rooms[idx_worst_room])
    print("............................")
    print("El valor de mértio promedio es: ", np.mean(merit_general))
    print("............................")
    print("El tiempo de ejecución en minutos fue de: ", (time.time() - start_time)/60)
    
    best_room_mag = mag_responses[idx_best_room]
    worst_room_mag = mag_responses[idx_worst_room]
    random_room_mag = mag_responses[np.random.randint(0, M - 1)]
    
    plt.figure("Resultado magnitud")
    plt.plot(f_tot, best_room_mag, label= "Best Modal Dist")
    plt.plot(f_tot, worst_room_mag, label= "Worst Modal Dist")
    plt.plot(f_tot, random_room_mag, label= "Random Modal Dist")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plot_room_outline(Lx, Ly, Dx, Dy, source_position, receptor_position, rooms[idx_best_room], "Distribución mejor cuarto")
    plot_room_outline(Lx, Ly, Dx, Dy, source_position, receptor_position, rooms[idx_worst_room], "Distribución peor cuarto")
    print("        ")

example_optim()