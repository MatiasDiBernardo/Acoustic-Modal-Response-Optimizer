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
    f_max = 200
    start_time = time.time()
    
    # Parametros de control
    N = 250        # Densidad de la grilla del generador de geometrías
    M = 200        # Cantidad de salas a generar
    n_walls = 4    # Número de cortes en las paredes
    freqs_eval = np.arange(20, 200, 2)  # Frecuencias a evaluar
    
    # Almacenar toda la data
    rooms = calculation_of_geometry(Lx, Ly, Dx, Dy, N, M, n_walls)
    mesh = "room_mesh_complex"  # Crear malla con el script correspondiente
    mag_responses = []
    merit_sv_values = []
    merit_md_values = []

    # Itera y almacena los valores para encontrar el mejor cuarto
    for i in range(M):
        print("Vamos por el room: ", i)

        # Crea la malla de la geometría selecionada
        Z = (Lz - np.random.uniform(0, Dz))/100
        create_complex_mesh(rooms[i], Z, source_position, f_max, mesh)
        
        # Evalua la rta en frecuencia para esa sala
        mag = FEM_Source_Solver_Average(freqs_eval, f'mallado/{mesh}.msh', receptor_position)

        # Calcula figuras de mérito
        sv_merit = merit_spatial_deviation(mag)
        md_merit = merit_magnitude_deviation(mag)
        
        merit_sv_values.append(sv_merit)
        merit_md_values.append(md_merit)
        mag_responses.append(mag)
    
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
    
    best_room_mag = np.sum(mag_responses[idx_best_room], axis=0)/7
    worst_room_mag = np.sum(mag_responses[idx_worst_room], axis=0)/7
    random_room_mag = np.sum(mag_responses[np.random.randint(0, M - 1)], axis=0)/7
    
    plt.figure("Resultado magnitud")
    plt.plot(freqs_eval, best_room_mag, label= "Best Modal Dist")
    plt.plot(freqs_eval, worst_room_mag, label= "Worst Modal Dist")
    plt.plot(freqs_eval, random_room_mag, label= "Random Modal Dist")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plot_room_outline(Lx, Ly, Dx, Dy, source_position, receptor_position, rooms[idx_best_room], "Distribución mejor cuarto")
    plot_room_outline(Lx, Ly, Dx, Dy, source_position, receptor_position, rooms[idx_worst_room], "Distribución peor cuarto")

example_optim()