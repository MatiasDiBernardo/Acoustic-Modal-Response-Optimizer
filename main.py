from geometry_generator import calculation_of_geometry
from mesh_3D_generator import generate_mesh
from FEM_eigensolver import FEM_solver, FEM_solver_display
from FEM_visualization import mode_shape_visulization
from merit_figure import modal_response_merit

import numpy as np
import matplotlib.pyplot as plt
import os

def example_with_visualization():
    # Estos en en centímetros
    Lx = 400       # Largo de la sala en X 
    Ly = 600       # Largo de la sala en Y
    Lz = 220       # Alto de la sala
    Dx = 80        # Delta X
    Dy = 100       # Delta Y

    # Parametros de control
    N = 250        # Densidad de la grilla del generador de geometrías
    M = 100        # Cantidad de salas a generar
    n_walls = 4    # Número de cortes en las paredes
    n_eigen = 12   # Número de autovalores que calculo
    idx_mode = 9   # Número del modo a visualizar

    # Calculo y visualización
    sala_base = [(0, 0), (Lx/100, 0), (Lx/100, Ly/100), (0, Ly/100)]
    salas = calculation_of_geometry(Lx, Ly, Dx, Dy, N, M, n_walls)
    Z = Lz/100

    # Generar y calcular autovalores para 4 salas
    for i in range(4):
        print(salas[i])
        generate_mesh(salas[i], Z)
        FEM_solver_display(n_eigen, idx_mode)
        mode_shape_visulization()

        # Borro todos los archivos generados para dejar repo clean
        path_files = ["mode1.h5", "mode1.vtu", "mode1.xdmf", "room_mesh.h5", "room_mesh.vtu", "room_mesh.xdmf", "room.h5", "room.xdmf"]
        for path in path_files:
            os.remove(path)

def example_optimization():
    # Estos en en centímetros
    Lx = 400       # Largo de la sala en X 
    Ly = 600       # Largo de la sala en Y
    Lz = 220       # Alto de la sala
    Dx = 80        # Delta X
    Dy = 100       # Delta Y

    # Parametros de control
    N = 250        # Densidad de la grilla del generador de geometrías
    M = 50        # Cantidad de salas a generar
    n_walls = 4    # Número de cortes en las paredes
    n_eigen = 15   # Número de autovalores que calculo
    
    # Almacenar toda la data
    rooms = calculation_of_geometry(Lx, Ly, Dx, Dy, N, M, n_walls)
    frecuency_dist = []
    merit_values = []
    Z = Lz/100

    for i in range(M):
        print("Vamos por el room: ", i)
        generate_mesh(rooms[i], Z)
        f_dist = FEM_solver(n_eigen)
        m_value = modal_response_merit(f_dist)
        
        frecuency_dist.append(f_dist)
        merit_values.append(m_value)
    
    merit_values = np.array(merit_values)

    idx_best_room = np.argmin(merit_values)
    idx_worst_room = np.argmax(merit_values)
    
    print("El valor de mértio mínimo es: ", merit_values[idx_best_room])
    print("La sala que lo genero es: ", rooms[idx_best_room])
    print("............................")
    print("El valor de mértio máximo es: ", merit_values[idx_worst_room])
    print("La sala que lo genero es: ", rooms[idx_worst_room])
    print("............................")
    print("El valor de mértio promedio es: ", np.mean(merit_values))
    
    modes_amp = np.ones(len(frecuency_dist[idx_best_room])) * 1.1
    modes_amp2 = np.ones(len(frecuency_dist[idx_worst_room])) 

    plt.stem(frecuency_dist[idx_best_room], modes_amp, 'b', markerfmt='bo', label='Best Modal Dist')
    plt.stem(frecuency_dist[idx_worst_room], modes_amp2, 'g', markerfmt='go', label='Worst Modal Dist')
    # plt.stem(frecuency_dist[np.random.randint(0, M)], modes_amp2, 'n', markerfmt='go', label='Random Modal Dist')
    plt.xlabel("Frequency")
    plt.ylabel("Modal Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

example_optimization()