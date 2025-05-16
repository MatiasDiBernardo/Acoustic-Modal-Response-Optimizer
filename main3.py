from geometry_generator import calculation_of_geometry
from mesh_3D_generator import generate_mesh
from FEM_source import FEM_Source_Solver_Spatial_Average
from merit_figure import merit_spatial_deviation, merit_magnitude_deviation

import numpy as np
import matplotlib.pyplot as plt

# freqs_eval = np.arange(20, 200, 2)
# mesh = "mallado/esfera_en_paralelepipedo_refined.msh"  # Crear malla con el script correspondiente
# receptor_position1 = (0.7, 0.7, 0.7)

# res1 = FEM_Source_Solver_Spatial_Average(freqs_eval, mesh, receptor_position1)


# sv_merit = merit_spatial_deviation(res1)
# md_merit = merit_magnitude_deviation(res1)

# print("Mértio espacial: ", sv_merit)
# print("Mértio magnitud: ", md_merit)

# Optimización con el solver por fuente
def example_optim():
    # Dimensiones sala (en centímetros)
    Lx = 400       # Largo de la sala en X 
    Ly = 600       # Largo de la sala en Y
    Lz = 220       # Alto de la sala
    Dx = 80        # Delta X
    Dy = 100       # Delta Y
    
    # Posiciones fuente y receptor (en metros)
    source_position = (2.5, 2, 1.2)
    source_radio = 0.3
    
    receptor_position = (2, 3.5, 1.2)

    # Parametros de control
    N = 250        # Densidad de la grilla del generador de geometrías
    M = 100        # Cantidad de salas a generar
    n_walls = 4    # Número de cortes en las paredes
    freqs_eval = np.arange(20, 200, 2)  # Frecuencias a evaluar
    
    # Almacenar toda la data
    rooms = calculation_of_geometry(Lx, Ly, Dx, Dy, N, M, n_walls)
    mesh = "room.msh"  # Crear malla con el script correspondiente
    mag_responses = []
    merit_sv_values = []
    merit_md_values = []
    Z = Lz/100
    
    for i in range(M):
        print("Vamos por el room: ", i)

        # Crea la malla de la geometría selecionada
        generate_mesh(rooms[i], Z, source_position, source_radio)

        # Evalua la rta en frecuencia para esa sala
        mag = FEM_Source_Solver_Spatial_Average(freqs_eval, mesh, receptor_position)

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
    print("La sala que lo genero es: ", rooms[idx_best_room])
    print("............................")
    print(f"El peor valor de mérito es: {merit_general[idx_worst_room]} | Con SV: {merit_sv_values[idx_worst_room]} | Con MD: {merit_md_values[idx_worst_room]}" )
    print("La sala que lo genero es: ", rooms[idx_worst_room])
    print("............................")
    print("El valor de mértio promedio es: ", np.mean(merit_general))
    
    plt.plot(freqs_eval, mag_responses[idx_best_room], label="Best Modal Dist")
    plt.plot(freqs_eval, mag_responses[idx_worst_room], label="Worst Modal Dist")
    plt.plot(freqs_eval, mag_responses[np.random.randint(0, M - 1)], label="Random Modal Dist")

    plt.xlabel("Frequency")
    plt.xscale('log')
    plt.ylabel("Modal Amplitude")
    plt.legend()
    plt.grid()
    plt.show()