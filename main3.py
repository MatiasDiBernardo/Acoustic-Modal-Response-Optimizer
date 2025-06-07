from geometry_generator import calculation_of_geometry, calculation_of_geometry_simple
from mesh_3D_generator import generate_mesh_parallelepiped
from mesh_3D_simple import create_simple_mesh
from FEM_source import FEM_Source_Solver_Average
from merit_figure import merit_spatial_deviation, merit_magnitude_deviation

import numpy as np
import matplotlib.pyplot as plt

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
    
    receptor_position = (2, 3.5, 1.2)

    # Parametros de control
    N = 250        # Densidad de la grilla del generador de geometrías
    M = 50        # Cantidad de salas a generar
    n_walls = 4    # Número de cortes en las paredes
    freqs_eval = np.arange(20, 200, 2)  # Frecuencias a evaluar
    
    # Almacenar toda la data
    rooms = calculation_of_geometry(Lx, Ly, Dx, Dy, N, M, n_walls)
    mesh = "room_mesh.msh"  # Crear malla con el script correspondiente
    mag_responses = []
    merit_sv_values = []
    merit_md_values = []
    Z = Lz/100
    
    for i in range(M):
        print("Vamos por el room: ", i)

        # Crea la malla de la geometría selecionada
        #mesh_generator_complex(rooms[i], Z, source_position)
        
        # Evalua la rta en frecuencia para esa sala
        mag = FEM_Source_Solver_Average(freqs_eval, mesh, receptor_position)

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
    freqs_eval = np.arange(20, 200, 2)  # Frecuencias a evaluar
    
    # Almacenar toda la data
    mesh = "mallado/room_simple.msh"  # Crear malla con el script correspondiente

    rooms = []
    mag_responses = []
    merit_sv_values = []
    merit_md_values = []
    
    for i in range(M):
        print("Vamos por el room: ", i)

        # Simple version
        Lx_new, Ly_new, Lz_new = calculation_of_geometry_simple(Lx, Ly, Lz, Dx, Dy, Dz)
        create_simple_mesh(Lx_new, Ly_new, Lz_new, source_position)

        # Evalua la rta en frecuencia para esa sala
        mag = FEM_Source_Solver_Average(freqs_eval, mesh, receptor_position)

        # Calcula figuras de mérito
        sv_merit = merit_spatial_deviation(mag)
        md_merit = merit_magnitude_deviation(mag)
        
        merit_sv_values.append(sv_merit)
        merit_md_values.append(md_merit)
        mag_avg = np.sum(mag, axis=0)/7
        mag_responses.append(mag_avg)
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
    
    plt.plot(freqs_eval, mag_responses[idx_best_room], label="Best Modal Dist")
    plt.plot(freqs_eval, mag_responses[idx_worst_room], label="Worst Modal Dist")
    plt.plot(freqs_eval, mag_responses[np.random.randint(0, M - 1)], label="Random Modal Dist")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()

example_optim_simple()