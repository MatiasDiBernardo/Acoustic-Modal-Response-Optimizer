import numpy as np
import matplotlib.pyplot as plt

from room.geometry_generator import calculation_of_geometry
from mesh.mesh_3D_complex import create_complex_mesh
from mesh.mesh_3D_simple import create_simple_mesh
from FEM.FEM_source import FEM_Source_Solver_Average
from aux.merit_figure import merit_magnitude_deviation, merit_spatial_deviation
from plots.graph_room_outline import plot_room_outline

def find_complex_outline_gen2(Lx, Ly, Lz, Dx, Dy, source_position, receptor_position):
    # Dimensiones sala (en centímetros)
    Lx = int(Lx * 100)       # Largo de la sala en X 
    Ly = int(Ly * 100)       # Largo de la sala en Y
    Dx = int(Dx * 100)       # Delta X
    Dy = int(Dy * 100)       # Delta Y
    
    # Parametros de control
    N = 250        # Densidad de la grilla del generador de geometrías
    M = 200        # Cantidad de salas a generar
    n_walls = 2    # Número de cortes en las paredes
    f_max = 140
    res_freq = 2
    
    # Almacenar toda la data
    rooms = calculation_of_geometry(Lx, Ly, Dx, Dy, N, M, n_walls)
    mesh1 = "room_mesh_complex1"  # Crear malla con el script correspondiente
    mesh2 = "room_mesh_complex2"  # Crear malla con el script correspondiente
    mag_responses = []
    merit_sv_values = []
    merit_md_values = []

    # Itera y almacena los valores para encontrar el mejor cuarto
    for i in range(M):
        # Crea la malla de la geometría selecionada
        # Z = (Lz - np.random.uniform(0, Dz))/100 (vamos a asumir que la altura que se encontró es la mejor)
        create_complex_mesh(rooms[i], Lz, source_position, 80, mesh1)
        create_complex_mesh(rooms[i], Lz, source_position, f_max, mesh2)

        # Evalua la rta en frecuencia para esa sala
        f1 = np.arange(20, 80, res_freq)
        res1 = FEM_Source_Solver_Average(f1, f'mallado/{mesh1}.msh', receptor_position)

        f2 = np.arange(80, f_max, res_freq)
        res2 = FEM_Source_Solver_Average(f2, f'mallado/{mesh2}.msh', receptor_position)

        res_tot = np.hstack([res1, res2])
        res_tot_prom = np.sum(res_tot, axis=0) / 7

        # Calcula figuras de mérito
        sv_merit = merit_spatial_deviation(res_tot)
        md_merit = merit_magnitude_deviation(res_tot)
        
        merit_sv_values.append(sv_merit)
        merit_md_values.append(md_merit)
        mag_responses.append(res_tot_prom)
    
    merit_sv_values = np.array(merit_sv_values)
    merit_md_values = np.array(merit_md_values)
    merit_general = merit_md_values + merit_sv_values

    merit_index = list(enumerate(merit_general))
    merit_sorted = sorted(merit_index, key=lambda x: x[1])

    index_merit_sored, merit_values = zip(*merit_sorted)
    best_rooms = [rooms[i] for i in index_merit_sored]
    best_mag = [mag_responses[i] for i in index_merit_sored]
    
    return best_rooms, merit_values, best_mag

def find_complex_outline_gen3(Lx, Ly, Lz, Dx, Dy, source_position, receptor_position, rooms_prev_gen):
    # Dimensiones sala (en centímetros)
    Lx = int(Lx * 100)       # Largo de la sala en X 
    Ly = int(Ly * 100)       # Largo de la sala en Y
    Dx = int(Dx * 100)       # Delta X
    Dy = int(Dy * 100)       # Delta Y
    
    # Parametros de control
    N = 250        # Densidad de la grilla del generador de geometrías
    M = 80       # Cantidad de salas a generar
    n_walls = 2    # Número de cortes en las paredes
    f_max = 180
    res_freq = 2

    # Almacenar toda la data
    rooms_new = calculation_of_geometry(Lx, Ly, Dx, Dy, N, M, n_walls)
    rooms = rooms_prev_gen + rooms_new
    mesh1 = "room_mesh_complex1"  # Crear malla con el script correspondiente
    mesh2 = "room_mesh_complex2"  # Crear malla con el script correspondiente
    mesh3 = "room_mesh_complex3"  # Crear malla con el script correspondiente
    mag_responses = []
    merit_sv_values = []
    merit_md_values = []

    # Itera y almacena los valores para encontrar el mejor cuarto
    for i in range(M):
        # Crea la malla de la geometría selecionada
        # Z = (Lz - np.random.uniform(0, Dz))/100 (vamos a asumir que la altura que se encontró es la mejor)
        create_complex_mesh(rooms[i], Lz, source_position, 80, mesh1)
        create_complex_mesh(rooms[i], Lz, source_position, 140, mesh2)
        create_complex_mesh(rooms[i], Lz, source_position, f_max, mesh3)

        # Evalua la rta en frecuencia para esa sala
        f1 = np.arange(20, 80, res_freq)
        res1 = FEM_Source_Solver_Average(f1, f'mallado/{mesh1}.msh', receptor_position)

        f2 = np.arange(80, 140, res_freq)
        res2 = FEM_Source_Solver_Average(f2, f'mallado/{mesh2}.msh', receptor_position)

        f3 = np.arange(140, f_max, res_freq)
        res3 = FEM_Source_Solver_Average(f3, f'mallado/{mesh3}.msh', receptor_position)

        res_tot = np.hstack([res1, res2, res3])
        res_tot_prom = np.sum(res_tot, axis=0) / 7

        # Calcula figuras de mérito
        sv_merit = merit_spatial_deviation(res_tot)
        md_merit = merit_magnitude_deviation(res_tot)
        
        merit_sv_values.append(sv_merit)
        merit_md_values.append(md_merit)
        mag_responses.append(res_tot_prom)
    
    merit_sv_values = np.array(merit_sv_values)
    merit_md_values = np.array(merit_md_values)
    merit_general = merit_md_values + merit_sv_values

    merit_index = list(enumerate(merit_general))
    merit_sorted = sorted(merit_index, key=lambda x: x[1])

    index_merit_sored, merit_values = zip(*merit_sorted)
    best_rooms = [rooms[i] for i in index_merit_sored]
    best_mag = [mag_responses[i] for i in index_merit_sored]
    
    return best_rooms, merit_values, best_mag

def find_complex_outline_gen4(Lx, Ly, Lz, Dx, Dy, source_position, receptor_position, rooms_prev_gen):
    # Dimensiones sala (en centímetros)
    Lx = int(Lx * 100)       # Largo de la sala en X 
    Ly = int(Ly * 100)       # Largo de la sala en Y
    Dx = int(Dx * 100)       # Delta X
    Dy = int(Dy * 100)       # Delta Y
    
    # Parametros de control
    f_max = 200
    res_freq = 1

    # Almacenar toda la data
    rooms = rooms_prev_gen  
    mesh1 = "room_mesh_complex1"  # Crear malla con el script correspondiente
    mesh2 = "room_mesh_complex2"  # Crear malla con el script correspondiente
    mesh3 = "room_mesh_complex3"  # Crear malla con el script correspondiente
    mag_responses = []
    merit_sv_values = []
    merit_md_values = []

    # Itera y almacena los valores para encontrar el mejor cuarto
    for i in range(len(rooms)):
        # Crea la malla de la geometría selecionada
        # Z = (Lz - np.random.uniform(0, Dz))/100 (vamos a asumir que la altura que se encontró es la mejor)
        create_complex_mesh(rooms[i], Lz, source_position, 80, mesh1)
        create_complex_mesh(rooms[i], Lz, source_position, 140, mesh2)
        create_complex_mesh(rooms[i], Lz, source_position, f_max, mesh3)

        # Evalua la rta en frecuencia para esa sala
        f1 = np.arange(20, 80, res_freq)
        res1 = FEM_Source_Solver_Average(f1, f'mallado/{mesh1}.msh', receptor_position)

        f2 = np.arange(80, 140, res_freq)
        res2 = FEM_Source_Solver_Average(f2, f'mallado/{mesh2}.msh', receptor_position)

        f3 = np.arange(140, f_max, res_freq)
        res3 = FEM_Source_Solver_Average(f3, f'mallado/{mesh3}.msh', receptor_position)

        res_tot = np.hstack([res1, res2, res3])
        res_tot_prom = np.sum(res_tot, axis=0) / 7

        # Calcula figuras de mérito
        sv_merit = merit_spatial_deviation(res_tot)
        md_merit = merit_magnitude_deviation(res_tot)
        
        merit_sv_values.append(sv_merit)
        merit_md_values.append(md_merit)
        mag_responses.append(res_tot_prom)
    
    merit_sv_values = np.array(merit_sv_values)
    merit_md_values = np.array(merit_md_values)
    merit_general = merit_md_values + merit_sv_values

    merit_index = list(enumerate(merit_general))
    merit_sorted = sorted(merit_index, key=lambda x: x[1])

    index_merit_sored, merit_values = zip(*merit_sorted)
    best_rooms = [rooms[i] for i in index_merit_sored]
    best_mag = [mag_responses[i] for i in index_merit_sored]
    
    return best_rooms, merit_values, best_mag

def calculate_initial(Lx, Ly, Lz, source_position, receptor_position):
    """
    Calcula el valor de mérito para la sala inicial
    """
    fmax = 200
    freq_res = 1

    mesh = "room_base_optim"
    create_simple_mesh(Lx, Ly, Lz, source_position, fmax, mesh)
    f1 = np.arange(20, fmax, freq_res)
    mag = FEM_Source_Solver_Average(f1, f'mallado/{mesh}.msh', receptor_position)
    mag_prom = np.sum(mag, axis=0) / 7
    sv_merit = merit_spatial_deviation(mag)
    md_merit = merit_magnitude_deviation(mag)
    merit = (md_merit + sv_merit, md_merit, sv_merit)
        
    return merit, mag_prom
