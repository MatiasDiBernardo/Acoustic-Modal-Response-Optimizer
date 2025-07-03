import numpy as np

from room.geometry_generator import calculation_of_geometry
from mesh.mesh_3D_complex import create_complex_mesh
from mesh.mesh_3D_simple import create_simple_mesh
from FEM.FEM_source import FEM_Source_Solver_Average
from aux.merit_figure import merit_magnitude_deviation, merit_spatial_deviation

def calculate_initial(Lx, Ly, Lz, source_position, receptor_position, optim_type):
    """
    Calcula el valor de mérito para la sala inicial utilizando FEM Freq solver.
    Args:
        Lx (float): Ancho en metros
        Ly (float): Largo en metros
        Lz (float): Alto en metros
        source_position (tuple(x, y, z)): Posicion de la fuente en metros
        receptor_position (tuple(x, y, z)): Posicion del receptor en metros
        optim_type: Tipo de optimización, puede ser "slow", "medium" or "fast"
    Returns:
        merti_figure (tuple(FM, MD, SD)): Figuras de mérito
        mag (array): Respuesta en frecuencia del mejor cuarto
    """
    fmax = 200
    
    if optim_type == "Fast":
        res_freq = 2
    else:
        res_freq = 1

    mesh1 = "room_base_optim1"
    mesh2 = "room_base_optim2"
    mesh3 = "room_base_optim3"
    create_simple_mesh(Lx, Ly, Lz, source_position, 80, mesh1)
    create_simple_mesh(Lx, Ly, Lz, source_position, 140, mesh2)
    create_simple_mesh(Lx, Ly, Lz, source_position, fmax, mesh3)

    # Evalua la rta en frecuencia para esa sala
    f1 = np.arange(20, 80, res_freq)
    res1 = FEM_Source_Solver_Average(f1, f'mallado/{mesh1}.msh', receptor_position)

    f2 = np.arange(80, 140, res_freq)
    res2 = FEM_Source_Solver_Average(f2, f'mallado/{mesh2}.msh', receptor_position)

    f3 = np.arange(140, fmax, res_freq)
    res3 = FEM_Source_Solver_Average(f3, f'mallado/{mesh3}.msh', receptor_position)

    res_tot = np.hstack([res1, res2, res3])
    res_tot_prom = np.sum(res_tot, axis=0) / 7

    # Calcula figuras de mérito
    sv_merit = merit_spatial_deviation(res_tot)
    md_merit = merit_magnitude_deviation(res_tot)

    merit = (md_merit + sv_merit, md_merit, sv_merit)
        
    return merit, res_tot_prom

def find_complex_random_optim(Lx, Ly, Lz, Dx, Dy, source_position, receptor_position, n_walls, optim_type):
    """Encuentra el mejor cuarto paralelepipedo en base a las dimensiones del cuarto y un
    margen para hacer pruebas de opimización

    Args:
        Lx (float): Ancho en metros
        Ly (float): Largo en metros
        Lz (float): Alto en metros
        Dx (float): Espaciado del ancho en metros
        Dy (float): Espaciado del largo en metros
        source_position (tuple(x, y, z)): Posicion de la fuente en metros
        receptor_position (tuple(x, y, z)): Posicion del receptor en metros
        n_walls (int): Numero de paredes
        optim_type: Tipo de optimización, puede ser "slow", "medium" or "fast"
    Returns:
        best_room (np.array): Coordenadas del mejor cuarto encontrado 
        merti_figure (tuple(FM, MD, SD)): Figuras de mérito
        mag (array): Respuesta en frecuencia del mejor cuarto
    """
    
    # Control de la optimización
    denisty_grid = 250           # Grilla del generador de geometría
    n_walls_vert = n_walls - 1   # Numero de paredes como vertices
    
    # Pasa a centrimetros para generador de geometría
    Lx = int(Lx * 100)       # Largo de la sala en X 
    Ly = int(Ly * 100)       # Largo de la sala en Y
    Dx = int(Dx * 100)       # Delta X
    Dy = int(Dy * 100)       # Delta Y
    
    # Cantidad de salas a generar
    if optim_type == "Fast":
        rooms_iterate = 5
        res_freq = 2
    if optim_type == "Medium":
        rooms_iterate = 100 
        res_freq = 1
    if optim_type == "Slow":
        rooms_iterate = 200
        res_freq = 1

    # Agregar control dentro de gem generation de que si tarda mucho en contrar los M salas corte
    rooms = calculation_of_geometry(Lx, Ly, Dx, Dy, denisty_grid, rooms_iterate, n_walls_vert)

    mesh1 = "room_mesh_complex1"  # Crear malla con el script correspondiente
    mesh2 = "room_mesh_complex2"  # Crear malla con el script correspondiente
    mesh3 = "room_mesh_complex3"  # Crear malla con el script correspondiente
    mag_responses = []
    merit_sv_values = []
    merit_md_values = []

    # Itera y almacena los valores para encontrar el mejor cuarto
    for i in range(len(rooms)):
        print("Vamos por el room: ", i)

        # Crea la malla de la geometría selecionada
        # Z = (Lz - np.random.uniform(0, Dz))/100 (saco  esto porque se supone que ya tenemos mejor Z)
        create_complex_mesh(rooms[i], Lz, source_position, 80, mesh1)
        create_complex_mesh(rooms[i], Lz, source_position, 140, mesh2)
        create_complex_mesh(rooms[i], Lz, source_position, 200, mesh3)

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

    merit_sv_values = np.array(merit_sv_values)
    merit_md_values = np.array(merit_md_values)
    merit_general = merit_md_values + merit_sv_values

    idx_best_room = np.argmin(merit_general)
    
    merit = (merit_md_values[idx_best_room] + merit_sv_values[idx_best_room], merit_md_values[idx_best_room], merit_sv_values[idx_best_room])
    best_mag = mag_responses[idx_best_room]
    best_room = rooms[idx_best_room]
    
    return best_room, merit, best_mag

def solve_complex_geom(Lx, Ly, Lz, Dx, Dy, source_position, receptor_position, room, res_freq=1):
    Lx = int(Lx * 100)       # Largo de la sala en X 
    Ly = int(Ly * 100)       # Largo de la sala en Y
    Dx = int(Dx * 100)       # Delta X
    Dy = int(Dy * 100)       # Delta Y
    
    # Parametros de control
    f_max = 200

    mesh1 = "room_mesh_complex1"  # Crear malla con el script correspondiente
    mesh2 = "room_mesh_complex2"  # Crear malla con el script correspondiente
    mesh3 = "room_mesh_complex3"  # Crear malla con el script correspondiente

    create_complex_mesh(room, Lz, source_position, 80, mesh1)
    create_complex_mesh(room, Lz, source_position, 140, mesh2)
    create_complex_mesh(room, Lz, source_position, f_max, mesh3)

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
    merit = (md_merit + sv_merit, md_merit, sv_merit)
    
    return merit, res_tot_prom
