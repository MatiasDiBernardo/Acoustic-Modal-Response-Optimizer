from geometry_generator import calculation_of_geometry
from mesh_3D_generator import generate_mesh
from FEM import FEM_solver
from FEM_visualization import mode_shape_visulization
import os

# Estos en en centímetros
Lx = 400     # Largo de la sala en X 
Ly = 600     # Largo de la sala en Y
Lz = 220     # Alto de la sala
Dx = 80      # Delta X
Dy = 100     # Delta Y

# Parametros de control
N = 250      # Densidad de la grilla del generador de geometrías
M = 100      # Cantidad de salas a generar
n_walls = 4  # Número de cortes en las paredes
idx_modal_shape_plot = 9

# Calculo y visualización
salas = calculation_of_geometry(Lx, Ly, Dx, Dy, N, M, n_walls)

generate_mesh(salas[0], Lz/100)

FEM_solver(idx_modal_shape_plot)

mode_shape_visulization()

# Borro todos los archivos generados para poder cambiar cosas
path_files = ["model1.h5", "model1.vtu", "model1.xdmf", "room_mesh.h5", "room_mesh.vtu", "room_mesh.xdmf", "room.h5", "room.xdmf"]
for path in path_files:
    os.remove(path)