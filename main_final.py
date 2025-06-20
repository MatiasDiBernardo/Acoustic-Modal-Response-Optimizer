import numpy as np
import matplotlib.pyplot as plt

from room.geometry_generator import calculation_of_geometry, calculation_of_geometry_simple
from mesh.mesh_3D_generator import generate_mesh_parallelepiped
from mesh.mesh_3D_simple import create_simple_mesh
from FEM.FEM_source import FEM_Source_Solver_Average
from aux.merit_figure import merit_spatial_deviation, merit_magnitude_deviation

# Inputs usuario
## Dimensiones sala (en centímetros)
Lx = 250       # Largo de la sala en X 
Ly = 300       # Largo de la sala en Y
Lz = 220       # Alto de la sala
Dx = 50        # Delta X
Dy = 80        # Delta Y
Dz = 10        # Delta Z

## Posiciones fuente y receptor (en metros)
source_position = (1.9, 1.0, 1.3)
receptor_position = (1.25, 1.9, 1.2)

## Número de cortes en las paredes
n_walls = 2    

# Iteración inicial (geometría cuadrada con mode summ)
