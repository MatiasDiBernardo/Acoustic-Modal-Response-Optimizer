import  sys
import os
import numpy as np
# --- USAR ESTE BLOQUE EN SU LUGAR ---
# Obtiene la ruta del directorio donde se encuentra este script (/.../notebooks)
dir_script = os.path.dirname(os.path.abspath(__file__))
# Sube un nivel para obtener la raíz del proyecto (/.../Acoustic-Modal-Response-Optimizer)
dir_raiz = os.path.dirname(dir_script)
# Añadir la ruta raíz del proyecto al sys.path si aún no está
if dir_raiz not in sys.path:
    sys.path.append(dir_raiz)
print(f"Ruta raíz del proyecto añadida al path: {dir_raiz}")
from mesh.mesh_3D_generator import generate_mesh_parallelepiped
from mesh.mesh_3D_complex import create_complex_mesh

#Frecuencia maxima de simulacion
f_max = 200
f_min = 20

#Dimensiones del Recinto
floor_points = np.array([
    [0, 0.2],
    [1, 0],
    [2.5, 0],
    [4, 0.2],
    [4, 2.8],
    [2.5, 3],
    [1, 3],
    [0, 2.8]
])

height = 2



f_max = 150

#Posicion de Emisor 
Sx = .3
Sy = 1.5
Sz = 1.2
source_pos = [Sx, Sy, Sz]

nombre = "foto"
#Generamos la malla
create_complex_mesh(floor_points, height, source_pos, f_max, nombre, verbose=True)