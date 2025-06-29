
import  sys
import os 
import numpy as np 

# Obtiene la ruta del directorio del script actual (tests/)
dir_script = os.path.dirname(os.path.abspath(__file__))
# Obtiene la ruta raíz del proyecto (el directorio padre de tests/)
dir_raiz = os.path.dirname(dir_script)
# Añadir la ruta raíz al sys.path si aún no está
if dir_raiz not in sys.path:
    sys.path.append(dir_raiz)
print(f"Ruta raíz del proyecto añadida al path: {dir_raiz}")

from mesh.mesh_3D_generator import generate_mesh_parallelepiped
from FEM.FEM_source import generar_campo_presion_paraview

#Frecuencia maxima de simulacion
f_max = 200
f_min = 20

#Dimensiones del Recinto
Lx = 3
Ly = 4
Lz = 2.5
L_floor = [Lx,Ly]
dimensions = [Lx, Ly, Lz]

#Posicion de Emisor 
Sx = 2
Sy = 0.5
Sz = 1.5
Source_position = [Sx, Sy, Sz]

#Posicion de Receptor 
Rx = 1.5
Ry = 1.3
Rz = 1.2 
Receptor_position = [Rx, Ry, Rz]

ruta_relativa_malla = os.path.join("mallado", "malla_gruesa_TD.msh")
ruta_absoluta_malla = os.path.abspath(ruta_relativa_malla)

malla1 = "malla_gruesa_TD.msh"
output_filename = "FD_output.xdmf"
generate_mesh_parallelepiped(L_floor, Lz, Source_position, f_max, elements_per_wavelenght=7 ,name = malla1)

frecs = np.array([42.88, 57.17, 68.60, 71.45, 80.90, 85.75, 89.29, 99.05, 103.06, 109.81, 114.33, 122.11, 123.79, 128.63, 133.26, 137.20, 140.05, 140.75, 142.86, 143.72])



fd_f_response = generar_campo_presion_paraview(ruta_absoluta_malla, output_filename , frecs, degree =2 )