import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from outline_optim import find_best_outline
from complex_outline_optim import find_complex_random_optim, calculate_initial
from plots.graph_mag_response import general_mag_response
from plots.graph_room_outline import plot_room_iterative

# El usuario ingresa las dimensiones
Lx = 2.5       # Largo de la sala en X 
Ly = 3         # Largo de la sala en Y
Lz = 2.2       # Alto de la sala
Dx = 0.4       # Delta X
Dy = 0.6       # Delta Y
Dz = 0.15      # Delta Z

## Posiciones fuente y receptor (en metros)
source_position = (1.9, 1.0, 1.3)
receptor_position = (1.25, 1.9, 1.2)

## Tipo de optimización y paredes
optimization_type = "Fast"  # ("Fast", "Medium", "Slow")
number_walls = 3

## Caso Base: Cacula el valor inicial de la sala seleccionada
freqs = np.arange(20, 200, 1)
merit0, mag0 = calculate_initial(Lx, Ly, Lz, source_position, receptor_position)

mag0 = np.load("example/mag0.npy")
merit0 = np.load("example/merit_0.npy")

plot_room_iterative((Lx, Ly, Dx, Dy), source_position, receptor_position, [], [])
general_mag_response(freqs, mag0)

## Mejor Cuarto geometría simple
best_simple_room, spacing_simple_room, merit1_sm, mag1_sm = find_best_outline(Lx, Ly, Lz, Dx, Dy, Dz, source_position, receptor_position, optimization_type)

# Calculo nuevas posiciones en función a la mejor sala simple
Lx_new, Ly_new, Lz_new = best_simple_room
Dx_new, Dy_new = spacing_simple_room
dx_room = (Lx - Lx_new)/2
dy_room = (Ly - Ly_new)/2
new_source_pos = (source_position[0] - dx_room, source_position[1] - dy_room, source_position[2])
new_receptor_pos = (receptor_position[0] - dx_room, receptor_position[1] - dy_room, receptor_position[2])

# Recalcula marito y mag con FEM
merit1, mag1 = calculate_initial(Lx_new, Ly_new, Lz_new, new_source_pos, new_receptor_pos)

simple_room_cords = [Lx_new, Ly_new, Dx_new, Dy_new]
plot_room_iterative((Lx, Ly, Dx, Dy), source_position, receptor_position, simple_room_cords, [])
general_mag_response(freqs, mag1)

## Mejor cuarto geometría compleja
best_complex_room, merit2, mag2 = find_complex_random_optim(Lx_new, Ly_new, Lz_new, Dx_new, Dy_new, new_source_pos, new_receptor_pos, number_walls, optimization_type)

plot_room_iterative((Lx, Ly, Dx, Dy), source_position, receptor_position, simple_room_cords, best_complex_room)

if optimization_type == "Fast":
    freqs = np.arange(20, 200, 2)
general_mag_response(freqs, mag2)

print("El valor de mérito inicial es: ", merit0)
print("El mejor valor de mérito para geometría simple es: ", merit1)
print("El mejor valor de mérito para geometría compleja es: ", merit2)
