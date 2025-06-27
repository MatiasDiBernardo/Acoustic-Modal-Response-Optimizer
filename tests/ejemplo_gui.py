import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plots.graph_room_outline import plot_room_iterative
from plots.graph_mag_response import general_mag_response

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

# Una vez que el usuario le da a optimizar, primero se valida si los inputs
# son correctos, y si se cumple comienza la iteración:
# En cada caso, se grafica magnitud y sala. Y se muestra en pantalla el valor de merito
freqs = np.arange(20, 200, 1)

## Caso Base: Cacula el valor inicial de la sala seleccionada
mag0 = np.load("example/mag0.npy")
merit0 = np.load("example/merit_0.npy")

print("El valor de mérito inicial es: ", merit0)
plot_room_iterative((Lx, Ly, Dx, Dy), source_position, receptor_position, [], [])
general_mag_response(freqs, mag0)

## Mejor Cuarto geometría simple
best_simple_room = np.load("example/best_dimensiones_g1.npy")
simple_room_cords = [best_simple_room[0], best_simple_room[1], best_simple_room[3], best_simple_room[4]]

mag1 = np.load("example/mag_g1.npy")
merit1 = np.load("example/merit_g1.npy")

print("El mejor valor de mérito para geometría simple es: ", merit1)
plot_room_iterative((Lx, Ly, Dx, Dy), source_position, receptor_position, simple_room_cords, [])
general_mag_response(freqs, mag1)

## Mejor cuarto geometría compleja
best_rooms_complex = np.load("example/rooms_g4.npy")

# Acá se podría elegir para mostrar las 3 mejores opciones cambiando el indice de choice_best_room
# Están ordenados, aunque no se verifica que sean geometría significativamente diferentes
choice_best_room = 0  # Valor a camibar para ver otra geometría
complex_room_cords = best_rooms_complex[choice_best_room]
mag4 = np.load("example/mag_g4.npy")
mag_display = mag4[choice_best_room]
merit4 = np.load("example/merits_g4.npy")
merit_display = merit4[choice_best_room]

print("El mejor valor de mérito para geometría compleja es: ", merit_display)
plot_room_iterative((Lx, Ly, Dx, Dy), source_position, receptor_position, simple_room_cords, complex_room_cords)
general_mag_response(freqs, mag_display)

