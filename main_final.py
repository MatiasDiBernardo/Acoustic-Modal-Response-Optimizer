from plots.graph_room_outline import plot_outline_simple, plot_room_outline
import time

from outline_optim import find_best_outline
from complex_outline_optim import find_complex_outline_gen2, find_complex_outline_gen3, find_complex_outline_gen4, calculate_initial

# Inputs usuario
## Inputs dimensiones
Lx = 2.5       # Largo de la sala en X 
Ly = 3         # Largo de la sala en Y
Lz = 2.2       # Alto de la sala
Dx = 0.4       # Delta X
Dy = 0.6       # Delta Y
Dz = 0.15      # Delta Z

## Posiciones fuente y receptor (en metros)
source_position = (1.9, 1.0, 1.3)
receptor_position = (1.25, 1.9, 1.2)

## Número de cortes en las paredes
n_walls = 2    

## Tipo de optimización (slow, medium, fast)
optim_type = "fast"

if optim_type == "fast":
    salas_g1 = 500
    salas_g2 = 150     # Cantidad de salas para iterar
    next_gen_g3 = 10   # Cantidad de salas que pasan de la iteración anterior
    salas_g3 = 10      # Cantidad de salas random para agregar
    next_gen_g4 = 3    # Cantidad de salas que pasan de la iteración anterior
    mut_ammount = 2    # Cantidad de mutaciones por sala

if optim_type == "medium":
    salas_g1 = 1000
    salas_g2 = 300     # Cantidad de salas para iterar
    next_gen_g3 = 20   # Cantidad de salas que pasan de la iteración anterior
    salas_g3 = 20      # Cantidad de salas random para agregar
    next_gen_g4 = 3    # Cantidad de salas que pasan de la iteración anterior
    mut_ammount = 4    # Cantidad de mutaciones por sala

if optim_type == "slow":
    salas_g1 = 5000
    salas_g2 = 500     # Cantidad de salas para iterar
    next_gen_g3 = 50   # Cantidad de salas que pasan de la iteración anterior
    salas_g3 = 50      # Cantidad de salas random para agregar
    next_gen_g4 = 3    # Cantidad de salas que pasan de la iteración anterior
    mut_ammount = 6    # Cantidad de mutaciones por sala

## 0) Evalúa condiciones iniciales
start0 = time.time()
merit_0, mag0 = calculate_initial(Lx, Ly, Lz, source_position, receptor_position)
time0 = time.time() - start0

## 1) Mejores dimensiones con geometrías simples
start1 = time.time()
final_best_room, best_room_spacing, merit_gen1, mag_best = find_best_outline(Lx, Ly, Lz, Dx, Dy, Dz, source_position, receptor_position, salas_g1)
time1 = time.time() - start1

## 2) Geometría compleja partiendo de mejor geometría simple
start2 = time.time()
Lx_new, Ly_new, Lz_new = final_best_room
Dx_new, Dy_new = best_room_spacing
dx_room = (Lx - Lx_new)/2
dy_room = (Ly - Ly_new)/2
# Esto sería como una translación del problema al subcuadrado mas chico
new_source_pos = (source_position[0] - dx_room, source_position[1] - dy_room, source_position[2])
new_receptor_pos = (receptor_position[0] - dx_room, receptor_position[1] - dy_room, receptor_position[2])

best_rooms_g2, merits_g2, mags_g2 = find_complex_outline_gen2(Lx_new, Ly_new, Lz_new, Dx_new, Dy_new, new_source_pos, new_receptor_pos, n_walls, salas_g2) 
time2 = time.time() - start2

## 3) Geometría compleja mas ancho de banda
start3 = time.time()
best_rooms_g3, merits_g3, mag_g3 = find_complex_outline_gen3(Lx_new, Ly_new, Lz_new, Dx_new, Dy_new, new_source_pos, new_receptor_pos, n_walls, best_rooms_g2[:next_gen_g3], salas_g3) 
time3 = time.time() - start3

## 4) Geometría compleja mas ancho de banda y resolución (se introducen mutaciones)
start4 = time.time()
best_rooms_g4, merits_g4, mag_g4 = find_complex_outline_gen4(Lx_new, Ly_new, Lz_new, Dx_new, Dy_new, new_source_pos, new_receptor_pos, best_rooms_g3[:next_gen_g4], mut_ammount) 
time4 = time.time() - start4

print("El valor original de mérito es: ", merit_0)
print("Tiempo de ejecución en minutos fue de: ", time0/60)
print("...................")
print("Gen 1(con modal sum y geometrías simples)")
print("El mejor mérito es: ", merit_gen1)
print("Tiempo de ejecución en minutos fue de: ", time1/60)
print("...................")
print("Gen 2 (con FEM Source hasta 150 Hz | res=2)")
print("El mejor mérito es ", merits_g2[0])
print("Tiempo de ejecución en minutos fue de: ", time2/60)
print("...................")
print("Gen 3 (con FEM Source hasta 180 Hz | res=2)")
print("El mejor mérito es ", merits_g3[0])
print("Tiempo de ejecución en minutos fue de: ", time3/60)
print("...................")
print("Gen 4 (con FEM Source hasta 200 Hz | res=1)")
print("El mejor mérito es ", merits_g4[0])
print("Tiempo de ejecución en minutos fue de: ", time4/60)
