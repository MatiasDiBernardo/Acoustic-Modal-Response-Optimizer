from plots.graph_room_outline import plot_outline_simple

from outline_optim import find_best_outline

# Inputs usuario
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

# Inicio del ciclo de iteración
## 0) Evalúa condiciones iniciales

## 1) Mejores dimensiones con geometrías simples
final_best_room, best_room_spacing, merit, mag_best = find_best_outline(Lx, Ly, Lz, Dx, Dy, Dz, source_position, receptor_position)

## 2) Geometría compleja partiendo de mejor geometría simple
    
print("Espaciado en dX: ", best_room_spacing[0])
print("Espaciado en dY: ", best_room_spacing[1])
print(merit)

plot_outline_simple(Lx, Ly, Dx, Dy, final_best_room[0], final_best_room[1])
