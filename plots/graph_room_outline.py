import matplotlib.pyplot as plt
import numpy as np

def plot_room_outline(Lx_cm, Ly_cm, Dx_cm, Dy_cm, source_position, receptor_position, floor_cords, name):
    # Convertir a metros
    Lx = Lx_cm / 100
    Ly = Ly_cm / 100
    Dx = Dx_cm / 100
    Dy = Dy_cm / 100

    # Crear la figura y el eje
    fig, ax = plt.subplots()

    # Contorno de la sala completa
    outer_x = [0, Lx, Lx, 0, 0]
    outer_y = [0, 0, Ly, Ly, 0]
    ax.plot(outer_x, outer_y, label="Sala completa")

    # Contorno de la sala reducida (centrada)
    inner_x = [0 + Dx, Lx - Dx, Lx - Dx, 0 + Dx, 0 + Dx]
    inner_y = [0 + Dy, 0 + Dy, Ly - Dy, Ly - Dy, 0 + Dy]
    ax.plot(inner_x, inner_y, label="Sala reducida")

    # Posicionar fuente y receptor
    ax.scatter(source_position[0], source_position[1], marker='o', label="Fuente")
    ax.scatter(receptor_position[0], receptor_position[1], marker='s', label="Receptor")

    # Contorno de la geometría compleja
    if len(floor_cords) != 0:
        geo_x, geo_y = zip(*floor_cords)
        # Cerrar el polígono
        geo_x = list(geo_x) + [geo_x[0]]
        geo_y = list(geo_y) + [geo_y[0]]
        ax.plot(geo_x, geo_y, linestyle='--', label="Geometría compleja")

    # Ajustes de gráfico
    #ax.set_aspect('equal', 'box')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc='upper right')
    ax.set_title(name)

    plt.show()

def plot_outline_simple(Lx, Ly, Dx, Dy, Lx2, Ly2):
    # Crear la figura y el eje
    fig, ax = plt.subplots()

    # Contorno de la sala completa
    outer_x = [0, Lx, Lx, 0, 0]
    outer_y = [0, 0, Ly, Ly, 0]
    ax.plot(outer_x, outer_y, label="Sala completa")

    # Contorno de la sala reducida (centrada)
    inner_x = [0 + Dx, Lx - Dx, Lx - Dx, 0 + Dx, 0 + Dx]
    inner_y = [0 + Dy, 0 + Dy, Ly - Dy, Ly - Dy, 0 + Dy]
    ax.plot(inner_x, inner_y, label="Sala reducida límite")

    # Contorno sala optima
    dx_room = (Lx - Lx2)/2
    dy_room = (Ly - Ly2)/2

    outer_x2 = np.array([0, Lx2, Lx2, 0, 0]) + dx_room
    outer_y2 = np.array([0, 0, Ly2, Ly2, 0]) + dy_room
    ax.plot(outer_x2, outer_y2, label="Sala optimizada")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc='upper right')
    plt.show()

def plot_multiple_rooms(Lx_cm, Ly_cm, Dx_cm, Dy_cm, source_position, receptor_position, floor_cords_multi, name):
    # Convertir a metros
    Lx = Lx_cm / 100
    Ly = Ly_cm / 100
    Dx = Dx_cm / 100
    Dy = Dy_cm / 100

    # Crear la figura y el eje
    fig, ax = plt.subplots()

    # Contorno de la sala reducida (centrada)
    inner_x = [0 + Dx, Lx - Dx, Lx - Dx, 0 + Dx, 0 + Dx]
    inner_y = [0 + Dy, 0 + Dy, Ly - Dy, Ly - Dy, 0 + Dy]
    ax.plot(inner_x, inner_y, label="Sala reducida")

    # Posicionar fuente y receptor
    ax.scatter(source_position[0], source_position[1], marker='o', label="Fuente")
    ax.scatter(receptor_position[0], receptor_position[1], marker='s', label="Receptor")

    # Contorno de la geometría compleja
    for floor_cords in floor_cords_multi:
        i = 0
        geo_x, geo_y = zip(*floor_cords)
        # Cerrar el polígono
        geo_x = list(geo_x) + [geo_x[0]]
        geo_y = list(geo_y) + [geo_y[0]]
        ax.plot(geo_x, geo_y, linestyle='--', label=f"Room {i}")
        i += 1

    # Ajustes de gráfico
    #ax.set_aspect('equal', 'box')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc='upper right')
    ax.set_title(name)

    plt.show()

def plot_room_iterative(original_room, source_position, receptor_position, simple_room_optim, complex_room_optim):
    # Convertir a metros
    Lx, Ly, Dx, Dy = original_room

    # Crear la figura y el eje
    fig, ax = plt.subplots()

    # Contorno de la sala completa
    outer_x = [0, Lx, Lx, 0, 0]
    outer_y = [0, 0, Ly, Ly, 0]
    ax.plot(outer_x, outer_y, label="Sala completa")

    # Contorno de la sala reducida (centrada)
    inner_x = [0 + Dx, Lx - Dx, Lx - Dx, 0 + Dx, 0 + Dx]
    inner_y = [0 + Dy, 0 + Dy, Ly - Dy, Ly - Dy, 0 + Dy]
    ax.plot(inner_x, inner_y, label="Sala reducida")

    # Posicionar fuente y receptor
    ax.scatter(source_position[0], source_position[1], marker='o', label="Fuente")
    ax.scatter(receptor_position[0], receptor_position[1], marker='s', label="Receptor")
    
    # Contorno mejor sala simple
    if len(simple_room_optim) != 0:
        Lx_new, Ly_new, Dx_new, Dy_new = simple_room_optim

        # Contorno de la mejor sala simple
        outer_x = [0, Lx_new, Lx_new, 0, 0]
        outer_y = [0, 0, Ly_new, Ly_new, 0]
        ax.plot(outer_x, outer_y, label="Sala completa")


    # Contorno de la geometría compleja
    if len(complex_room_optim) != 0 and len(simple_room_optim) != 0:
        geo_x, geo_y = zip(*complex_room_optim)
        # Cerrar el polígono
        geo_x = list(geo_x) + [geo_x[0]]
        geo_y = list(geo_y) + [geo_y[0]]
        geo_x = np.array(geo_x) + Dx_new
        geo_y = np.array(geo_y) + Dy_new
        ax.plot(geo_x, geo_y, linestyle='--', label="Geometría compleja")

    # Ajustes de gráfico
    #ax.set_aspect('equal', 'box')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc='upper right')

    plt.show()