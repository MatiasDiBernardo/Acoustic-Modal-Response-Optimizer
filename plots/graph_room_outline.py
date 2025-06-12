import matplotlib.pyplot as plt

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
