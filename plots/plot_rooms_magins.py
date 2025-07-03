import matplotlib.pyplot as plt

def plot_room_outline(Lx_cm, Ly_cm, Dx_cm, Dy_cm, source_position, receptor_position, floor_cords, name):
    # Convertir a metros
    Lx = Lx_cm / 100
    Ly = Ly_cm / 100
    Dx = Dx_cm / 100
    Dy = Dy_cm / 100

    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(4, 6))

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
    # ax.legend(loc='upper right')
    ax.set_title(name)

    plt.show()

# Definimos el cuarto
Lx = 250       # Largo de la sala en X 
Ly = 300       # Largo de la sala en Y
Lz = 220       # Alto de la sala
Dx = 10        # Delta X
Dy = 20        # Delta Y

Dx1 = 20        # Delta X
Dy2 = 40        # Delta Y
# Posiciones fuente y receptor (en metros)
source_position = (1.9, 1.0, 1.3)
receptor_position = (1.25, 1.9, 1.2)

margin1 = [[0.20833333 ,0.        ],
            [0.78      ,0.06      ],
            [0.        ,0.18      ],
            [0.10416667,3.        ],
            [2.39583333,3.        ],
            [2.5       ,0.18      ],
            [1.72      ,0.06      ],
            [2.29166667,0.        ]]

margin2 = [[0.625      ,0.        ],
            [0.        ,0.56      ],
            [0.08      ,2.48      ],
            [0.41666667,3.        ],
            [2.08333333,3.        ],
            [2.42      ,2.48      ],
            [2.5       ,0.56      ],
            [1.875     ,0.        ]]

plot_room_outline(Lx, Ly, Dx, Dy, source_position, receptor_position, margin1, "Mejor resultado con margenes chicos")
plot_room_outline(Lx, Ly, Dx1, Dy2, source_position, receptor_position, margin2, "Mejor resultado con márgenes medios")