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
Dx = 50        # Delta X
Dy = 80        # Delta Y
Dz = 10        # Delta Z

# Posiciones fuente y receptor (en metros)
source_position = (1.9, 1.0, 1.3)
receptor_position = (1.25, 1.9, 1.2)

# Graficos de rooms por iteraciones
room_100 = [[0.625     ,0.        ],
            [0.        ,0.33      ],
            [0.11      ,1.54      ],
            [0.10416667,3.        ],
            [2.39583333,3.        ],
            [2.39      ,1.54      ],
            [2.5       ,0.33      ],
            [1.875     ,0.        ]]

room_200 = [[0.3125    ,0.        ],
            [0.11      ,0.88      ],
            [0.22      ,1.98      ],
            [0.83333333,3.        ],
            [1.66666667,3.        ],
            [2.28      ,1.98      ],
            [2.39      ,0.88      ],
            [2.1875    ,0.        ]]

room_300 = [[0.41666667,0.        ],
            [0.        ,0.77      ],
            [0.11      ,0.88      ],
            [0.625     ,3.        ],
            [1.875     ,3.        ],
            [2.39      ,0.88      ],
            [2.5       ,0.77      ],
            [2.08333333,0.        ]]

# Para la comparación entre paredes
paredes_2 = [[0.8333333,0.        ],
            [0.11      ,0.88      ],
            [0.52083333,3.        ],
            [1.97916667,3.        ],
            [2.39      ,0.88      ],
            [1.66666667,0.        ]]

paredes_4 = [[0.1041666,0.        ],
            [0.11      ,1.43      ],
            [0.22      ,2.09      ],
            [0.66      ,2.31      ],
            [0.83333333,3.        ],
            [1.66666667,3.        ],
            [1.84      ,2.31      ],
            [2.28      ,2.09      ],
            [2.39      ,1.43      ],
            [2.39583333,0.        ]]

paredes_5 = [[0.83333333,0.        ],
            [0.88      ,0.11      ],
            [0.        ,1.32      ],
            [0.33      ,1.87      ],
            [0.44      ,2.2       ],
            [0.10416667,3.        ],
            [2.39583333,3.        ],
            [2.06      ,2.2       ],
            [2.17      ,1.87      ],
            [2.5       ,1.32      ],
            [1.62      ,0.11      ],
            [1.66666667,0.        ]]

plot_room_outline(Lx, Ly, Dx, Dy, source_position, receptor_position, paredes_2, "Mejor resultado 2 paredes")
plot_room_outline(Lx, Ly, Dx, Dy, source_position, receptor_position, paredes_4, "Mejor resultado 4 paredes")
plot_room_outline(Lx, Ly, Dx, Dy, source_position, receptor_position, paredes_5, "Mejor resultado 5 paredes")