import sys
import pygame
from geometry import WallsGenerator
import numpy as np

def grid_random_sampler(points_grid, ammount_of_walls):
    """Genera puntos random para marcar las intersecciones (pardes).

    Args:
        points_grid (int): Cantidad de puntos en la grilla.
        ammount_of_walls (int): Cantidad de paredes (cortes).

    Returns:
        np.array: Vector con un 1 donde tiene que ir un vértice.
    """
    mtx = np.zeros(points_grid)
    for i in range(ammount_of_walls):
        idx_rndm = np.random.randint(0, points_grid)
        mtx[idx_rndm] = 1
    
    return mtx

def comparison(M, V):
    """Comapra si de la matrix M contiene el elemento V.

    Args:
        M (np.array[AxB]): Matriz con vectores V estaqueados. Tiene A vector de dimensión B.
        V (np.array[A]): Vector a evaluar si esta en la matriz o no.

    Returns:
        bool: Si V esta en M
    """
    for i in range(M.shape[0]):
        if np.all(M[i,:] == V):
            return True

    return False

def visualization_of_geometry():
    # Initialize Pygame
    pygame.init()

    # Screen settings
    WIDTH, HEIGHT = 600, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont(None, 36)  # None = default font, 36 = font size
    pygame.display.set_caption("Walls Generator")

    # Parámetros de la sala
    Lx = 200     # Largo de la sala en X
    Ly = 400     # Largo de la sala en Y
    Dx = 80      # Delta X
    Dy = 100     # Delta Y
    PadX = 100   # Espaciado en X (visualizacion)
    PadY = 100   # Espaciado en Y (visualización)
    N = 250      # Densidad de la grilla
    n_walls = 4  # Número de cortes en las paredes

    wall = WallsGenerator(Lx, Ly, Dx, Dy, PadX, PadY, N, n_walls, True)

    # Main loop
    # Define the event
    refresh_rate = 1000  # In ms
    UPDATE_EVERY_SECOND = pygame.USEREVENT + 1
    pygame.time.set_timer(UPDATE_EVERY_SECOND, refresh_rate)
    # Define the clock
    clock = pygame.time.Clock()
    running = True

    # Defini intial condition
    grid_random = grid_random_sampler(N, n_walls)
    label_valid = font.render("VALID :)", True, (0, 0, 255))
    label_invalid = font.render("INVALID :(", True, (255, 0, 0))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == UPDATE_EVERY_SECOND:
                wall.score += 1
                # Genera puntos random
                grid_random = grid_random_sampler(N, n_walls)
                #print(wall.normalize_coordinates())
                
                # Actualiza los puntos de inicio y final las dimensiones
                wall.update_walls()

        # Fill the screen with white background
        screen.fill((0,0,0))

        # Draw outline
        wall.room_geometry_outline(screen)

        # Genera la grilla de puntos
        wall.intial_condition_grid(screen)
        wall.dots_grid(screen, grid_random)
        
        # Gráfica las paredes
        wall.plot_walls(screen)

        # Verifica si el camino es válido
        if wall.is_valid():
            screen.blit(label_valid, (Lx//2 + 200, PadY//2))
            label_points = font.render(f"Score: {wall.score}", True, (255, 255, 255))
            screen.blit(label_points, (Lx//2 - 200, PadY//2))
        else:
            screen.blit(label_invalid, (Lx//2 + 200, PadY//2))
            label_points = font.render(f"Score: {wall.score}", True, (255, 255, 255))
            screen.blit(label_points, (Lx//2 - 200, PadY//2))

        # Update the display
        pygame.display.flip()

        # Cap the frame rate at 30 FPS
        clock.tick(30)

    # Clean up and exit
    pygame.quit()
    sys.exit()
    

def calculation_of_geometry(Lx, Ly, Dx, Dy, N, M, n_walls):
    """Genera puntos de posibles geometrías de control room.

    Args:
        Lx (int): Largo de la sala en X
        Ly (int): Largo de la sala en Y
        Dx (int): Delta X
        Dy (int): Delta Y
        N (int): Densidad de la grilla interna para generar los puntos
        M (int): Cantidad de salas
        n_walls (int): Cantidad de paredes (cortes) en un solo lado del cuarto

    Returns:
        list[(points)]: Devuelve una lista de M salas con los puntos de los vérticas de las paredes
    """

    # Parámetros de la sala
    PadX = 0   # Espaciado en X (visualizacion)
    PadY = 0   # Espaciado en Y (visualización)
    screen = 0
    
    # Salas a generar
    prev_generations = np.zeros((1, N))
    valid_rooms = []

    # Inicializa el generador
    wall = WallsGenerator(Lx, Ly, Dx, Dy, PadX, PadY, N, n_walls, False)
    
    # Genera las salas
    while len(valid_rooms) < M:
        grid_random = grid_random_sampler(N, n_walls)
        
        # Genera grilla y puntos
        wall.intial_condition_grid(screen)
        wall.dots_grid(screen, grid_random)
        
        # Si es válido y no se repite agregarlo
        if wall.is_valid():
            coord = wall.normalize_coordinates()
            repetition = comparison(prev_generations, grid_random)

            if not repetition:
                # Agrega las paredes simetricas
                coord_copy = np.copy(coord)
                for i in range(len(coord_copy), 0, -1):
                    new_x = (Lx/100) - coord_copy[i - 1, 0]
                    new_y = coord_copy[i - 1, 1]
                    coord = np.vstack([coord, (new_x, new_y)])

                valid_rooms.append(coord)
        
        # Guarda el cuarto generado
        prev_generations = np.vstack([prev_generations, grid_random])
    
    return valid_rooms
