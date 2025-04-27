import sys
import pygame
from geometry import WallsGenerator
import numpy as np

def visualization_of_geometry():
    # Initialize Pygame
    pygame.init()

    # Screen settings
    WIDTH, HEIGHT = 600, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont(None, 36)  # None = default font, 36 = font size
    pygame.display.set_caption("Walls Generator")

    # Parámetros de la sala
    Lx = 400     # Largo de la sala en X
    Ly = 600     # Largo de la sala en Y
    Dx = 80      # Delta X
    Dy = 100     # Delta Y
    PadX = 100   # Espaciado en X (visualizacion)
    PadY = 100   # Espaciado en Y (visualización)
    N = 250      # Densidad de la grilla
    n_walls = 4  # Número de cortes en las paredes

    wall = WallsGenerator(Lx, Ly, Dx, Dy, PadX, PadY, N, n_walls)

    # Main loop
    # Define the event
    refresh_rate = 1000  # In ms
    UPDATE_EVERY_SECOND = pygame.USEREVENT + 1
    pygame.time.set_timer(UPDATE_EVERY_SECOND, refresh_rate)
    # Define the clock
    clock = pygame.time.Clock()
    running = True

    def grid_random_sampler(points_grid, ammount_of_walls):
        mtx = np.zeros(points_grid)
        for i in range(ammount_of_walls):
            idx_rndm = np.random.randint(0, points_grid)
            mtx[idx_rndm] = 1
        
        return mtx

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

def calculation_of_geometry():

    # Parámetros de la sala
    Lx = 400     # Largo de la sala en X
    Ly = 600     # Largo de la sala en Y
    Dx = 80      # Delta X
    Dy = 100     # Delta Y
    PadX = 100   # Espaciado en X (visualizacion)
    PadY = 100   # Espaciado en Y (visualización)
    N = 250      # Densidad de la grilla
    n_walls = 4  # Número de cortes en las paredes

    wall = WallsGenerator(Lx, Ly, Dx, Dy, PadX, PadY, N, n_walls)