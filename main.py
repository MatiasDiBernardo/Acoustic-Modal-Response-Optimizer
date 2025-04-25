import sys
import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont(None, 36)  # None = default font, 36 = font size
pygame.display.set_caption("Walls Generator")

# Colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Parámetros de la sala
Lx = 400
Ly = 600
Dx = 80
Dy = 100
PadX = 100
PadY = 100
M = 250

def room_geometry_exterior(screen, Lx, Ly, PadX, PadY):
    # Posiciones borde exterior
    top_left = (PadX, PadY)
    top_right = (PadX + Lx, PadY)
    bottom_left = (PadX, PadY + Ly)
    bottom_rignt = (PadX + Lx, PadY + Ly)
    
    # Borde exterior
    line_width = 2
    pygame.draw.line(screen, WHITE, top_left, top_right, line_width)
    pygame.draw.line(screen, WHITE, top_right, bottom_rignt, line_width)
    pygame.draw.line(screen, WHITE, bottom_rignt, bottom_left, line_width)
    pygame.draw.line(screen, WHITE, bottom_left, top_left, line_width)
    
def room_geometry_interior(screen, Lx, Ly, Dx, Dy, PadX, PadY):
    # Posiciones borde interior
    top_left = (PadX + Dx, PadY + Dy)
    rect_width = Lx - 2*Dx
    rect_height = Ly - 2*Dy
    
    # Borde interior
    line_width = 2
    rect = pygame.Rect(
        top_left[0],
        top_left[1],
        rect_width,
        rect_height
    )
    
    pygame.draw.rect(screen, WHITE, rect, line_width)

    return rect

def intial_wall_random_samples(points_initial):
    arr = np.zeros(points_initial)
    idx_rndm = np.random.randint(0, points_initial)
    arr[idx_rndm] = 1
    
    return arr

def grid_random_sampler(points_grid, ammount_of_walls):
    mtx = np.zeros(points_grid)
    for i in range(ammount_of_walls):
        idx_rndm = np.random.randint(0, points_grid)
        mtx[idx_rndm] = 1
    
    return mtx

def intial_condition_grid(screen, Lx, Ly, PadX, PadY, initial_wall, finish_wall):
    spx = PadX  # Start Position X
    spy = PadY  # Start Position Y
    N = len(initial_wall)
    dx = (Lx//2) / N
    pos_inicial = []
    pos_final = []
    thickness = 2
    
    # Grid pared inicial (esta la posibilidad de hacer len(initial) - 1 para dejar el nodo libre)
    for i in range(len(initial_wall)):
        pos = (spx + dx * i, spy)
        if initial_wall[i] == 1:
            color = (255, 0, 0) 
            pos_inicial.append(pos)
        else:
            color = (0, 0, 255) 
        pygame.draw.circle(screen, color, pos, thickness)

    # Grid pared final
    for i in range(len(finish_wall)):
        pos = (spx + dx * i, spy + Ly)
        if finish_wall[i] == 1:
            color = (255, 0, 0) 
            pos_final.append(pos)
        else:
            color = (0, 0, 255) 
        pygame.draw.circle(screen, color, pos, thickness)
    
    return pos_inicial, pos_final

def dots_grid(screen, Lx, Ly, Dx, Dy, PadX, PadY, M, grid_points):
    # Area calc
    top_left = (PadX, PadY)

    top_area = Lx//2 * Dy
    middle_area = Dx * (Ly - 2 * Dy)
    bottom_area = Lx//2 * Dy
    
    area_total = top_area + middle_area + bottom_area

    # Define setp: Si propongo que dx == dy
    dx = 1
    for i in range(1, Lx//2):
        area_dots = M * dx**2
        if area_dots > area_total:
            break
        dx = i 
    
    # Iterar 
    thickness = 2
    row = 0
    col = 0

    # Intial position and conditions
    spx = top_left[0]  # Start position X
    spy = top_left[1] + dx #  Start position Y
    
    # Margenes
    x_margin1 = PadX + Lx//2
    y_margin1 = PadY + Dy

    x_margin2 = PadX + Dx
    y_margin2 = PadY + Ly - Dy
    
    x_margin3 = PadX + Lx//2
    y_margin3 = PadY + Ly

    # Crea la grilla
    value_walls = []
    for i in range(M):
        if grid_points[i] == 1:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        
        # Primera área
        if spx + dx * row >= x_margin1:
            col += 1
            row = 0
            pos = (spx + dx * row, spy + dx * col)
            row += 0
            pygame.draw.circle(screen, color, pos, thickness)
            if grid_points[i] == 1:
                value_walls.append(pos)
            continue

        if spx + dx * row < x_margin1 and spy + dx * col < y_margin1:  
            pos = (spx + dx * row, spy + dx * col)
            row += 1
            pygame.draw.circle(screen, color, pos, thickness)
            if grid_points[i] == 1:
                value_walls.append(pos)
            continue
        
        # Segunda área
        if spx + dx * row >= x_margin2 and spy + dx * col < y_margin2:
            col += 1
            row = 0
            pos = (spx + dx * row, spy + dx * col)
            row += 1
            pygame.draw.circle(screen, color, pos, thickness)
            if grid_points[i] == 1:
                value_walls.append(pos)
            continue

        if spx + dx * row < x_margin2 and spy + dx * col < y_margin2:  
            pos = (spx + dx * row, spy + dx * col)
            row += 1
            pygame.draw.circle(screen, color, pos, thickness)
            if grid_points[i] == 1:
                value_walls.append(pos)
            continue

        # Tercera área
        if spx + dx * row >= x_margin3:
            col += 1
            row = 0
            pos = (spx + dx * row, spy + dx * col)
            row += 1
            pygame.draw.circle(screen, color, pos, thickness)
            if grid_points[i] == 1:
                value_walls.append(pos)
            continue

        if spx + dx * row < x_margin3 and spy + dx * col < y_margin3:  
            pos = (spx + dx * row, spy + dx * col)
            row += 1
            pygame.draw.circle(screen, color, pos, thickness)
            if grid_points[i] == 1:
                value_walls.append(pos)
            continue

    return value_walls

def plot_walls(screen, walls_pos):
    color = (255, 255, 0)
    line_width = 4
    for i in range(len(walls_pos) - 1):
        pygame.draw.line(screen, color, walls_pos[i], walls_pos[i + 1], line_width)

# Main loop
# Define the event
refresh_rate = 500  # In ms
UPDATE_EVERY_SECOND = pygame.USEREVENT + 1
pygame.time.set_timer(UPDATE_EVERY_SECOND, refresh_rate)
# Define the clock
clock = pygame.time.Clock()
running = True

# Defini intial condition
num_walls = 6
intial_points_walls = 12
grid_random = grid_random_sampler(M, num_walls)
initial_wall = intial_wall_random_samples(intial_points_walls)
finish_wall = intial_wall_random_samples(intial_points_walls)
label_valid = font.render("VALID :)", True, (0, 0, 255))
label_invalid = font.render("INVALID :(", True, (255, 0, 0))
points = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == UPDATE_EVERY_SECOND:
            grid_random = grid_random_sampler(M, num_walls)
            initial_wall = intial_wall_random_samples(intial_points_walls)
            finish_wall = intial_wall_random_samples(intial_points_walls)

    # Fill the screen with white background
    screen.fill(BLACK)

    room_geometry_exterior(screen, Lx, Ly, PadX, PadY)

    # Draw the line
    rect = room_geometry_interior(screen, Lx, Ly, Dx, Dy, PadX, PadY)

    # Genera la grilla de puntos
    wall_inicial, wall_finish = intial_condition_grid(screen, Lx, Ly, PadX, PadY, initial_wall, finish_wall)
    wall_middle = dots_grid(screen, Lx, Ly, Dx, Dy, PadX, PadY, M, grid_random)
    walls = wall_inicial + wall_middle + wall_finish
    
    # Grafica las paredes
    plot_walls(screen, walls)

    valid_geometry = True
    for i in range(len(walls) - 1):
        if rect.clipline(walls[i], walls[i + 1]):
            valid_geometry = False
    
    if valid_geometry:
        screen.blit(label_invalid, (Lx//2 + 200, PadY//2))
        points = 0
        label_points = font.render(f"Score: {points}", True, (255, 255, 255))
        screen.blit(label_points, (Lx//2 - 200, PadY//2))
    else:
        screen.blit(label_valid, (Lx//2 + 200, PadY//2))
        points += 1
        label_points = font.render(f"Score: {points}", True, (255, 255, 255))
        screen.blit(label_points, (Lx//2 - 200, PadY//2))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate at 30 FPS
    clock.tick(30)

# Clean up and exit
pygame.quit()
sys.exit()
