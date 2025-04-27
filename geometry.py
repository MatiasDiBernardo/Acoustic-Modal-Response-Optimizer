import pygame
import numpy as np

class WallsGenerator():
    def __init__(self, Lx, Ly, Dx, Dy, PadX, PadY, N, num_walls):
        self.Lx = Lx
        self.Ly = Ly
        self.Dx = Dx
        self.Dy = Dy
        self.PadX = PadX
        self.PadY = PadY
        self.N = N
        self.num_walls = num_walls
        self.points_initial_wall = 12
        
        # Condiciones de control
        self.score = 0
        self.valid_geometries = []
        
        # Condición inicial de paredes
        self.calculate_interior_rect()
        self.initial_wall = self.intial_wall_random_samples(self.points_initial_wall)
        self.finish_wall = self.intial_wall_random_samples(self.points_initial_wall)
        
        # General Design
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.line_width = 2
        
        # Control de espacialidad
        if Ly < Lx:
            self.Lx = Ly
            self.Ly = Lx

    def update_walls(self):
        # Agergar para variar las dimensiones

        # Paredes random
        self.initial_wall = self.intial_wall_random_samples(self.points_initial_wall)
        self.finish_wall = self.intial_wall_random_samples(self.points_initial_wall)
    
    def calculate_interior_rect(self):
        # Posiciones borde interior
        top_left = (self.PadX + self.Dx, self.PadY + self.Dy)
        rect_width = self.Lx - 2 * self.Dx
        rect_height = self.Ly - 2 * self.Dy
        
        # Borde interior
        self.interior_rect = pygame.Rect(
            top_left[0],
            top_left[1],
            rect_width,
            rect_height
        )

    def room_geometry_outline(self, screen):
        # Posiciones borde exterior
        top_left = (self.PadX, self.PadY)
        top_right = (self.PadX + self.Lx, self.PadY)
        bottom_left = (self.PadX, self.PadY + self.Ly)
        bottom_rignt = (self.PadX +self.Lx, self.PadY + self.Ly)
        
        # Borde exterior
        pygame.draw.line(screen, self.WHITE, top_left, top_right, self.line_width)
        pygame.draw.line(screen, self.WHITE, top_right, bottom_rignt, self.line_width)
        pygame.draw.line(screen, self.WHITE, bottom_rignt, bottom_left, self.line_width)
        pygame.draw.line(screen, self.WHITE, bottom_left, top_left, self.line_width)
        
        # Border Interior 
        pygame.draw.rect(screen, self.WHITE, self.interior_rect, self.line_width)

    def intial_wall_random_samples(self, points_initial):
        arr = np.zeros(points_initial)
        idx_rndm = np.random.randint(0, points_initial)
        arr[idx_rndm] = 1
        
        return arr
    
    def intial_condition_grid(self, screen):
        spx = self.PadX  # Start Position X
        spy = self.PadY  # Start Position Y
        M = len(self.initial_wall)
        dx = (self.Lx//2) / M
        self.pos_wall_initial = []
        self.pos_wall_final = []
        thickness = 2
        
        # Grid pared inicial (esta la posibilidad de hacer len(initial) - 1 para dejar el nodo libre)
        for i in range(len(self.initial_wall)):
            pos = (spx + dx * i, spy)
            if self.initial_wall[i] == 1:
                color = (255, 0, 0) 
                self.pos_wall_initial.append(pos)
            else:
                color = (0, 0, 255) 
            pygame.draw.circle(screen, color, pos, thickness)

        # Grid pared final
        for i in range(len(self.finish_wall)):
            pos = (spx + dx * i, spy + self.Ly)
            if self.finish_wall[i] == 1:
                color = (255, 0, 0) 
                self.pos_wall_final.append(pos)
            else:
                color = (0, 0, 255) 

            pygame.draw.circle(screen, color, pos, thickness)
        
        return self.pos_wall_initial, self.pos_wall_final

    def dots_grid(self, screen, grid_points):
        # Area calc
        top_left = (self.PadX, self.PadY)

        top_area =self.Lx//2 * self.Dy
        middle_area = self.Dx * (self.Ly - 2 * self.Dy)
        bottom_area =self.Lx//2 * self.Dy
        
        area_total = top_area + middle_area + bottom_area

        # Define setp: Si propongo que dx == dy
        dx = 1
        for i in range(1,self.Lx//2):
            area_dots = self.N * dx**2
            if area_dots > area_total:
                break
            dx = i 
        
        # Iterar 
        thickness = 3
        row = 0
        col = 0

        # Intial position and conditions
        spx = top_left[0]  # Start position X
        spy = top_left[1] + dx #  Start position Y
        
        # Margenes
        x_margin1 = self.PadX +self.Lx//2
        y_margin1 = self.PadY + self.Dy

        x_margin2 = self.PadX + self.Dx
        y_margin2 = self.PadY + self.Ly - self.Dy
        
        x_margin3 = self.PadX +self.Lx//2
        y_margin3 = self.PadY + self.Ly

        # Crea la grilla
        self.pos_wall_middle = []
        for i in range(self.N):
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
                    self.pos_wall_middle.append(pos)
                continue

            if spx + dx * row < x_margin1 and spy + dx * col < y_margin1:  
                pos = (spx + dx * row, spy + dx * col)
                row += 1
                pygame.draw.circle(screen, color, pos, thickness)
                if grid_points[i] == 1:
                    self.pos_wall_middle.append(pos)
                continue
            
            # Segunda área
            if spx + dx * row >= x_margin2 and spy + dx * col < y_margin2:
                col += 1
                row = 0
                pos = (spx + dx * row, spy + dx * col)
                row += 1
                pygame.draw.circle(screen, color, pos, thickness)
                if grid_points[i] == 1:
                    self.pos_wall_middle.append(pos)
                continue

            if spx + dx * row < x_margin2 and spy + dx * col < y_margin2:  
                pos = (spx + dx * row, spy + dx * col)
                row += 1
                pygame.draw.circle(screen, color, pos, thickness)
                if grid_points[i] == 1:
                    self.pos_wall_middle.append(pos)
                continue

            # Tercera área
            if spx + dx * row >= x_margin3:
                col += 1
                row = 0
                pos = (spx + dx * row, spy + dx * col)
                row += 1
                pygame.draw.circle(screen, color, pos, thickness)
                if grid_points[i] == 1:
                    self.pos_wall_middle.append(pos)
                continue

            if spx + dx * row < x_margin3 and spy + dx * col < y_margin3:  
                pos = (spx + dx * row, spy + dx * col)
                row += 1
                pygame.draw.circle(screen, color, pos, thickness)
                if grid_points[i] == 1:
                    self.pos_wall_middle.append(pos)
                continue

    def plot_walls(self, screen):
        color = (255, 255, 0)
        self.line_width = 4
        walls = self.pos_wall_initial + self.pos_wall_middle + self.pos_wall_final
        for i in range(len(walls) - 1):
            pygame.draw.line(screen, color, walls[i], walls[i + 1], self.line_width)
    
    def is_valid(self):
        walls = self.pos_wall_initial + self.pos_wall_middle + self.pos_wall_final
        valid_geometry = True
        for i in range(len(walls) - 1):
            if self.interior_rect.clipline(walls[i], walls[i + 1]):
                valid_geometry = False
                self.score = 0

        return valid_geometry
    
    def normalize_coordinates(self):
        """
        Esta función mapea las posiciones en el plano de PyGame a un escenario real.
        """
        a = 0

# Idea para implementar 
# Yo genero con este algoritmo 1000 salas válidas
# De esas mil salas válidas, se ordenan por mérito y con las primeras 200 entran en el algo genético
# De esas doscientas, se generar pequeñas perturbaciones (mover a derecha o izquierda de la grilla)
# Para determinar la mejor geometría.

