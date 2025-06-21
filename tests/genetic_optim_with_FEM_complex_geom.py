import pygad
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FEM.FEM_source import FEM_Source_Solver_Average
from mesh.mesh_3D_complex import create_complex_mesh
from aux.merit_figure import merit_magnitude_deviation, merit_spatial_deviation
from room.geometry_generator import calculation_of_geometry, WallsGenerator

def fitness_func(ga_instance, solution, solution_idx):
    print("El indice es: ", solution_idx)
    Lx = 250       # Largo de la sala en X 
    Ly = 300       # Largo de la sala en Y
    Lz = 220       # Alto de la sala
    Dx = 50        # Delta X
    Dy = 80        # Delta Y
    Dz = 10        # Delta Z

    # Posiciones fuente y receptor (en metros)
    source_position = (1.9, 1.0, 1.3)
    receptor_position = (1.25, 1.9, 1.2)
    name_mesh = "mesh_genetic_simple"
    f_max = 140
    freqs = np.arange(20, f_max, 2)
    Z = 2.2

    # Parametros de control
    N = 250        # Densidad de la grilla del generador de geometrías
    n_walls = 2    # Número de cortes en las paredes

    wall_gen = WallsGenerator(Lx, Ly, Dx, Dy, 0, 0, N, n_walls, False)
    coords = np.array(solution).reshape((8, 2))

    # NO valida si hay simetría, debería areglar eso para utilizar mutación
    if not wall_gen.is_valid_genetic(coords):
        print("La siguiente geometría no es válida: ")
        print(coords)
        return 0.01

    create_complex_mesh(coords, Z, source_position, f_max, name_mesh)
    rta_tot = FEM_Source_Solver_Average(freqs, f'mallado/{name_mesh}.msh', receptor_position)
    
    MD = merit_magnitude_deviation(rta_tot)
    SD = merit_spatial_deviation(rta_tot)
    merit_value = MD + SD
    
    return 1 / merit_value

def print_gen(ga_instance):
    print(" ")
    print("-------------------------------------------")
    # print(f"Generation {ga_instance.generations_completed}: Best Fitness = {1/ga_instance.best_solution()[1]:.4f}")
    print("-------------------------------------------")
    print(" ")

Lx = 250       # Largo de la sala en X 
Ly = 300       # Largo de la sala en Y
Lz = 220       # Alto de la sala
Dx = 50        # Delta X
Dy = 80        # Delta Y
Dz = 10        # Delta Z

# Posiciones fuente y receptor (en metros)
source_position = (1.9, 1.0, 1.3)
receptor_position = (1.25, 1.9, 1.2)

# Parametros de control
N = 250        # Densidad de la grilla del generador de geometrías
n_walls = 2    # Número de cortes en las paredes

# Genetic config
N_initial_pop = 200
gene_space = [{'low': 0, 'high': Dx/100},
              {'low': 0, 'high': Dy/100},
              {'low': 0, 'high': Dx/100},
              {'low': 0, 'high': Dy/100},
              {'low': 0, 'high': Dx/100},
              {'low': 0, 'high': Dy/100},
              {'low': 0, 'high': Dx/100},
              {'low': 0, 'high': Dy/100},
              {'low': 0, 'high': Dx/100},
              {'low': 0, 'high': Dy/100},
              {'low': 0, 'high': Dx/100},
              {'low': 0, 'high': Dy/100},
              {'low': 0, 'high': Dx/100},
              {'low': 0, 'high': Dy/100},
              {'low': 0, 'high': Dx/100},
              {'low': 0, 'high': Dy/100}]

initial_population = calculation_of_geometry(Lx, Ly, Dx, Dy, N, N_initial_pop, n_walls)
initial_population = np.array(initial_population)

# Se hace un reshape para matcher el output necesario por PyGAD
POP, POINTS, COORDS = initial_population.shape
initial_pop_flat = initial_population.reshape(POP, POINTS * COORDS)

# ga_instance = pygad.GA(
#     num_generations=10,
#     sol_per_pop=POP,  # 15
#     num_parents_mating=4,
#     initial_population=initial_pop_flat,      # ← your custom start
#     fitness_func=fitness_func,
#     num_genes=POINTS * COORDS,
#     gene_space=gene_space,
#     parent_selection_type="sss",    # steady-state selection
#     crossover_type="scattered",
#     mutation_type=None,
#     on_generation=print_gen,
# )

# Optimización de chat gpt bro
ga_instance = pygad.GA(
    num_generations       = 3,                    # keep it small
    sol_per_pop           = POP,                     # tiny pop
    num_parents_mating    = 4,
    initial_population=initial_pop_flat,      # ← your custom start
    fitness_func=fitness_func,
    num_genes=POINTS * COORDS,
    gene_space=gene_space,
    
    # 1) Early stopping if no improvement for 5 gens
    # stop_criteria         = ["saturate_5"],        
    
    # 2) Parallel fitness calls (use 4 processes)
    # parallel_processing   = ["process", 4],        
    
    # 3) Cache fitness of seen solutions
    save_solutions        = True,
    save_best_solutions   = True,
    keep_elitism          = 1,
    keep_parents          = 0,
    
    # 4) Steady‐state + high mutation
    parent_selection_type = "sss",
    crossover_type        = "scattered",
    mutation_type=None,
    on_generation=print_gen,
)

ga_instance.run()

# After the run, we can get the details of the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"La mejor geometría encontrada es: ", solution)
print(f"Fitness value: {1/solution_fitness:.4f}")

# Plot the fitness evolution over generations
generations = np.arange(len(ga_instance.best_solutions_fitness))
plt.plot(generations, 1/np.array(ga_instance.best_solutions_fitness))
plt.title("Fitness over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()
