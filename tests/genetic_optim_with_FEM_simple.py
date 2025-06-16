import pygad
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mesh.mesh_3D_simple import create_simple_mesh
from FEM.FEM_source import FEM_Source_Solver_Average
from aux.merit_figure import merit_magnitude_deviation, merit_spatial_deviation

# Define the fitness function
def fitness_func(ga_instance, solution, solution_idx):
    # Posiciones fuente y receptor (en metros)
    source_position = (1.9, 1.0, 1.3)
    receptor_position = (1.25, 1.9, 1.2)
    name_mesh = "mesh_genetic_simple"
    f_max = 200
    freqs = np.arange(20, f_max, 2)
    
    create_simple_mesh(solution[0], solution[1], solution[2], source_position, f_max, name_mesh)
    rta_tot = FEM_Source_Solver_Average(freqs, f"mallado/{name_mesh}.msh", receptor_position)

    MD = merit_magnitude_deviation(rta_tot)
    SD = merit_spatial_deviation(rta_tot)
    merit_value = MD + SD
    
    return 1 / merit_value

def print_gen(ga_instance):
    print(f"Generation {ga_instance.generations_completed}: Best Fitness = {1/ga_instance.best_solution()[1]:.4f}")

Lx = 2.5
Ly = 3
Lz = 2.2       
Dx = 0.5        
Dy = 0.8       
Dz = 0.1       
    
# Genetic Algorithm parameters
gene_space = [
    {'low': Lx - Dx, 'high': Lx},      # Range for x1
    {'low': Ly - Dy, 'high': Ly},      # Range for x2
    {'low': Lz - Dz, 'high': Lz}   # Range for x3
]

ga_instance = pygad.GA(
    num_generations=8,
    sol_per_pop=10,
    num_parents_mating=4,
    fitness_func=fitness_func,
    num_genes=3,
    gene_space=gene_space,
    parent_selection_type="sss",    # steady-state selection
    keep_parents=2,
    crossover_type="scattered",
    mutation_type="random",
    mutation_percent_genes=20,
    on_generation=print_gen,
)

# Run the GA
ga_instance.run()

# After the run, we can get the details of the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best solution: x = {solution[0]:.4f}, y = {solution[1]:.4f}, z = {solution[2]:.4f}")
print(f"Fitness value: {1/solution_fitness:.4f}")

# Plot the fitness evolution over generations
import matplotlib.pyplot as plt
generations = np.arange(len(ga_instance.best_solutions_fitness))
plt.plot(generations, 1/np.array(ga_instance.best_solutions_fitness))
plt.title("Fitness over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()

