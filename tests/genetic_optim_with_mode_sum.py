import pygad
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FEM.mode_sumation import compute_modal_transfer
from aux.merit_figure import merit_magnitude_deviation

# Define the fitness function
def fitness_func(ga_instance, solution, solution_idx):
    # Posiciones fuente y receptor (en metros)
    source_position = (1.9, 1.0, 1.3)
    receptor_position = (1.25, 1.9, 1.2)
    freqs = np.arange(20, 200, 1)

    rta1 = compute_modal_transfer(source_position, receptor_position, solution, freqs)
    receptor_position1 = (1.25, 2, 1.2)
    rta2 = compute_modal_transfer(source_position, receptor_position1, solution, freqs)
    receptor_position2 = (1.25, 1.8, 1.2)
    rta3 = compute_modal_transfer(source_position, receptor_position2, solution, freqs)
    rta_tot = np.vstack([rta1, rta2, rta3])
    
    merit_value = merit_magnitude_deviation(rta_tot)
    
    return 1 / merit_value

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
    num_generations=200,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=3,
    gene_space=gene_space,
    parent_selection_type="sss",    # steady-state selection
    keep_parents=2,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=20
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

