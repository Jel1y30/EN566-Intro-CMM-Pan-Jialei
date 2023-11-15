import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def initialize_grid(x, y, split):
    """
    Initialize the grid with two gases.
    :param x: Width of the grid.
    :param y: Height of the grid.
    :param split: Fraction of the grid to be filled with each gas.
    :return: A numpy array representing the grid.
    """
    grid = np.zeros((y, x), dtype=int)
    grid[:, :int(x * split)] = 1  # Gas A
    grid[:, int(x * (1 - split)):] = 2  # Gas B
    return grid

def random_walk_revised(grid):
    """
    Perform a random walk step for one particle in the grid, with revised logic to prevent boundary issues.
    :param grid: The grid representing the gas particles.
    :return: The updated grid.
    """
    occupied = np.argwhere(grid > 0)
    if occupied.size > 0:
        rand_index = np.random.randint(len(occupied))
        y, x = occupied[rand_index]

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        np.random.shuffle(directions)

        for dy, dx in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < grid.shape[1] and 0 <= new_y < grid.shape[0] and grid[new_y, new_x] == 0:
                grid[new_y, new_x], grid[y, x] = grid[y, x], 0
                break

    return grid

def plot_grid(grid, iteration):
    """
    Plot the current state of the grid.
    :param grid: The grid to be plotted.
    :param iteration: The current iteration number.
    """
    cmap = ListedColormap(['white', 'red', 'blue'])
    plt.imshow(grid, cmap=cmap)
    plt.title(f"Gas Mixing - Iteration {iteration}")
    plt.axis('off')
    plt.show()

def population_density(grid):
    """
    Calculate the linear population density for each gas.
    :param grid: The grid representing the gas particles.
    :return: Densities for Gas A and Gas B.
    """
    nA = np.sum(grid == 1, axis=0) / grid.shape[0]
    nB = np.sum(grid == 2, axis=0) / grid.shape[0]
    return nA, nB

def plot_densities(densities, iteration):
    """
    Plot the linear population densities of the gases.
    :param densities: Tuple containing densities of Gas A and Gas B.
    :param iteration: The current iteration number.
    """
    nA, nB = densities
    plt.plot(nA, label='Gas A Density')
    plt.plot(nB, label='Gas B Density')
    plt.title(f"Population Densities - Iteration {iteration}")
    plt.xlabel('x Position')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Simulation parameters
width, height = 60, 40
split_fraction = 1/3
iterations = 100000  # Number of iterations for mixing
larger_sample_iterations = [0, 2500, 5000, 7500, 10000, 50000, 100000]  # Larger time intervals for sampling

# Initialize the grid
grid = initialize_grid(width, height, split_fraction)

# Run the simulation with larger time intervals
for i in range(iterations + 1):
    if i in larger_sample_iterations:
        plot_grid(grid, i)
        densities = population_density(grid)
        plot_densities(densities, i)

    grid = random_walk_revised(grid)

# veraging the densities over multiple trials for increased accuracy
def run_simulation(iterations, sample_iterations, width, height, split_fraction):
    """
    Run the gas mixing simulation for a given number of iterations and return the densities at specified sample intervals.
    :param iterations: Total number of iterations for the simulation.
    :param sample_iterations: Iterations at which to sample the grid and densities.
    :param width: Width of the grid.
    :param height: Height of the grid.
    :param split_fraction: Fraction of the grid to be filled with each gas.
    :return: Dictionary of densities at specified sample iterations.
    """
    densities_at_samples = {i: [] for i in sample_iterations}
    grid = initialize_grid(width, height, split_fraction)

    for i in range(iterations + 1):
        if i in sample_iterations:
            densities = population_density(grid)
            densities_at_samples[i].append(densities)
        
        grid = random_walk_revised(grid)

    return densities_at_samples

def average_densities(densities_list):
    """
    Average the densities over multiple trials.
    :param densities_list: List of density tuples from multiple trials.
    :return: Averaged densities.
    """
    summed_nA = np.sum([densities[0] for densities in densities_list], axis=0)
    summed_nB = np.sum([densities[1] for densities in densities_list], axis=0)
    avg_nA = summed_nA / len(densities_list)
    avg_nB = summed_nB / len(densities_list)
    return avg_nA, avg_nB

# Parameters for averaging over multiple trials
num_trials = 100
all_densities = {i: [] for i in larger_sample_iterations}

# Run multiple trials
for trial in range(num_trials):
    trial_densities = run_simulation(iterations, larger_sample_iterations, width, height, split_fraction)
    for i in larger_sample_iterations:
        all_densities[i].append(trial_densities[i][0])

# Calculate and plot average densities
for i in larger_sample_iterations:
    avg_densities = average_densities(all_densities[i])
    plot_densities(avg_densities, i)

