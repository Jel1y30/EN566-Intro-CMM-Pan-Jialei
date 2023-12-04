# Part 1
import numpy as np
import matplotlib.pyplot as plt

# Constants
kB = 1.0  # Boltzmann constant (set to 1 for simplicity)
J = 1.5  # Interaction strength
n = 50  # Lattice size
N = n * n  # Total number of spins

# Initialize the lattice with random spins
np.random.seed(0)  # Seed for reproducibility
lattice = np.random.choice([-1, 1], size=(n, n))

def metropolis_step_fast(lattice, T):
    """Perform one Monte Carlo sweep over the lattice using numpy for fast computation."""
    for _ in range(N):
        x, y = np.random.randint(0, n, 2)  # Select a random spin
        S = lattice[x, y]
        nb = (lattice[(x+1)%n, y] + lattice[x, (y+1)%n] + lattice[(x-1)%n, y] + lattice[x, (y-1)%n])
        deltaE = 2 * J * S * nb
        # Spin flip condition
        if deltaE < 0 or np.random.rand() < np.exp(-deltaE / (kB * T)):
            lattice[x, y] = -S

def calculate_magnetization(lattice):
    """Calculate the magnetization of the lattice."""
    return np.abs(np.sum(lattice)) / N

# Perform the simulation for a smaller range of temperatures for efficiency
T_range = np.linspace(2.2, 2.6, 20)  # Reduced temperature range around the expected critical point
equilibrium_steps = 200  # Reduced number of Monte Carlo sweeps to reach equilibrium
measurement_steps = 50  # Reduced number of steps to average the magnetization

magnetizations = []

for T in T_range:
    # Start with a new random lattice for each temperature
    lattice = np.random.choice([-1, 1], size=(n, n))
    
    # Equilibrate the system
    for _ in range(equilibrium_steps):
        metropolis_step_fast(lattice, T)
    
    # Measure the magnetization
    M = 0
    for _ in range(measurement_steps):
        metropolis_step_fast(lattice, T)
        M += calculate_magnetization(lattice)
    magnetizations.append(M / measurement_steps)

# Plot the magnetization as a function of temperature
plt.figure(figsize=(10, 5))
plt.plot(T_range, magnetizations, marker='o', linestyle='-', color='blue')
plt.xlabel('Temperature (T)')
plt.ylabel('Magnetization per spin (M/N)')
plt.title('Magnetization vs Temperature for the 2D Ising Model (Optimized)')
plt.grid(True)
plt.show()

# Returning the magnetizations and temperature range for further analysis if needed
magnetizations, T_range

import numpy as np
import matplotlib.pyplot as plt

# Constants
kB = 1.0  # Boltzmann constant
J = 1.5   # Interaction strength
T_min = 2.0  # Minimum temperature
T_max = 3.0  # Maximum temperature
T_steps = 20  # Number of temperature steps to simulate

# Function to perform the Metropolis algorithm step
def metropolis_step(lattice, T):
    N = lattice.shape[0]
    for _ in range(N**2):
        x, y = np.random.randint(0, N, size=2)
        spin = lattice[x, y]
        neighbour_spin_sum = lattice[(x+1)%N, y] + lattice[x, (y+1)%N] + lattice[(x-1)%N, y] + lattice[x, (y-1)%N]
        deltaE = 2 * J * spin * neighbour_spin_sum
        if deltaE < 0 or np.random.random() < np.exp(-deltaE / (kB * T)):
            lattice[x, y] = -spin
    return lattice

# Function to calculate the specific heat
def calculate_specific_heat(energy_list, T):
    energy_array = np.array(energy_list)
    energy_sq_array = energy_array**2
    C = (energy_sq_array.mean() - energy_array.mean()**2) / (kB * T**2)
    return C

def calculate_energy(lattice, J):
    """Calculate the total energy of the lattice configuration."""
    energy = 0
    N = lattice.shape[0]
    for i in range(N):
        for j in range(N):
            # Sum over nearest neighbor interactions
            # Apply periodic boundary conditions using modulo operator
            S = lattice[i, j]
            nb = lattice[(i+1)%N, j] + lattice[i, (j+1)%N] + lattice[(i-1)%N, j] + lattice[i, (j-1)%N]
            energy += -J * S * nb
    return energy / 4.0  # Divide by 4 to correct for over-counting

# Simulation parameters
lattice_sizes = [5, 10, 15, 20]  # Reduced lattice sizes for demonstration
temperatures = np.linspace(T_min, T_max, T_steps)  # Temperature range

# Data structures to hold simulation results
specific_heats = {n: [] for n in lattice_sizes}

# Run the simulation for each lattice size and temperature
for n in lattice_sizes:
    N = n * n  # Total number of spins
    lattice = np.random.choice([-1, 1], size=(n, n))
    for T in temperatures:
        # Equilibrate the system
        for _ in range(1000):  # Reduced number of sweeps for demonstration
            metropolis_step(lattice, T)
        
        # Calculate the energy after the system has reached equilibrium
        energies = [calculate_energy(lattice) for _ in range(100)]  # Reduced number of measurements
        
        # Calculate and store the specific heat
        specific_heats[n].append(calculate_specific_heat(energies, T))

# Plot the specific heat per spin for different lattice sizes
for n, C_values in specific_heats.items():
    plt.plot(temperatures, C_values, marker='o', linestyle='-', label=f'Lattice size {n}x{n}')

plt.xlabel('Temperature (T)')
plt.ylabel('Specific heat per spin (C/N)')
plt.title('Specific Heat per Spin vs Temperature for Various Lattice Sizes')
plt.legend()
plt.grid(True)
plt.show()

# Part 2
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Constants
kB = 1.0  # Boltzmann constant (set to 1 for simplicity)
J = 1.5   # Interaction strength

@jit(nopython=True)
def metropolis_step(lattice, T):
    """Perform one Monte Carlo sweep over the lattice."""
    n = lattice.shape[0]
    for _ in range(n * n):
        x, y = np.random.randint(0, n, 2)  # Select a random spin
        S = lattice[x, y]
        nb = (lattice[(x+1)%n, y] + lattice[x, (y+1)%n] + lattice[(x-1)%n, y] + lattice[x, (y-1)%n])
        deltaE = 2 * J * S * nb
        if deltaE < 0 or np.random.rand() < np.exp(-deltaE / (kB * T)):
            lattice[x, y] = -S

def compute_energy(lattice):
    """Compute the total energy of the lattice configuration."""
    energy = 0
    n = lattice.shape[0]
    for i in range(n):
        for j in range(n):
            S = lattice[i, j]
            nb = lattice[(i+1)%n, j] + lattice[i, (j+1)%n] + lattice[(i-1)%n, j] + lattice[i, (j-1)%n]
            energy += -J * S * nb
    return energy / 4.0  # Each pair counted twice, each bond counted twice

def calculate_specific_heat(energies, T):
    """Calculate the specific heat per spin."""
    energy_fluctuation = np.var(energies)
    return energy_fluctuation / (T**2)

# Simulation parameters
lattice_sizes = [5, 10, 20, 30, 40, 50, 75, 100]  # Lattice sizes to simulate
T_range = np.linspace(2.0, 2.5, 20)  # Temperature range

# Dictionary to store specific heats for each lattice size
specific_heats = {n: [] for n in lattice_sizes}

for n in lattice_sizes:
    lattice = np.random.choice([-1, 1], size=(n, n))
    for T in T_range:
        energies = []

        # Equilibrate the system
        for _ in range(1000):
            metropolis_step(lattice, T)

        # Measure the energy
        for _ in range(100):
            metropolis_step(lattice, T)
            energies.append(compute_energy(lattice))

        # Calculate and store the specific heat
        C = calculate_specific_heat(np.array(energies), T)
        specific_heats[n].append(C)

# Plotting the specific heat for different lattice sizes
plt.figure(figsize=(10, 6))
for n, Cs in specific_heats.items():
    plt.plot(T_range, Cs, label=f'Lattice size {n}x{n}')
plt.xlabel('Temperature (T)')
plt.ylabel('Specific heat per spin (C/N)')
plt.title('Specific Heat per Spin for Various Lattice Sizes')
plt.legend()
plt.show()

# ising.py
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Run Ising Model simulation.')
parser.add_argument('--part', type=int, choices=[1, 2], help='Part of the simulation to run')

# Parse arguments
args = parser.parse_args()

# Conditional execution based on the part argument
if args.part == 1:
    print("Running Part 1 of the simulation...")
    # Add code for Part 1 here
elif args.part == 2:
    print("Running Part 2 of the simulation...")
    # Add code for Part 2 here
