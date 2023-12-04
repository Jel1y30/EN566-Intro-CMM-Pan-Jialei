import numpy as np
import matplotlib.pyplot as plt

# Constants
kB = 1.0  # Boltzmann constant (set to 1 for simplicity)
J = 1.5   # Interaction strength

# Function to perform the Metropolis algorithm step
def metropolis_step(lattice, T):
    N = lattice.shape[0]
    for _ in range(N**2):
        x, y = np.random.randint(0, N, size=2)
        spin = lattice[x, y]
        neighbour_spin_sum = lattice[(x + 1) % N, y] + lattice[(x - 1) % N, y] + lattice[x, (y + 1) % N] + lattice[x, (y - 1) % N]
        deltaE = 2 * J * spin * neighbour_spin_sum
        if deltaE < 0 or np.random.random() < np.exp(-deltaE / (kB * T)):
            lattice[x, y] = -spin
    return lattice

# Function to calculate the specific heat
def calculate_specific_heat(energy_list, T):
    energy_array = np.array(energy_list)
    energy_sq_array = energy_array**2
    C = (energy_sq_array.mean() - energy_array.mean()**2) / (T**2)
    return C

# Lattice sizes
lattice_sizes = [5, 10, 20, 30, 40, 50, 75, 100, 200, 500]

# Temperature range
T_min = 1.5
T_max = 3.5
T_steps = 20
temperatures = np.linspace(T_min, T_max, T_steps)

# Monte Carlo sweeps and steps
MC_sweeps = 10000
MC_steps = 100

# Data structures to hold simulation results
specific_heats = {n: [] for n in lattice_sizes}
C_max = []

# Main simulation loop
for n in lattice_sizes:
    N = n**2
    lattice = np.random.choice([-1, 1], (n, n))
    C_values = []
    for T in temperatures:
        # Equilibration
        for _ in range(MC_sweeps):
            metropolis_step(lattice, T)
        
        # Measurement
        energies = []
        for _ in range(MC_steps):
            metropolis_step(lattice, T)
            energy = -J * sum(lattice[(i + 1) % n, j] + lattice[i, (j + 1) % n] for i in range(n) for j in range(n))
            energies.append(energy)
        
        # Calculate specific heat
        C = calculate_specific_heat(energies, T) / N
        C_values.append(C)
    
    specific_heats[n] = C_values
    C_max.append(max(C_values))

# Plot C(T) for a few sample lattice sizes
for n in [5, 10, 50, 100]:
    plt.plot(temperatures, specific_heats[n], label=f'n={n}')

plt.xlabel('Temperature (T)')
plt.ylabel('Specific heat per spin (C/N)')
plt.title('Specific Heat per Spin vs Temperature for Various Lattice Sizes')
plt.legend()
plt.show()

# Plot C_max/N vs. log(n)
plt.plot(np.log(lattice_sizes), np.log(C_max), 'o-')
plt.xlabel('log(Lattice size)')
plt.ylabel('log(Max Specific heat per spin)')
plt.title('Log-log Plot of Max Specific Heat per Spin vs Lattice Size')
plt.show()
