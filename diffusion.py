import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the parameters for the simulation
D = 2.0  # Diffusion constant
dx = 0.1  # Space step
dt = (dx ** 2) / (2 * D)  # Time step, chosen via the stability condition for diffusion
length = 10.0  # Total length of the space domain
time = 2.0  # Total time to run the simulation

# Create the space and time grids
x = np.arange(-length / 2, length / 2, dx)
t = np.arange(0, time, dt)
num_x = len(x)
num_t = len(t)

# Initialize the density profile (box profile)
density = np.zeros(num_x)
density[(x >= -1.0) & (x <= 1.0)] = 1.0  # Initial sharp peak around x=0

# Function to update the density profile using the finite difference method
def update_density(density, D, dx, dt):
    # Copy the current density to not overwrite the data as we go
    new_density = np.copy(density)
    # Update the density at each point, except the boundaries
    for i in range(1, len(density) - 1):
        new_density[i] = density[i] + (D * dt / dx**2) * (density[i - 1] - 2 * density[i] + density[i + 1])
    return new_density

# Function to plot and fit the density profile to a normal distribution
def fit_normal_distribution(x, density, t):
    # Fit the density data to a Gaussian function
    def gaussian(x, A, mean, stddev):
        return A * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
    
    params, _ = curve_fit(gaussian, x, density, p0=[1, 0, np.sqrt(2 * D * t)])
    A, mean, stddev = params
    
    # Plot the density and the fitted Gaussian
    plt.figure(figsize=(10, 6))
    plt.plot(x, density, label='Numerical solution')
    plt.plot(x, gaussian(x, *params), label='Fitted Gaussian', linestyle='dashed')
    plt.title(f'Time = {t:.2f} s, Fitted stddev = {stddev:.2f}, Theoretical stddev = {np.sqrt(2*D*t):.2f}')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    return stddev

# Run the simulation and fit at various time snapshots
stddevs = []  # To store the standard deviations from the fit
snapshot_times = np.linspace(0, time, 5)[1:]  # Skip the initial condition
for current_time in snapshot_times:
    # Update the density profile over time
    for _ in range(int(current_time / dt)):
        density = update_density(density, D, dx, dt)
    
    # Fit the density profile to a normal distribution
    stddev = fit_normal_distribution(x, density, current_time)
    stddevs.append(stddev)

# Check if the standard deviations from the fit match the theoretical prediction
stddevs = np.array(stddevs)
theoretical_stddevs = np.sqrt(2 * D * snapshot_times)
comparison = np.isclose(stddevs, theoretical_stddevs, atol=0.1)  # Allowing some tolerance

# Output the comparison results
stddevs, theoretical_stddevs, comparison
