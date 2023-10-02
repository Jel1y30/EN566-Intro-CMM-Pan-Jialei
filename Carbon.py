import numpy as np
import matplotlib.pyplot as plt

# Constants
T_half = 5700  # Half-life in years
lambda_ = np.log(2) / T_half  # Decay constant

# Time values
t_values = np.arange(0, 20001, 10)  # Time steps of 10 years
t_values_100 = np.arange(0, 20001, 100)  # Time steps of 100 years

# Analytical solution for R(t)
R_exact = -lambda_ * np.exp(-lambda_ * t_values)
R_exact_100 = -lambda_ * np.exp(-lambda_ * t_values_100)

# Numerical approximation for R(t)
R_numerical = np.diff(-np.exp(-lambda_ * t_values)) / np.diff(t_values)
R_numerical_100 = np.diff(-np.exp(-lambda_ * t_values_100)) / np.diff(t_values_100)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t_values[:-1], R_numerical, label='Numerical (Δt = 10 years)')
plt.plot(t_values_100[:-1], R_numerical_100, label='Numerical (Δt = 100 years)')
plt.plot(t_values, R_exact, label='Analytical', linestyle='--')
plt.xlabel('Time (years)')
plt.ylabel('Activity R(t)')
plt.legend()
plt.title('Radioactive Decay')
plt.grid(True)
plt.show()
