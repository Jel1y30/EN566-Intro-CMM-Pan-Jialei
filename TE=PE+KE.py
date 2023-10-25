import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.8
l = 9.8
gamma = 0.25
alpha_D = 0.2
Omega_D_resonance = 1.0  # Resonance frequency from Part 1 in s^-1
m = 1  # Assuming a pendulum mass of 1kg

# Differential equation system for the linear pendulum
def pendulum_system(t, y, Omega_D):
    theta, omega = y
    dydt = np.array([omega, -(g/l)*theta - 2*gamma*omega + alpha_D*np.sin(Omega_D*t)])
    return dydt

# RK4 method
def rk4_step(func, t, y, dt, Omega_D):
    k1 = np.multiply(func(t, y, Omega_D), dt)
    k2 = np.multiply(func(t + 0.5*dt, y + 0.5*k1, Omega_D), dt)
    k3 = np.multiply(func(t + 0.5*dt, y + 0.5*k2, Omega_D), dt)
    k4 = np.multiply(func(t + dt, y + k3, Omega_D), dt)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

def solve_rk4(theta0, omega0, dt, T, Omega_D):
    times = np.arange(0, T, dt)
    results = np.zeros((2, len(times)))
    results[:, 0] = [theta0, omega0]

    for i, t in enumerate(times[:-1]):
        results[:, i+1] = rk4_step(pendulum_system, t, results[:, i], dt, Omega_D)
    return times, results[0], results[1]

# Solve the ODE
times, theta, omega = solve_rk4(0, 0, 0.01, 10 * (2 * np.pi / Omega_D_resonance), Omega_D_resonance)

# Compute the energies
U = m * g * l * (1 - np.cos(theta))
K = 0.5 * m * l**2 * omega**2
E = U + K

# Plot the energies
plt.figure(figsize=(10,6))
plt.plot(times, U, label='Potential Energy')
plt.plot(times, K, label='Kinetic Energy')
plt.plot(times, E, label='Total Energy', linestyle='--', color='black')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.legend()
plt.title('Energy vs Time')
plt.grid(True)
plt.show()
