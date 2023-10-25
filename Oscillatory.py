import numpy as np
import matplotlib.pyplot as plt

g = 9.8
l = 9.8
gamma = 0.25
alpha_D = 0.2
Omega_D_resonance = 1.0  # Resonance frequency from Part 1 in s^-1
delta = 0.1
Omega_D_values = np.arange(Omega_D_resonance - 5*delta, Omega_D_resonance + 5*delta, delta)

# Differential equation system for the linear pendulum
def pendulum_system(t, y, Omega_D):
    theta, omega = y
    dydt = [omega, -(g/l)*theta - 2*gamma*omega + alpha_D*np.sin(Omega_D*t)]
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

# Solve and plot for various Omega_D values
for Omega_D_val in Omega_D_values:
    times, theta, omega = solve_rk4(0, 0, 0.01, 50, Omega_D_val)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(times, theta, label=f'Omega_D = {Omega_D_val:.2f}')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (rad)')
    plt.title('Theta vs Time')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(times, omega, label=f'Omega_D = {Omega_D_val:.2f}')
    plt.xlabel('Time (s)')
    plt.ylabel('Omega (rad/s)')
    plt.title('Omega vs Time')
    plt.legend()
    plt.tight_layout()
    plt.show()
