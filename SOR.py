import numpy as np
import matplotlib.pyplot as plt

def poisson_jacobi(V, rho, tol=1e-5):
    """
    Solve Poisson's equation using the Jacobi relaxation method.
    """
    V_new = V.copy()
    delta = tol + 1
    iterations = 0
    while delta > tol:
        V_old = V_new.copy()
        for i in range(1, V.shape[0]-1):
            for j in range(1, V.shape[1]-1):
                V_new[i, j] = 0.25 * (V_old[i+1, j] + V_old[i-1, j] + V_old[i, j+1] + V_old[i, j-1] + rho[i, j])
        delta = np.max(np.abs(V_new - V_old))
        iterations += 1
    return V_new, iterations

def poisson_SOR(V, rho, omega=1.5, tol=1e-5):
    V_new = V.copy()
    delta = tol + 1
    iterations = 0
    while delta > tol:
        V_old = V_new.copy()
        for i in range(1, V.shape[0]-1):
            for j in range(1, V.shape[1]-1):
                V_new[i, j] = (1-omega) * V_old[i, j] + omega * 0.25 * (V_old[i+1, j] + V_old[i-1, j] + V_old[i, j+1] + V_old[i, j-1] + rho[i, j])
        delta = np.max(np.abs(V_new - V_old))
        iterations += 1
    return V_new, iterations

# Parameters
L = 20
N = 100
a = 0.6
dx = L/N
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Initialize potential and charge distribution
V = np.zeros((N, N))
rho = np.zeros((N, N))
rho[int(N/2), int(N/2 - a/(2*dx))] = 1   # Positive charge
rho[int(N/2), int(N/2 + a/(2*dx))] = -1  # Negative charge

grid_sizes = [50, 100, 200, 400]
iterations_jacobi = []
iterations_SOR = []

for N in grid_sizes:
    dx = L/N
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    V = np.zeros((N, N))
    rho = np.zeros((N, N))
    rho[int(N/2), int(N/2 - a/(2*dx))] = 1   # Positive charge
    rho[int(N/2), int(N/2 + a/(2*dx))] = -1  # Negative charge

    _, iter_j = poisson_jacobi(V, rho)
    _, iter_s = poisson_SOR(V, rho)
    iterations_jacobi.append(iter_j)
    iterations_SOR.append(iter_s)

plt.plot(grid_sizes, iterations_jacobi, 'o-', label='Jacobi')
plt.plot(grid_sizes, iterations_SOR, 's-', label='SOR')
plt.xlabel('Grid Size (n)')
plt.ylabel('Number of Iterations')
plt.title('Iterations vs. Grid Size')
plt.legend()
plt.grid(True)
plt.show()
