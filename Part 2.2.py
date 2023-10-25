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

V, iterations = poisson_jacobi(V, rho)

tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
iterations_list = []

for tol in tolerances:
    _, iterations = poisson_jacobi(V, rho, tol)
    iterations_list.append(iterations)

plt.plot(tolerances, iterations_list, 'o-')
plt.xscale('log')
plt.xlabel('Tolerance')
plt.ylabel('Number of Iterations')
plt.title('Iterations vs. Tolerance for Jacobi Relaxation')
plt.grid(True)
plt.show()
