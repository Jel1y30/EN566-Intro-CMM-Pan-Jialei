import numpy as np
import matplotlib.pyplot as plt

# Define the number of walks and steps
n_walks = 10**4
n_steps = 100

# Function to perform a single random walk
def random_walk(n):
    # Randomly choose direction, +1 or -1, for each step in x and y
    steps_x = np.random.choice([-1, 1], n)
    steps_y = np.random.choice([-1, 1], n)
    # Cumulatively sum the steps taken to find the position at each step
    position_x = np.cumsum(steps_x)
    position_y = np.cumsum(steps_y)
    return position_x, position_y

# Perform multiple walks and calculate mean square displacement
msd = np.zeros(n_steps)
for i in range(n_walks):
    x, y = random_walk(n_steps)
    msd += x**2 + y**2

# Average over all walks
msd /= n_walks

# Calculate <x_n> and <x_n^2> for plotting
x_mean = np.sqrt(msd) # <x_n>
x2_mean = msd # <x_n^2>

# Plot <x_n> and <x_n^2>
plt.figure(figsize=(14, 7))

# Plot <x_n>
plt.subplot(1, 2, 1)
plt.plot(x_mean, label=r'$\langle x_n \rangle$')
plt.title(r'Plot of $\langle x_n \rangle$')
plt.xlabel('Step number n')
plt.ylabel(r'$\langle x_n \rangle$')
plt.legend()

# Plot <x_n^2>
plt.subplot(1, 2, 2)
plt.plot(x2_mean, label=r'$\langle x_n^2 \rangle$')
plt.title(r'Plot of $\langle x_n^2 \rangle$')
plt.xlabel('Step number n')
plt.ylabel(r'$\langle x_n^2 \rangle$')
plt.legend()

plt.tight_layout()
plt.show()
