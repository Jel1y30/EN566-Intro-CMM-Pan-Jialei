import random
import matplotlib.pyplot as plt
import numpy as np

# Part 1: Uniform Distribution

# Generate 1,000 random numbers uniformly distributed between 0 and 1
uniform_random_numbers_1000 = np.random.uniform(0, 1, 1000)

# Function to plot the distribution with different subdivisions
def plot_uniform_distribution(data, subdivisions, sample_size):
    plt.hist(data, bins=subdivisions, density=True)
    plt.title(f'Uniform Distribution with {subdivisions} Bins (Sample Size: {sample_size})')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.show()

# Plotting for 1,000 random numbers with 10, 20, 50, and 100 subdivisions
subdivisions_list = [10, 20, 50, 100]

# Generate the plots for 1,000 random numbers
for subdivisions in subdivisions_list:
    plot_uniform_distribution(uniform_random_numbers_1000, subdivisions, 1000)

# Generate 1,000,000 random numbers uniformly distributed between 0 and 1
uniform_random_numbers_1000000 = np.random.uniform(0, 1, 1000000)

# Generate the plots for 1,000,000 random numbers
for subdivisions in subdivisions_list:
    plot_uniform_distribution(uniform_random_numbers_1000000, subdivisions, 1000000)

from scipy.stats import norm

# Part 2: Gaussian Distribution

# Define the parameters for the Gaussian distribution
mu = 0  # mean
sigma = 1.0  # standard deviation

# Generate random numbers with Gaussian distribution using numpy's built-in function
gaussian_random_numbers_1000 = np.random.normal(mu, sigma, 1000)

# Function to plot the Gaussian distribution with different subdivisions
def plot_gaussian_distribution(data, subdivisions, sample_size):
    count, bins, ignored = plt.hist(data, bins=subdivisions, density=True, alpha=0.6, color='g')

    # Overlay the Gaussian distribution
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu)**2 / (2 * sigma**2)),
             linewidth=2, color='r')
    plt.title(f'Gaussian Distribution with {subdivisions} Bins (Sample Size: {sample_size})')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.show()

# Generate the plots for 1,000 Gaussian random numbers
for subdivisions in subdivisions_list:
    plot_gaussian_distribution(gaussian_random_numbers_1000, subdivisions, 1000)
