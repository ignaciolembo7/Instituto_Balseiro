import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def discrete_population_evolution(initial_population, d, time_steps):
    population = np.zeros(time_steps+1)
    population[0] = initial_population

    for t in range(time_steps):
        for i in range(int(population[t])):
            if np.random.rand() > d:
                population[t+1] += 1

    return population

def plot_population_distribution(population, time_steps, d):
    plt.figure(figsize=(10, 6))
    plt.bar(range(time_steps+1), population, color='blue', alpha=0.7)
    plt.title(f'Population Distribution over Time (d={d})')
    plt.xlabel('Time Steps')
    plt.ylabel('Population')
    plt.grid(True)
    plt.show()

def compare_with_binomial(initial_population, d, time_steps):
    p_survive = 1 - d
    binomial_population = binom.pmf(range(time_steps+1), time_steps, p_survive) * initial_population

    plt.figure(figsize=(10, 6))
    plt.bar(range(time_steps+1), binomial_population, color='green', alpha=0.5, label='Binomial')
    plt.plot(range(time_steps+1), population, color='blue', marker='o', linestyle='-', label='Simulation')
    plt.title(f'Comparison with Binomial Distribution (d={d})')
    plt.xlabel('Population Size')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

# Parameters
initial_population = 100
d_values = [0.1, 0.3]
time_steps = 20

for d in d_values:
    population = discrete_population_evolution(initial_population, d, time_steps)
    plot_population_distribution(population, time_steps, d)
    compare_with_binomial(initial_population, d, time_steps)
