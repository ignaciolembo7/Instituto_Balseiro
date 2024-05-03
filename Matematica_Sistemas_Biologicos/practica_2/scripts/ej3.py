import numpy as np
import matplotlib.pyplot as plt

def gillespie_algorithm(b, d, initial_population, max_time):
    time = 0
    population = initial_population
    time_points = [time]
    population_values = [population]

    while time < max_time:
        # Calculate reaction rates
        reproduction_rate = b * population
        competition_rate = d * population * (population - 1)
        total_rate = reproduction_rate + competition_rate

        # Generate time of next reaction
        delta_t = np.random.exponential(1 / total_rate)

        # Determine reaction
        r = np.random.rand()
        if r < reproduction_rate / total_rate:
            population += 1
        else:
            if population > 0:
                population -= 1

        time += delta_t
        time_points.append(time)
        population_values.append(population)

    return time_points, population_values

# Parameters
b = 0.1
d = 0.001
initial_population = 100
max_time = 1000

# Run Gillespie algorithm
time_points, population_values = gillespie_algorithm(b, d, initial_population, max_time)

# Plot population dynamics
plt.plot(time_points, population_values)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Population Dynamics')
plt.grid(True)
plt.show()
