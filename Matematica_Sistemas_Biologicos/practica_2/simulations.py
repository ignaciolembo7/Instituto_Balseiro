from tqdm import tqdm
import numpy as np

class binomial_simulation:
    def __init__(self, num_simulations: int, N0: int):
        self.num_simulations = num_simulations
        self.N0 = N0 

    def simulate_population(self,  d: float, time_step: int):
        populations = []
        for _ in tqdm(range(self.num_simulations)):
            population = self.N0
            for t in range(1, time_step+1):
                alive = np.random.binomial(population, 1 - d)
                population = alive
            populations.append(alive)
        return populations

class langevin_simulation:
    def __init__(self, a: float, sigma: float, x0: float, time_steps: int):
        self.a = a
        self.sigma = sigma
        self.x0 = x0
        self.time_steps = time_steps

    def simulate_trajectories_additive(self, num_trajectories: int):
        trajectories = np.zeros((num_trajectories, self.time_steps + 1))
        for i in tqdm(range(num_trajectories)):
            x = self.x0
            for t in range(self.time_steps + 1):
                trajectories[i, t] = x
                x = self.a * x + np.random.normal(0, self.sigma)
        return trajectories

    def simulate_trajectories_multiplicative(self, num_trajectories: int):
        trajectories = np.zeros((num_trajectories, self.time_steps + 1))
        for i in tqdm(range(num_trajectories)):
            x = self.x0
            for t in range(self.time_steps + 1):
                trajectories[i, t] = x
                x = self.a * x + np.random.normal(0, self.sigma) * x
        return trajectories

class gillespie_simulation:
    def __init__(self):
        pass
    """
    def gillespie_algorithm(self, b: float, d: float, n0: int, T: int):
        t = 0
        n = n0
        time_points = [t]
        population = [n]

        while t < T:
            rates = [b * n, d * n * (n - 1)]
            total_rate = sum(rates)
            dt = np.random.exponential(scale=1/total_rate)
            reaction = np.random.choice(range(2), p=[r/total_rate for r in rates])

            if reaction == 0:
                n += 1
            else:
                n -= 1

            t += dt
            time_points.append(t)
            population.append(n)

        return np.array(time_points), np.array(population)
    """

    def gillespie_algorithm(self, b: float, d: float, n0: int, T: int):

        time = 0
        population = n0
        time_points = [time]
        population_values = [population]

        while time < T:
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