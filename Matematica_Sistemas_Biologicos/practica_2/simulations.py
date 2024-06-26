from tqdm import tqdm
import numpy as np

class binomial_simulation:
    def __init__(self, num_simulations: int, N0: int):
        self.num_simulations = num_simulations
        self.N0 = N0 
    
    def simulate_population(self, d: float, time_steps: int):
        population = self.N0
        populations = [self.N0]
        for t in range(1, time_steps + 1):
            alive = np.random.binomial(population, 1 - d)
            populations.append(alive)
            population = alive
        return populations

    def simulate_population_at_time(self,  d: float, time_step: int):
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
                if x <= 0:
                    break
        return trajectories

    def simulate_trajectories_multiplicative(self, num_trajectories: int):
        trajectories = np.zeros((num_trajectories, self.time_steps + 1))
        for i in tqdm(range(num_trajectories)):
            x = self.x0
            for t in range(self.time_steps + 1):
                trajectories[i, t] = x
                x = self.a * x + np.random.normal(0, self.sigma) * x
                if x <= 0:
                    break
        return trajectories

class gillespie_simulation:
    def __init__(self):
        pass

    def sol_logistic(self, b: float, d: float, n0: int, T: np.arange):
        N = b / (d - (d - b/n0) * np.exp(-b * T))
        return N

    def gillespie_algorithm(self, b: float, d: float, n0: int, T: int):

        time = 0
        population = n0
        time_points = [time]
        population_values = [population]

        while time < T:
            reproduction_rate = b * population
            competition_rate = d * population * (population - 1)
            total_rate = reproduction_rate + competition_rate

            delta_t = np.random.exponential(1 / total_rate)

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