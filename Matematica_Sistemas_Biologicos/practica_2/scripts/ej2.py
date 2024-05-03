import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def additive_noise_simulation(a, sigma, time_steps, num_trajectories):
    trajectories = np.zeros((num_trajectories, time_steps+1))
    for i in range(num_trajectories):
        x = np.zeros(time_steps+1)
        x[0] = 1
        for t in range(time_steps):
            z = np.random.normal(0, sigma)
            x[t+1] = a * x[t] + z
        trajectories[i] = x
    return trajectories

def multiplicative_noise_simulation(a, sigma, time_steps, num_trajectories):
    trajectories = np.zeros((num_trajectories, time_steps+1))
    for i in range(num_trajectories):
        x = np.zeros(time_steps+1)
        x[0] = 1
        for t in range(time_steps):
            z = np.random.normal(0, sigma)
            x[t+1] = a * x[t] + z * x[t]
        trajectories[i] = x
    return trajectories

def plot_trajectories(trajectories, time_steps, title):
    plt.figure(figsize=(10, 6))
    for i in range(len(trajectories)):
        plt.plot(range(time_steps+1), trajectories[i], alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('x')
    plt.grid(True)
    plt.show()

def plot_probability_distribution(trajectories, time_steps, bins=30):
    final_states = trajectories[:, -1]
    plt.figure(figsize=(10, 6))
    plt.hist(final_states, bins=bins, density=True, alpha=0.7)
    plt.title('Probability Distribution')
    plt.xlabel('Final State (x)')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.show()

def plot_3d_probability_distribution(trajectories, time_steps, bins=30):
    final_states = trajectories[:, -1]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(final_states, range(time_steps+1), bins=bins, density=True)
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = 0.1 * np.ones_like(zpos)
    dz = hist.ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax.set_xlabel('Final State (x)')
    ax.set_ylabel('Time')
    ax.set_zlabel('Probability Density')
    plt.title('Probability Distribution')
    plt.show()
# Parameters
a = 1.05
sigma = 0.2
time_steps = 50
num_trajectories = 12

# Additive Noise Simulation
additive_trajectories = additive_noise_simulation(a, sigma, time_steps, num_trajectories)
plot_trajectories(additive_trajectories, time_steps, 'Additive Noise Trajectories')
plot_probability_distribution(additive_trajectories, time_steps)
plot_3d_probability_distribution(additive_trajectories, time_steps)

# Multiplicative Noise Simulation
multiplicative_trajectories = multiplicative_noise_simulation(a, sigma, time_steps, num_trajectories)
plot_trajectories(multiplicative_trajectories, time_steps, 'Multiplicative Noise Trajectories')
plot_probability_distribution(multiplicative_trajectories, time_steps)
plot_3d_probability_distribution(multiplicative_trajectories, time_steps)
