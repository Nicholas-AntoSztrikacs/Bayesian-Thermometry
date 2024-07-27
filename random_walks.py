import numpy as np
import matplotlib.pyplot as plt

# Given a random walk process: dX = sigma*dW estimate sigma
# Since we know that the standard deviation of the walk
# is related to the temperature of the condensate

# sigma has units of X/sqrt(t)
# dW has units of sqrt(t)
def walker(num_steps, num_simulations, sigma, initial_value=0, dt=0.01):
    # Wiener increments
    dW = np.random.normal(loc=0, scale=1, size=(num_steps, num_simulations))

    # Initialize the process trajectories
    X = np.zeros((num_steps, num_simulations))
    X[0, :] = initial_value

    for i in range(1, num_steps):
        X[i, :] = X[i - 1, :] + sigma * np.sqrt(dt) * dW[i, :]
    return X


def plot_trajectory(X, t_array, Num_trajectories):
    plt.figure(72)
    plt.plot(t_array, X[:, 0:Num_trajectories], linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Position")
    return
