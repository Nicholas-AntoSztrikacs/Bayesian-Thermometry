import numpy as np
from scipy import integrate

# prior distribution function
def prior_distribution(sig, sigmax, sigmin):
    return (sig * np.log(sigmax / sigmin))**(-1)

# x is the displacement in a time step dt
def likelihood(x, t, sig):
    return (1 / np.sqrt(2 * np.pi * sig**2 * t)) * \
        np.exp(-x**2 / (2 * sig**2 * t))


def compute_log_likelihood(trajectory, dt, sig):
    N_steps = len(trajectory)
    Loglikelihood = np.zeros((N_steps, len(sig)))
    for i in range(1, N_steps):
        # dx between two time steps dt apart
        diff = trajectory[i] - trajectory[i - 1]
        for j in range(len(sig)):
            Loglikelihood[i, j] = Loglikelihood[i - 1, j] + \
                np.log(likelihood(diff, dt, sig[j]))
    return Loglikelihood


def compute_posterior(Loglikelihood_dist, prior_dist, sigma):
    N_steps = len(Loglikelihood_dist)
    nn = len(Loglikelihood_dist[0])
    Pq = np.zeros(N_steps)
    posterior = np.zeros((N_steps, nn))

    for i in range(N_steps):
        for j in range(nn):
            Pq[i] = Pq[i] + (np.exp(Loglikelihood_dist[i, j] -
                             max(Loglikelihood_dist[i, :])) * prior_dist[j])
    for i in range(N_steps):
        for j in range(nn):
            posterior[i, j] = (np.exp(Loglikelihood_dist[i, j] - \
                               max(Loglikelihood_dist[i, :])) * prior_dist[j]) / Pq[i]

    for i in range(N_steps):
        posterior[i, :] = posterior[i, :] / \
            integrate.simpson(posterior[i, :], sigma)
    return posterior
