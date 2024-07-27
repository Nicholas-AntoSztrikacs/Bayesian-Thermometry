import numpy as np
from scipy import integrate


def fisher_information(sigma):
    return 2 / sigma**2

# Bernstein von-misses theorem. Posterior converges to gaussian(0,NF**-1)
def bernstein_von_misses_distribution(N, sigma, sigma_true):
    F = fisher_information(sigma_true)
    return np.sqrt(N * F / (2 * np.pi)) * \
        np.exp(-N * F * (sigma - sigma_true)**2 / 2)


def bayes_estimator(distribution, sig):
    Estimator = np.zeros(len(distribution))
    for i in range(len(Estimator)):
        Estimator[i] = integrate.simpson(sig * distribution[i, :], sig)
    return Estimator


def optimal_estimator(distribution, sig):
    Estimator = np.zeros(len(distribution))
    for i in range(len(Estimator)):
        Estimator[i] = np.exp(integrate.simpson(
            np.log(sig) * distribution[i, :], sig))
    return Estimator


def prior_estimator(Prior, sig):
    return np.exp(integrate.simpson(np.log(sig) * Prior, sig))


def prior_error(Prior, estimate, sig):
    return integrate.simpson((np.log(estimate / sig))**2 * Prior, sig)


def log_error(sig_Estimate, sig_prior):
    return (np.log(sig_Estimate / sig_prior))**2


def mean_square_error(sigEstimate, sigTrue):
    return (sigEstimate - sigTrue)**2


def posterior_loss_error(distribution, sigEstimate, sig):
    EpsLoss = np.zeros_like(sigEstimate)
    for i in range(len(sigEstimate)):
        EpsLoss[i] = integrate.simpson(
            (sigEstimate[i] - sig)**2 * distribution[i, :], sig)
    return EpsLoss
