import numpy as np
import matplotlib.pyplot as plt
from random_walks import *
from scipy import integrate
from bayes_distributions import *
from estimators import *
from maximum_likelihood import *
from plots import *

################ Initialize simulation prameters ########################

# Set the seed for debugging
np.random.seed(99)

# Parameters
sigma = 0.25  # standard deviation # The 'true' parameter
t0 = 0  # initial time
t_end = 1  # end time
N_steps = 501  # number of time steps
N_sims = 50  # number of random walkers
dt = (t_end - t0) / N_steps  # time interval
t_array = np.linspace(t0, t_end, N_steps, endpoint=False)  # time array

# Create a list containing the possible sigma values
sig_min = 0.01
sig_max = 1
nn = 300  # number of points in sigma array
sig_arr = np.linspace(sig_min, sig_max, nn)

################ Setup the problem ########################

# Define the prior distribution of sigma and normalize it
prior = prior_distribution(sig_arr, sig_max, sig_min)
prior = prior / integrate.simpson(prior, sig_arr)

# Find the estimate of sigma before measuring
sigma_prior = prior_estimator(prior, sig_arr)
uncertainty_prior = prior_error(prior, sigma_prior, sig_arr)

# define the asymptotic distribution and normalize it
asymptotic = bernstein_von_misses_distribution(N_steps, sig_arr, sigma)
asymptotic = asymptotic / integrate.simpson(asymptotic, sig_arr)

# Generate the random walk trajectories
Trajectory = walker(N_steps, N_sims, sigma, initial_value=0, dt=dt)

# Define arrays for the posterior and likelihood distributions
Loglikelihood = np.zeros((N_sims, N_steps, nn))
posterior = np.zeros((N_sims, N_steps, nn))

# Define the empty arrays for the various estimators
SigBayes = np.zeros((N_sims, N_steps))
OptBayes = np.zeros((N_sims, N_steps))
MSE = np.zeros((N_sims, N_steps))
LogMSE = np.zeros((N_sims, N_steps))
PostLoss = np.zeros((N_sims, N_steps))
LogLoss = np.zeros((N_sims, N_steps))
MLE_Est = np.zeros((N_sims, N_steps - 1))
MLE_Var = np.zeros((N_sims, N_steps - 1))

################ Do calculations ########################
for k in range(N_sims):

    # Obtain the posterior and Loglikelihood distributions
    Loglikelihood[k, :, :] = compute_log_likelihood(Trajectory[:, k], dt, sig_arr)
    posterior[k, :, :] = compute_posterior(Loglikelihood[k, :, :], prior, sig_arr)

    # Calculate Bayesian estimates
    SigBayes[k, :] = bayes_estimator(posterior[k, :, :], sig_arr)
    OptBayes[k, :] = optimal_estimator(posterior[k, :, :], sig_arr)

    # Calculate Bayesian error measures
    MSE[k, :] = mean_square_error(SigBayes[k, :], sigma)  # need to average this
    PostLoss[k, :] = posterior_loss_error(posterior[k, :, :], SigBayes[k, :], sig_arr)

    LogMSE[k, :] = log_error(OptBayes[k, :], sigma)  # need to average this
    # average over realizations
    LogLoss[k, :] = log_error(OptBayes[k, :], sigma_prior)

    # MLE Estimate and Variance for a check
    MLE_Est[k, :] = mle_estimate(Trajectory[:, k], dt)
    MLE_Var[k, :] = mle_variance(MLE_Est[k, :])

# Average the Mean square and log error functions
AveragedMSE = np.average(MSE, axis=0)
AveragedLogMSE = np.average(LogMSE, axis=0)


################ Plots ########################
# Plot random walk trajectories
plot_trajectory(Trajectory, t_array, 20)

# Plot the distributions changing with number of measurements
plot_distributions(posterior, prior, sig_arr, sigma, asymptotic)

# Plot the all estimators evolving with number of steps for a single trajectory
plot_estimates(SigBayes, OptBayes, MLE_Est, sigma)
plot_all_estimates(SigBayes, OptBayes, MLE_Est, sigma)

# Plot frequentist Errors with number of steps
plot_errors(AveragedMSE, AveragedLogMSE, MLE_Var, MSE, LogMSE, sigma)
