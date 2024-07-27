import numpy as np

# Estimate the sigma parameter using MLE
# Option 1. Using two adjacent points in the timeseries and summing their squar
# this can be done by hand to give: sum(dx**2)/(N*dt)
#Variance is 2*sigma**2/sqrt(N)


def mle_estimate(trajectory, dt):
    Estimator = np.zeros(len(trajectory) - 1)
    running_sum = 0
    for i in range(1, len(Estimator)):
        squared_diff = (trajectory[i] - trajectory[i - 1])**2

        # Update the running sum
        running_sum += squared_diff

        # Estimate sigma as the sqrt of the running sum divided by the count
        Estimator[i] = np.sqrt(running_sum / i)
    return Estimator / np.sqrt(dt)

# Option 2.
# MLE Variance of the sigma estimate, comes from the CRB/Fisher info


def mle_variance(Estimate):
    Var = np.zeros(len(Estimate))
    for i in range(0, len(Estimate)):
        Var[i] = (Estimate[i] / (2 * (i + 1)))
    return Var
