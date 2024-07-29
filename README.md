# Bayesian-Thermometry
This repository contains a set of Python scripts to estimate the volatility parameter of a particle undergoing a random walk. This parameter is then used a proxy to estimate the temperature of a substance of interest to perform thermometry. The approach relies on simulating a collection of Brownian trajectories and updating the proability distribution function for the volatility using Bayes theorem. After many time steps, the probability distribution converges to a steady state distribution from which the statistics of the volatility parameter may be estimated. This project also compares the Bayesian approach to maximum likelihood estimates of the volatility, and uses various estimators to obtain the volatility. the output of this program should produce various figures showcasing the evolution of the probability distribution function and the convergence of the estimators.

## Getting Started
If you would like to use or add to this repository you can either clone or download it

```
git clone https://github.com/Nicholas-AntoSztrikacs/Bayesian-Thermometry
```

The main file in this repository is `bayesian_thermometry.py`. Here, the simulation parameters are provided such that the random walk trajectories may be performed in `random_walks.py` and the probability distribution functions are then computed in `bayes_distributions.py`. Following this, the various estimates are computed via `estimators.py`. Lastly, `maximum_likelihood.py` produces a comparison with maximum likelihood approaches and the output plots are generated using `plots.py

## Prerequisites
This repository makes use of external packages: `numpy`, `scipy`, and `matplotlib`. You can install them using `pip` with the following code:
```
pip install numpy scipy matplotlib
```



