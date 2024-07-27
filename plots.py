import numpy as np
import matplotlib.pyplot as plt


def plot_distributions(posterior, prior, sig_arr, sigma, asymptotic):
    LW = 3
    plt.figure(1)
    plt.plot(sig_arr, asymptotic, '--', linewidth=LW, label='BVM Distribution')
    plt.plot(sig_arr, posterior[0, 500, :],
             alpha=0.65, linewidth=LW, label='N=500')
    plt.plot(sig_arr, posterior[0, 50, :],
             alpha=0.55, linewidth=LW, label='N=50')
    plt.plot(sig_arr, posterior[0, 5, :],
             alpha=0.45, linewidth=LW, label='N=5')
    plt.plot(sig_arr, prior, linewidth=LW, alpha=0.35, label='N=0',)
    plt.axvline(x=sigma, color='k', linestyle=':', label="True")
    plt.xlabel(r"$\sigma$")
    plt.ylabel(r"P($\sigma$)")
    plt.legend()
    return


def plot_estimates(SigBayes, OptBayes, MLE_Est, sigma):
    LW = 3
    plt.figure(2)
    plt.semilogx(np.transpose(SigBayes[0:1]),
                 '-', linewidth=LW, label='Bayes estimate')
    plt.semilogx(np.transpose(OptBayes[0:1]), '--',
                 linewidth=LW, label='Optimal estimate')
    plt.semilogx(np.transpose(MLE_Est[0:1]),
                 ':', linewidth=LW, label='MLE Estimate')
    plt.axhline(
        sigma,
        color='k',
        linewidth=5,
        linestyle='--',
        label='True Value')
    plt.xlabel('N_steps')
    plt.ylabel(r'$\sigma$')
    plt.legend()


def plot_all_estimates(SigBayes, OptBayes, MLE_Est, sigma):
    LW = 3
    plt.figure(2)
    plt.semilogx(
        np.transpose(SigBayes),
        '-',
        linewidth=LW,
        label='Bayes estimate')
    plt.semilogx(
        np.transpose(OptBayes),
        '--',
        linewidth=LW,
        label='Optimal estimate')
    plt.semilogx(
        np.transpose(MLE_Est),
        ':',
        linewidth=LW,
        label='MLE Estimate')
    plt.axhline(
        sigma,
        color='k',
        linewidth=5,
        linestyle='--',
        label='True Value')
    plt.xlabel('N_steps')
    plt.ylabel(r'$\sigma$')


def plot_errors(MSEBayes, LogBayes, MLE_Var, MSE, LogMSE, sigma):
    N = np.logspace(0, np.log10(len(MSEBayes)))
    LW = 3
    plt.figure(4)
    plt.loglog(MSEBayes, linewidth=LW, label='Averaged MSE')
    plt.loglog(LogBayes, linewidth=LW, label='Averaged Log Error')
    plt.loglog(np.average(MLE_Var, axis=0), linewidth=LW, label='MLE variance')
    #plt.loglog(MLE_Var[0,:],linewidth=LW,label = 'MLE variance Single shot variance')
    plt.loglog(MSE[0, :], '--', linewidth=LW,
               alpha=0.35, label='MSE Single shot')
    plt.loglog(LogMSE[0, :], '--', linewidth=LW,
               alpha=0.35, label='Log MSE Single shot')
    plt.loglog(N, sigma**2 / (2 * N), 'k:', linewidth=LW, label='Cramer Rao')
    plt.xlabel('N_steps')
    plt.ylabel('Error (non Bayesian)')
    plt.legend()
