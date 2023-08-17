#Megan Cuevas
#August 15th, 2023

import numpy as np
import os
import matplotlib.pyplot as plt
import celerite2
from celerite2 import terms
from scipy.optimize import minimize
import emcee
from optparse import OptionParser
from astropy.stats import sigma_clip 

parser = OptionParser()
(options,args) = parser.parse_args()

#Loading in the data:
ZTF = np.loadtxt(args[0], usecols=[0,1,2])
jd = ZTF[:,0]
flux = ZTF[:,1]
flux_error = ZTF[:,2]


#Definitions:
#############################################################################
def plot_psd(gp):
    for n, term in enumerate(gp.kernel.terms):
        plt.loglog(freq, term.get_psd(omega), label="term {0}".format(n + 1))
    plt.loglog(freq, gp.kernel.get_psd(omega), ":k", label="full model")
    plt.xlim(freq.min(), freq.max())
    plt.legend()
    plt.xlabel("frequency [1 / day]")
    plt.ylabel("power [day ppt$^2$]")
    plt.show()

def set_params(params, gp):
    gp.mean = params[0]
    theta = np.exp(params[1:])
    gp.kernel = terms.SHOTerm(
        sigma=theta[0], rho=theta[1], tau=theta[2]
    ) + terms.SHOTerm(sigma=theta[3], rho=theta[4], Q=0.25)
    gp.compute(t, diag=yerr ** 2 + theta[5], quiet=True)
    return gp

def neg_log_like(params, gp):
    gp = set_params(params, gp)
    return -gp.log_likelihood(y)

def set_params_new(params, gp):
    gp.mean = params[0]
    theta = np.exp(params[1:])
    gp.kernel = terms.SHOTerm(
        sigma=theta[0], rho=theta[1], tau=theta[2]
    ) + terms.SHOTerm(sigma=theta[3], rho=theta[4], Q=0.25)
    gp.compute(t_new, diag=yerr_new ** 2 + theta[5], quiet=True)
    return gp

def neg_log_like_new(params, gp):
    gp = set_params_new(params, gp)
    return -gp.log_likelihood(valid_y)

def plot_prediction(gp):
#    plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3, label="data")
    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="truth")

    if gp:
        mu, variance = gp.predict(y, t=true_t, return_var=True)
        sigma = np.sqrt(variance)
        plt.plot(true_t, mu, label="prediction")
        plt.fill_between(true_t, mu - sigma, mu + sigma, color="C0", alpha=0.2)

        plt.xlabel("x [day]")
        plt.ylabel("y [flux difference]")
        plt.xlim(min(t), max(t))
        plt.legend()
        plt.show()

def plot_prediction_new(gp):
    plt.errorbar(t_new, valid_y, yerr=yerr_new, fmt=".k", capsize=0, label="truth")

    if gp:
        mu, variance = gp.predict(valid_y, t=true_t, return_var=True)
        sigma = np.sqrt(variance)
        plt.plot(true_t, mu, label="prediction")
        plt.fill_between(true_t, mu - sigma, mu + sigma, color="C0", alpha=0.2)

        plt.xlabel("x [day]")
        plt.ylabel("y [flux difference]")
        plt.xlim(min(t), max(t))
        plt.legend()
        plt.show()

def plot_residuals(gp,y,yerr):

    mu, variance = gp.predict(y, t=t, return_var=True)
    sigma = np.sqrt(variance)
    plt.plot(t, y-mu, 'ko')

    plt.xlabel("x [day]")
    plt.ylabel("y [flux difference]")
    plt.xlim(min(t), max(t))
    plt.show()

def log_prob(params, gp):
    gp = set_params(params, gp)
    output =  (
        gp.log_likelihood(y) - 0.5 * np.sum((params / prior_sigma) ** 2),
        gp.kernel.get_psd(omega),
    )

    # if it cannot be computed, it is a bad solution:
    if np.isnan(output[0]):
        output = (-np.inf,gp.kernel.get_psd(omega) )

    return output

def PeriodFinder():
    #taking our 3 individual arrays (jd_all, mag_all, omag_all) and put them into a 2 day array with 3 combined coloumns
    data = np.column_stack([jd, flux, flux_error])

    # preproc the data
    global t, y, yerr, true_t
    t = jd
    y = flux
    yerr = flux_error
    true_t = np.linspace(min(t), max(t), 10**3)

    sigma1= 1
    rho1 = 300
    tau1 = 1000
    sigma2 = .1
    rho2 = 300

    # Define the kernel
    #Quasi-periodic term
    term1 = terms.SHOTerm(sigma = sigma1, rho=rho1, tau=tau1)

    #Non-periodic component
    term2 = terms.SHOTerm(sigma=sigma2, rho=rho2, Q=0.25)
    kernel = term1 + term2

    # Setup the GP
    gp = celerite2.GaussianProcess(kernel, mean=np.median(y))
    gp.compute(t, yerr=yerr)

    print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))

    global freq, omega
    freq = np.linspace(1.0 / 10000, 1.0 / 100, 50000)
    omega = 2 * np.pi * freq
    plt.title("initial psd")
    plot_psd(gp)
    plt.show()

    plt.title("initial prediction")
    plot_prediction(gp)
    plt.show()


    # parameters: mean, sigma1, rho1, tau1, sigma2, rho2, errorscaling
    initial_params = [np.median(y), np.log(sigma1), np.log(rho1), np.log(tau1), np.log(sigma2), np.log(rho2), -1.]
    bnds = ((None, None), (None, None), (None, None), (np.log(rho1*3), None), (None, None), (np.log(rho1*5.), None), (None, None))
    soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", args=(gp,), bounds=bnds)
    opt_gp = set_params(soln.x, gp)
    print(soln)

    print(" old Rho (period in days) = {}".format(np.exp(soln.x[2])))
    print("old Median Flux of the Lightcurve = {}".format(soln.x[0]))
    print("old Sigma of the Periodic term = {}".format(np.exp(soln.x[1])))
    print("old Tau (damping of periodic term) = {} ".format(np.exp(soln.x[3])))
    print("old Sigma of the Non-Periodic term = {}".format(np.exp(soln.x[4])))
    print("old Tau (damping of the non-periodic term) = {}".format(np.exp(soln.x[5])))

    plt.figure()
    plt.title("maximum likelihood psd")
    plot_psd(opt_gp)
    plt.show()

    plt.figure()
    plt.title("maximum likelihood prediction")
    plot_prediction(opt_gp)
    plt.show()

    #sigma clipping:
    mu, variance = gp.predict(y, t=t, return_var=True)
    sigma = np.sqrt(variance)
    clip_data = y-mu
    filtered_data = sigma_clip(clip_data, sigma=4, maxiters=1)

    #Filtered Data Variables after Sigma Clipping:
    global valid_y, t_new, yerr_new
    valid_y = filtered_data.data[~filtered_data.mask]+mu[~filtered_data.mask]
    t_new = t[~filtered_data.mask]
    yerr_new = yerr[~filtered_data.mask]

    #Setting up the Kernel for the Filtered Data:
    prior_sigma = 2.0

    gp = celerite2.GaussianProcess(kernel, mean=np.median(valid_y))
    gp.compute(t_new, yerr=yerr_new)

    print("Initial log likelihood: {0}".format(gp.log_likelihood(valid_y)))

    #Showing new Filtered Data Plots:
    freq = np.linspace(1.0 / 10000, 1.0 / 100, 50000)
    omega = 2 * np.pi * freq
    plt.title("initial psd")
    plot_psd(gp)
    plt.show()

    plt.title("initial prediction")
    plot_prediction_new(gp)
    plt.show()

    #---Starting fit with Filtered Data---:
    sigma1 = np.exp(soln.x[1])
    rho1 = np.exp(soln.x[2])
    tau1 = np.exp(soln.x[3])
    sigma2 = np.exp(soln.x[4])
    rho2 = np.exp(soln.x[5])

    # parameters: mean, sigma1, rho1, tau1, sigma2, rho2, errorscaling
    initial_params_new = [np.median(valid_y), np.log(sigma1), np.log(rho1), np.log(tau1), np.log(sigma2), np.log(rho2), -1.]
    bnds = ((None, None), (None, None), (None, None), (np.log(rho1*3), None), (None, None), (np.log(rho1*5.), None), (None, None))
    soln_new = minimize(neg_log_like_new, initial_params_new, method="L-BFGS-B", args=(gp,), bounds=bnds)
    opt_gp = set_params_new(soln_new.x, gp)

    #Printing the Solutions:
    print("Rho (period in days) = {}".format(np.exp(soln_new.x[2])))
    print("Median Flux of the Lightcurve = {}".format(soln_new.x[0]))
    print("Sigma of the Periodic term = {}".format(np.exp(soln_new.x[1])))
    print("Tau (damping of periodic term) = {} ".format(np.exp(soln_new.x[3])))
    print("Sigma of the Non-Periodic term = {}".format(np.exp(soln_new.x[4])))
    print("Rho (Period of the non-periodic term) = {}".format(np.exp(soln_new.x[5])))


    #Plotting the Solutions:
    plt.figure()
    plt.title("maximum likelihood psd")
    plot_psd(opt_gp)
    plt.show()

    plt.figure()
    plt.title("maximum likelihood prediction")
    plot_prediction_new(opt_gp)
    plt.show()

PeriodFinder()
