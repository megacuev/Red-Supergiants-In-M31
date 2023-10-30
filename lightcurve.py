#Cenus of Red Supergiants in M31: Period fitting
#Megan Cuevas 

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from optparse import OptionParser
import celerite2
from celerite2 import terms
from scipy.optimize import minimize
import emcee
from astropy.stats import sigma_clip 

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

def set_params(params, t, yerr, y, gp):
    gp.mean = params[0]
    theta = np.exp(params[1:])
    #print(theta)
    gp.kernel = terms.SHOTerm(
        sigma=theta[0], rho=theta[1], tau=theta[2]
    ) + terms.SHOTerm(sigma=theta[3], rho=theta[4], Q=0.25)
    gp.compute(t, diag=yerr ** 2 + theta[5], quiet=True)
    return gp

def neg_log_like(params, t, yerr, y, gp):
    gp = set_params(params, t, yerr, y, gp)
    return -gp.log_likelihood(y)

def set_params_new(params, t_new, yerr_new, valid_y, gp):
    gp.mean = params[0]
    theta = np.exp(params[1:])
    gp.kernel = terms.SHOTerm(
        sigma=theta[0], rho=theta[1], tau=theta[2]
    ) + terms.SHOTerm(sigma=theta[3], rho=theta[4], Q=0.25)
    gp.compute(t_new, diag=yerr_new ** 2 + theta[5], quiet=True)
    return gp

def neg_log_like_new(params, t_new, yerr_new, valid_y, gp):
    gp = set_params_new(params, t_new, yerr_new, valid_y, gp)
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
        plt.ylabel("y [mag]")
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
        plt.ylabel("y [mag]")
        plt.xlim(min(t), max(t))
        plt.legend()
        plt.show()

def plot_residuals(gp,y,yerr):

    mu, variance = gp.predict(y, t=t, return_var=True)
    sigma = np.sqrt(variance)
    plt.plot(t, y-mu, 'ko')

    plt.xlabel("x [day]")
    plt.ylabel("y [mag]")
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

#period fitting of combined lightcurve data: ------------------------------------------
def PeriodFinder(jd_all, mag_all, omag_all, i):
    #taking our 3 individual arrays (jd_all, mag_all, omag_all) and put them into a 2 day array with 3 combined coloumns
    data = np.column_stack([jd_all, mag_all, omag_all])
    #print(data)
    #print(jd_all)

    # preproc the data
    t = jd_all
    yerr = omag_all
    y = mag_all
    true_t = np.linspace(min(t), max(t), 10**3)

    sigma1= 1
    rho1 = 400
    tau1 = 10000
    sigma2 = .1
    rho2 = 10000

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

    freq = np.linspace(1.0 / 10000, 1.0 / 100, 50000)
    omega = 2 * np.pi * freq
    #plt.title("initial psd")
    #plot_psd(gp)
    #plt.show()

    #plt.title("initial prediction")
    #plot_prediction(gp)
    #plt.show()


    # parameters: mean, sigma1, rho1, tau1, sigma2, rho2, errorscaling
    initial_params = [np.median(y), np.log(sigma1), np.log(rho1), np.log(tau1), np.log(sigma2), np.log(rho2), -1.]
    bnds = ((None, None), (None, None), (None, None), (np.log(rho1*3), None), (None, None), (np.log(rho1*5.), None), (None, None))
    print("Hello")
    soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", args=(t, yerr, y, gp,), bounds=bnds)
    print("Hi")
    opt_gp = set_params(soln.x, t, yerr, y, gp)
    print(soln)

    #print(" old Rho (period in days) = {}".format(np.exp(soln.x[2])))
    #print("old Median Magnitude of the Lightcurve = {}".format(soln.x[0]))
    #print("old Sigma of the Periodic term = {}".format(np.exp(soln.x[1])))
    #print("old Tau (damping of periodic term) = {} ".format(np.exp(soln.x[3])))
    #print("old Sigma of the Non-Periodic term = {}".format(np.exp(soln.x[4])))
    #print("old Tau (damping of the non-periodic term) = {}".format(np.exp(soln.x[5])))

    #plt.figure()
    #plt.title("maximum likelihood psd")
    #plot_psd(opt_gp)
    #plt.show()

    #plt.figure()
    #plt.title("maximum likelihood prediction")
    #plot_prediction(opt_gp)
    #plt.show()

    #sigma clipping:
    mu, variance = gp.predict(y, t=t, return_var=True)
    sigma = np.sqrt(variance)
    clip_data = y-mu
    filtered_data = sigma_clip(clip_data, sigma=3, maxiters=1)

    #Filtered Data Variables after Sigma Clipping:
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
    #plt.title("initial psd")
    #plot_psd(gp)
    #plt.show()

    #plt.title("initial prediction")
    #plot_prediction_new(gp)
    #plt.show()

    #---Starting fit with Filtered Data---:
    sigma1 = np.exp(soln.x[1])
    rho1 = np.exp(soln.x[2])
    tau1 = np.exp(soln.x[3])
    sigma2 = np.exp(soln.x[4])
    rho2 = np.exp(soln.x[5])

    # parameters: mean, sigma1, rho1, tau1, sigma2, rho2, errorscaling
    initial_params_new = [np.median(valid_y), np.log(sigma1), np.log(rho1), np.log(tau1), np.log(sigma2), np.log(rho2), -1.]
    bnds = ((None, None), (None, None), (None, None), (np.log(rho1*3), None), (None, None), (np.log(rho1*5.), None), (None, None))
    soln_new = minimize(neg_log_like_new, initial_params_new, method="L-BFGS-B", args=(t_new, yerr_new, valid_y, gp,), bounds=bnds)
    opt_gp = set_params_new(soln_new.x, t_new, yerr_new, valid_y, gp)

    #Printing the Solutions:
    #print("Rho (period in days) = {}".format(np.exp(soln_new.x[2])))
    #print("Median Magnitude of the Lightcurve = {}".format(soln_new.x[0]))
    #print("Sigma of the Periodic term = {}".format(np.exp(soln_new.x[1])))
    #print("Tau (damping of periodic term) = {} ".format(np.exp(soln_new.x[3])))
    #print("Sigma of the Non-Periodic term = {}".format(np.exp(soln_new.x[4])))
    #print("Rho (Period of the non-periodic term) = {}".format(np.exp(soln_new.x[5])))
    #np.savetxt("/Users/megan/Research/KupferResearch/CombinedLightCurves/" + i + "results" + ".txt",np.c_[np.exp(soln_new.x[2]), soln_new.x[0], np.exp(soln_new.x[1]), np.exp(soln_new.x[3]), np.exp(soln_new.x[4]), np.exp(soln_new.x[5])],  fmt='%.5f %g %g')

    line = f" {i} {np.exp(soln_new.x[2])} {soln_new.x[0]} {np.exp(soln_new.x[1])} {np.exp(soln_new.x[3])} {np.exp(soln_new.x[4])} {np.exp(soln_new.x[5])}\n"

    with open("results", "a") as results1:
        results1.write(line)


    #Plotting the Solutions:
    #plt.figure()
    #plt.title("maximum likelihood psd")
    #plot_psd(opt_gp)
    #plt.show()

    #plt.figure()
    #plt.title("maximum likelihood prediction")
    #plot_prediction_new(opt_gp)
    #plt.show()
#---------------------------------------------------------------------------

############################################################################
#Combining the ZTF and PTF data files and creating a combined lightcurve png and data set:
dtype1 = np.dtype([('filter', '|U5'), ('zpdiff', 'f8'),('JD', 'f8'),('flux','f8'),('flux_error','f8'),('nearestrefmag','f8'),('nearestrefmagunc','f8')])

ztf_data = "/Users/megan/Research/KupferResearch/Data/ZTF_Data"
ptf_data = "/Users/megan/Research/KupferResearch/Data/old_data"

ztf_list = [k[:-4] for k in os.listdir(ztf_data) if "J"==k[0]]
ptf_list = [k[:-4] for k in os.listdir(ptf_data) if "J"==k[0]]

combined_list = np.unique(ztf_list + ptf_list)

for i in combined_list:
    x = ztf_data + "/" + i + ".txt"

    if i + ".txt" in os.listdir(ztf_data) and False:
        zdata = np.loadtxt(x, usecols=[4,20,22,24,25,33,34], dtype=dtype1)

        filter = zdata['filter']
        ind = np.where(filter=='ZTF_r')
        zpdiff = zdata['zpdiff'][ind]
        jd = zdata['JD'][ind]
        flux = zdata['flux'][ind]
        flux_error = zdata['flux_error'][ind]
        nearestrefmag = zdata['nearestrefmag'][ind]
        nearestrefmagunc = zdata['nearestrefmagunc'][ind]

        nearestrefflux = 10**(0.4 * (zpdiff-nearestrefmag))
        nearestreffluxunc = nearestrefmagunc * nearestrefflux / 1.0857
        Fluxtot = flux + nearestrefflux
        Fluxunctot = np.sqrt( flux_error**2-nearestreffluxunc**2)
        SNRtot = Fluxtot/Fluxunctot
        ind1 = np.where(SNRtot > 3 )
        mag = zpdiff[ind1]-2.5*(np.log10(Fluxtot[ind1]))
        omag = 1.0857 / SNRtot[ind1]
        jd = jd[ind1] - 2400000.5 

    if i + ".txt" in os.listdir(ztf_data):
        jd = []
        zpdiff = []
        flux = []
        flux_error = []
        nearestrefmag = []
        nearestrefmagunc = []

        with open(x, "r") as readfile:
            for line in readfile.readlines():
                if line[0]=="#" or "null" in line:
                    continue
                if "ZTF_r" in line:
                    jd.append(line.split()[22])
                    zpdiff.append(line.split()[20])
                    flux.append(line.split()[24])
                    flux_error.append(line.split()[25])
                    nearestrefmag.append(line.split()[33])
                    nearestrefmagunc.append(line.split()[34])


        jd = np.array(jd, dtype = np.float64)
        zpdiff = np.array(zpdiff, dtype = np.float64)
        flux = np.array(flux, dtype = np.float64)
        flux_error = np.array(flux_error, dtype = np.float64) 
        nearestrefmag = np.array(nearestrefmag, dtype = np.float64) 
        nearestrefmagunc = np.array(nearestrefmagunc, dtype = np.float64)

        nearestrefflux = 10**(0.4 * (zpdiff-nearestrefmag))
        nearestreffluxunc = nearestrefmagunc * nearestrefflux / 1.0857
        Fluxtot = flux + nearestrefflux
        Fluxunctot = np.sqrt( flux_error**2-nearestreffluxunc**2)
        SNRtot = Fluxtot/Fluxunctot
        ind1 = np.where(SNRtot > 3 )
        mag = zpdiff[ind1]-2.5*(np.log10(Fluxtot[ind1]))
        omag = 1.0857 / SNRtot[ind1]
        jd = jd[ind1] - 2400000.5   

        
    else:
        continue
    pdata = np.loadtxt(ptf_data + "/" + i + ".csv", delimiter = ',', skiprows = 1)


    jd_all = np.append(pdata[:,0], jd)
    mag_all = np.append(pdata[:,1], mag)
    omag_all = np.append(pdata[:,2], omag)

    np.savetxt("/Users/megan/Research/KupferResearch/CombinedLightCurves/" + i + ".txt",np.c_[jd_all,mag_all,omag_all], fmt='%.5f %g %g')


    plt.plot(jd_all, mag_all, 'ko')
    plt.title(i)
    plt.xlabel('Days (MJD)')
    plt.ylabel('Magnitude')
    plt.savefig("/Users/megan/Research/KupferResearch/CombinedLightCurves/" + i + ".png")
    plt.clf()
    plt.cla()
    
    try:
        PeriodFinder(jd_all, mag_all, omag_all, i)

    except:
        with open("results", "a") as results1:
            results1.write(f"{i} failed\n")

