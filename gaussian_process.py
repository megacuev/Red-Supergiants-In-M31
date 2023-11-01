import numpy as np
import matplotlib.pyplot as plt
import celerite2
from celerite2 import terms
from scipy.optimize import minimize
import emcee
from optparse import OptionParser
from astropy.stats import sigma_clip 

parser = OptionParser()
(options,args) = parser.parse_args()

file = args[0][:-4]
print(file)

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
    #print(theta)
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
#############################################################
#PTF and ZTF combininer:
    #use pathways to ZTF file and PTF file to find corresponding file name and combine
    # filennames = glob.glob("folder/*"), filename is array with all filenames in that folder 
        #for i in len(filenames):
        #ZTF = np.loadtxt(filenames[i],....)

#dtype1 = np.dtype([('filter', '|U5'), ('zpdiff', 'f8'),('JD', 'f8'),('flux','f8'),('flux_error','f8'),('nearestrefmag','f8'),('nearestrefmagunc','f8')])

#ZTF = np.loadtxt(args[0], usecols=[4,20,22,24,25,33,34], dtype=dtype1)
ZTF = np.loadtxt(args[0], usecols = [0,1,2])
#clean ZTF here:
#PTF = np.loadtxt(args[1], delimiter = ',',skiprows = 1)


#filter = ZTF['filter']
#ind = np.where(filter=='ZTF_r')
#zpdiff = ZTF['zpdiff'][ind]
t = ZTF[:,0] - 2400000
y = ZTF[:,1]
yerr = ZTF[:,2]
#nearestrefmag = ZTF['nearestrefmag'][ind]
#nearestrefmagunc = ZTF['nearestrefmagunc'][ind]

#nearestrefflux = 10**(0.4 * (zpdiff-nearestrefmag))
#nearestreffluxunc = nearestrefmagunc * nearestrefflux / 1.0857
#Fluxtot = flux + nearestrefflux
#Fluxunctot = np.sqrt( flux_error**2-nearestreffluxunc**2)
#SNRtot = Fluxtot/Fluxunctot

#ind1 = np.where(SNRtot > 3 )
    # we have a “confident” detection, compute and plot mag with error bar:
#mag = zpdiff[ind1]-2.5*(np.log10(Fluxtot[ind1]))
#print(mag)
#omag = 1.0857 / SNRtot[ind1]
#jd = jd[ind1] - 2400000.5
#jd_all = np.append(PTF[:,0], jd)
#mag_all = np.append(PTF[:,1], mag)
#omag_all = np.append(PTF[:,2], omag)

# np.savetext('comb_' + filenames[i],...)
#np.savetxt(args[2],np.c_[jd_all,mag_all,omag_all], fmt='%.5f %g %g')
#############################################################
#period fitting of combined lightcurve data:
true_t = np.linspace(min(t), max(t), 10**3)
freq = np.linspace(1.0 / 10000, 1.0 / 100, 50000)
omega = 2 * np.pi * freq

def PeriodFinder(t, y, yerr):
    #taking our 3 individual arrays (jd_all, mag_all, omag_all) and put them into a 2 day array with 3 combined coloumns
    #data = np.column_stack([jd, mag, omag_all])
    #print(data)

    # preproc the data
    #t = jd
    #yerr = flux_error
    #y = flux
    true_t = np.linspace(min(t), max(t), 10**3)

    sigma1= 100
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
    print("old Median Magnitude of the Lightcurve = {}".format(soln.x[0]))
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
    soln_new = minimize(neg_log_like_new, initial_params_new, method="L-BFGS-B", args=(gp,), bounds=bnds)
    opt_gp = set_params_new(soln_new.x, gp)

    #Printing the Solutions:
    print("Rho (period in days) = {}".format(np.exp(soln_new.x[2])))
    print("Median Magnitude of the Lightcurve = {}".format(soln_new.x[0]))
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

PeriodFinder(t, y, yerr)

#EMCEE:
#############################################################
'''

np.random.seed(5693854)
coords = soln.x + 1e-5 * np.random.randn(32, len(soln.x))
sampler = emcee.EnsembleSampler(
    coords.shape[0], coords.shape[1], log_prob, args=(gp,)
)
state = sampler.run_mcmc(coords, 20, progress=True)
sampler.reset()
state = sampler.run_mcmc(state, 200, progress=True)



chain = sampler.get_chain(discard=100, flat=True)

for sample in chain[np.random.randint(len(chain), size=50)]:
    gp = set_params(sample, gp)
    conditional = gp.condition(y, true_t)
    plt.plot(true_t, conditional.sample(), color="C0", alpha=0.1)
    #plt.show()

plt.title("posterior prediction")
plot_prediction(None)
#plt.savefig('psd_'+file+'.pdf')

best = np.argmax(sampler.flatlnprobability)
bestpars = sampler.flatchain[best]

print("Rho (period in days) = {}".format(np.median(np.exp(sampler.flatchain[:,2]), axis=0)))
print("Standard Deviation of Rho = {}".format(np.std(np.exp(sampler.flatchain[:,2]), axis=0)))


print("Median Magnitude of the Lightcurve ??????? = {}".format(np.median(np.exp(sampler.flatchain[:,0]), axis=0)))
print("Sigma of the Periodic term = {}".format(np.median(np.exp(sampler.flatchain[:,1]), axis=0)))
print("Tau (damping of periodic term) = {} ".format(np.median(np.exp(sampler.flatchain[:,3]), axis=0)))
print("Sigma of the Non-Periodic term = {}".format(np.median(np.exp(sampler.flatchain[:,4]), axis=0)))
print("Tau (damping of the non-periodic term) = {}".format(np.median(np.exp(sampler.flatchain[:,5]), axis=0)))


psds = sampler.get_blobs(discard=100, flat=True)

q = np.percentile(psds, [16, 50, 84], axis=0)

plt.loglog(freq, q[1], color="C0")
plt.fill_between(freq, q[0], q[2], color="C0", alpha=0.1)

plt.xlim(freq.min(), freq.max())
plt.xlabel("frequency [1 / day]")
plt.ylabel("power [day ppt$^2$]")
plt.title("posterior psd using emcee")
#plt.savefig('solution_'+file+'.pdf')
plt.show()
#print(freq[np.argmax(q[1])])
#plt.savefig('power.pdf')

'''

##########################################################################
