import numpy as np
import matplotlib.pyplot as plt
import celerite2
from celerite2 import terms
from scipy.optimize import minimize
import emcee
from optparse import OptionParser

parser = OptionParser()
(options,args) = parser.parse_args()

file = args[0][:-4]
print(file)

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



# load the data
data = np.loadtxt(args[0],delimiter=" ", skiprows=1)

# preproc the data
t = data[:,0]
yerr = data[:,2]
y = data[:,1]
true_t = np.linspace(min(t), max(t), 10**3)



# Define the kernel
#Quasi-periodic term
term1 = terms.SHOTerm(sigma=args[1], rho=args[2], tau=args[3])

#Non-periodic component
term2 = terms.SHOTerm(sigma=args[4], rho=args[5], Q=0.25)
kernel = term1 + term2

# Setup the GP
gp = celerite2.GaussianProcess(kernel, mean=np.median(data[:,1]))
gp.compute(t, yerr=yerr)

print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))


freq = np.linspace(1.0 / 10000, 1.0 / 100, 50000)
omega = 2 * np.pi * freq
plt.title("initial psd")
plot_psd(gp)
plt.show()

plt.title("initial prediction")
plot_prediction(gp)
plt.show()


sigma1=np.float(args[1])
rho1 = np.float(args[2])
tau1 = np.float(args[3])
sigma2 = np.float(args[4])
rho2 = np.float(args[5])

# parameters: mean, sigma1, rho1, tau1, sigma2, rho2, errorscaling
initial_params = [np.median(data[:,1]), np.log(sigma1), np.log(rho1), np.log(tau1), np.log(sigma2), np.log(rho2), -1.]
bnds = ((None, None), (None, None), (None, None), (np.log(rho1*3), None), (None, None), (np.log(rho1*5.), None), (None, None))
soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", args=(gp,), bounds=bnds)
opt_gp = set_params(soln.x, gp)
print(soln)


plt.figure()
plt.title("maximum likelihood psd")
plot_psd(opt_gp)
plt.show()
#plt.savefig('maximum_likelihood.pdf')

plt.figure()
plt.title("maximum likelihood prediction")
plot_prediction(opt_gp)
plt.show()
#plt.savefig('prediction.pdf')

prior_sigma = 2.0




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




