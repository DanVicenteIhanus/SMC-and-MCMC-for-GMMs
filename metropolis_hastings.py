import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
PATH = "/Users/danvicente/skola/DD2420 - PGM/labs/Tutorial 7/"

def gaussian_mixture_model(a_1, mu_1, var_1, a_2, mu_2, var_2, x):
    # Calculate the weighted PDF of both Gaussian components
    return (a_1 * norm.pdf(x, loc=mu_1, scale=np.sqrt(var_1)) +
            a_2 * norm.pdf(x, loc=mu_2, scale=np.sqrt(var_2)))

def metropolis_hastings(n_samples: int, var: float,
                        a_1: float, mu_1: float, var_1: float, a_2: float,
                        mu_2: float, var_2: float) -> np.ndarray:
    x = np.zeros(n_samples + 1)
    x[0] = np.random.uniform(-5, 5)  
    for i in tqdm(range(1, n_samples + 1)):
        u = np.random.uniform(0, 1)
        x_gen = x[i - 1] + np.random.normal(loc=0, scale=np.sqrt(var))
        p_x_gen = gaussian_mixture_model(a_1, mu_1, var_1, a_2, mu_2, var_2, x_gen)
        p_x_i = gaussian_mixture_model(a_1, mu_1, var_1, a_2, mu_2, var_2, x[i - 1])
        if u < min(1, p_x_gen / p_x_i):
            x[i] = x_gen
        else:
            x[i] = x[i - 1]
    return x[1:]  # Skip the initial random point

# GMM parameters
a_1   = 0.5
a_2   = 0.5
mu_1  = 0.0
var_1 = 1.0
mu_2  = 3.0
var_2 = 0.5**2

# MH parameters
n_samples = 10000
var       = 0.1

samples = metropolis_hastings(n_samples, var, a_1, mu_1, var_1, a_2, mu_2, var_2)
x = np.linspace(-6, 7, 1000)
distribution = gaussian_mixture_model(a_1, mu_1, var_1, a_2, mu_2, var_2, x)

fig,ax = plt.subplots(figsize=(10, 6))
ax.plot(x, distribution, lw=1.7, color="#274E13", label="Gaussian Mixture Model")
ax.hist(samples, bins=100, density=True, alpha=0.5, lw=1, ec ="k", label="Metropolis-Hastings Samples")
ax.legend()
ax.set_title("Metropolis-Hastings for Gaussian Mixture Model")
ax.set_xlabel("$x$")
ax.set_ylabel("$p(x)$")
ax.set_xlim([-4,7])
fig.savefig(f"{PATH}/plots/metropolis_n_samples_=_{n_samples}_var_=_{var}.pdf", bbox_inches="tight")
plt.show()

fig_2, ax_2 = plt.subplots(figsize=(10,6))
iter_vector = np.arange(1, n_samples+1, 1)
ax_2.scatter(iter_vector, samples, color='k', s=0.5, cmap='viridis')
ax_2.set_xlabel("Iteration $i$")
ax_2.set_ylabel("$x[i]$")
ax_2.set_title("Random walk with Metropolis Hastings")
fig_2.savefig(f"{PATH}/plots/random_walk_n_samples_=_{n_samples}_var_=_{var}.pdf", bbox_inches="tight")