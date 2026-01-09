import numpy as np
import jax.numpy as jnp
import jax
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Helper: Design matrix ensuring concavity via positive cumulative slopes
# ----------------------------------------------------------------------
def build_X(K: int):
    """Design matrix with X[k,j] = min(k+1, j+1)."""
    X = np.zeros((K, K))
    for k in range(K):
        for j in range(K):
            X[k, j] = min(k + 1, j + 1)
    return jnp.array(X)

# ----------------------------------------------------------------------
# Model: Latent concave logit + Bernoulli likelihood
# ----------------------------------------------------------------------
def concave_bernoulli_model(k_obs, y_obs, K):
    """
    k_obs: indices (0..K-1)
    y_obs: Bernoulli(0/1) observations
    K: total number of cardinalities
    """
    X = build_X(K)
    beta0 = numpyro.sample("beta0", dist.Normal(0., 2.))
    eta = numpyro.sample("eta", dist.Normal(0., 1.).expand([K]))  # unconstrained slopes
    w = jnp.exp(eta)  # ensures positive slopes -> concave latent f
    f_all = beta0 + jnp.dot(X, w)  # latent concave logits
    p_all = jax.nn.sigmoid(f_all)  # map to Bernoulli probabilities
    numpyro.sample("y", dist.Bernoulli(p_all[k_obs]), obs=y_obs)

# ----------------------------------------------------------------------
# Simulate data from a concave Bernoulli process
# ----------------------------------------------------------------------
def simulate_data(K=10, n_obs=200, seed=0):
    rng = np.random.default_rng(seed)
    # True concave mean function (probabilities)
    f_true = 2.5 * np.sqrt(np.arange(K) + 1) - 5.0
    p_true = 1 / (1 + np.exp(-f_true))
    # Draw observations
    k_obs = rng.integers(0, K, size=n_obs)
    y_obs = rng.binomial(1, p_true[k_obs])
    return k_obs, y_obs, p_true

# ----------------------------------------------------------------------
# Run MCMC on Mac M1 CPU
# ----------------------------------------------------------------------
def run_inference():
    K = 10
    k_obs, y_obs, p_true = simulate_data(K, n_obs=250)
    rng_key = random.PRNGKey(42)

    nuts_kernel = NUTS(concave_bernoulli_model)
    mcmc = MCMC(nuts_kernel, num_warmup=800, num_samples=1200, num_chains=1)
    mcmc.run(rng_key, k_obs=k_obs, y_obs=y_obs, K=K)
    samples = mcmc.get_samples()

    # Posterior predictive probabilities
    eta = np.exp(np.array(samples["eta"]))
    beta0 = np.array(samples["beta0"])
    X = np.array(build_X(K))
    f_draws = beta0[:, None] + eta.dot(X.T)
    p_draws = 1 / (1 + np.exp(-f_draws))
    p_mean = p_draws.mean(axis=0)
    p_lower, p_upper = np.quantile(p_draws, [0.05, 0.95], axis=0)

    # Plot results
    ks = np.arange(K)
    plt.figure(figsize=(7, 4))
    plt.plot(ks, p_true, "k--", label="True prob (concave)")
    plt.plot(ks, p_mean, "b", label="Posterior mean")
    plt.fill_between(ks, p_lower, p_upper, color="blue", alpha=0.2, label="90% CI")
    plt.xlabel("Number of arms (k)")
    plt.ylabel("Reward probability")
    plt.title("Bayesian Concave Regression (Bernoulli, works on Mac M1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_inference()
