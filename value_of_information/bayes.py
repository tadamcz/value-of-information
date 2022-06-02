import numpy as np
from bayes_continuous.likelihood_func import LikelihoodFunction
from bayes_continuous.posterior import Posterior
from scipy.stats.distributions import rv_frozen


def posterior(prior: rv_frozen, likelihood: LikelihoodFunction):
	distribution = Posterior(prior, likelihood)
	if np.isnan(distribution.expect()):
		raise ValueError(f"Posterior expected value is NaN for {prior}, {likelihood}")
	return distribution
