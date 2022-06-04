from unittest.mock import Mock, patch

import bayes_continuous.utils
import numpy as np
from bayes_continuous.likelihood_func import NormalLikelihood
from bayes_continuous.utils import is_frozen_normal
from scipy import stats

from value_of_information.rounding import round_sig
from value_of_information.simulation import SimulationInputs
from value_of_information.voi import solve_threshold_b


def get_location_scale(distribution: stats._distn_infrastructure.rv_frozen):
	args, kwds = distribution.args, distribution.kwds
	shapes, loc, scale = distribution.dist._parse_args(*args, **kwds)
	return loc, scale


def normal_normal_closed_form(normal_prior, normal_likelihood):
	if not bayes_continuous.utils.is_frozen_normal(normal_prior):
		raise ValueError
	if not isinstance(normal_likelihood, NormalLikelihood):
		raise ValueError

	mu_1, sigma_1 = get_location_scale(normal_prior)
	mu_2, sigma_2 = normal_likelihood.mu, normal_likelihood.sigma
	posterior_mu, posterior_sigma = bayes_continuous.utils.normal_normal_closed_form(mu_1, sigma_1, mu_2, sigma_2)

	# Using mocks to avoid the onerous creation of an `rv_frozen`
	posterior_expect = stats.norm.mean(posterior_mu, posterior_sigma)
	posterior_cdf = lambda x: stats.norm.cdf(x, posterior_mu, posterior_sigma)
	mock_distribution = Mock()
	mock_distribution.expect = Mock(return_value=posterior_expect)
	mock_distribution.cdf = Mock(side_effect=posterior_cdf)

	return mock_distribution


def sim_param_idfn(inputs: SimulationInputs):
	pri_loc, pri_scale = get_location_scale(inputs.prior_T)
	if inputs.prior_family() == "lognorm_gen":
		fam = "lognorm"
	elif inputs.prior_family() == "norm_gen":
		fam = "norm"
	else:
		fam = inputs.prior_family()

	return f"fam={fam}, bar={round_sig(inputs.bar)}, E[T]~={round_sig(inputs.prior_T_ev)}, T_loc={round_sig(pri_loc)}, T_scale={round_sig(pri_scale)}, sd(B)~={round_sig(inputs.sd_B)}"


def is_decreasing(array):
	diff = np.diff(array)
	return np.all(diff <= 0)


def is_increasing(array):
	diff = np.diff(array)
	return np.all(diff >= 0)


def expected_voi_t(t, threshold_b, sd_B, bar, prior_ev):
	"""
	Direct simplified expression. Currently, it's only used in tests, because for the simulation
	we want to be able to store and display the building blocks of this expression.

	VOI(t) = E_B[VOI(T,B) | T=t] = F(b_*) * (bar-t) + t - U(decision_0, t)
	"""
	if prior_ev > bar:
		payoff_no_signal = t
	else:
		payoff_no_signal = bar

	return stats.norm.cdf(threshold_b, loc=t, scale=sd_B) * (bar - t) + t - payoff_no_signal


def patched_threshold_b(prior_T, sd_B, bar):
	"""
	Use patch for performance
	"""
	if is_frozen_normal(prior_T):
		with patch('value_of_information.bayes.posterior') as patched_posterior:
			patched_posterior.side_effect = normal_normal_closed_form
			threshold_b = solve_threshold_b(prior_T, sd_B, bar)
	else:
		threshold_b = solve_threshold_b(prior_T, sd_B, bar)
	return threshold_b
