from unittest.mock import Mock

import bayes_continuous.utils
import numpy as np
from bayes_continuous.likelihood_func import NormalLikelihood
from scipy import stats

from value_of_information.rounding import round_sig
from value_of_information.simulation import SimulationInputs


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


def simulation_input_idfn(inputs: SimulationInputs):
	pri_loc, pri_scale = get_location_scale(inputs.prior_T)
	return f"fam={inputs.prior_family()}, bar={inputs.bar}, E[T]~={round_sig(inputs.prior_ev)}, T_loc={round_sig(pri_loc)}, T_scale={round_sig(pri_scale)}, sd(B)~={round_sig(inputs.sd_B)}"


def rel_idfn(p):
	return f"rel={p}"


def iter_idfn(p):
	return f"iter={p}"


def is_decreasing(array):
	diff = np.diff(array)
	return np.all(diff <= 0)


def is_increasing(array):
	diff = np.diff(array)
	return np.all(diff >= 0)
