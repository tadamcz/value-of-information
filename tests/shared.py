import contextlib
from typing import List

import bayes_continuous.utils
import numpy as np
from bayes_continuous.likelihood_func import NormalLikelihood
from scipy import stats

from tests.seeds import RANDOM_SEEDS_1000
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
	posterior = stats.norm(posterior_mu, posterior_sigma)

	# For speed and accuracy, we can replace the method `expect` (numerical integration)
	# with `mean` (closed form).
	# Ideally, one would do this with patching/mocking instead of here.
	posterior.expect = posterior.mean

	return posterior


def simulation_input_idfn(inputs: SimulationInputs):
	pri_loc, pri_scale = get_location_scale(inputs.prior_T)
	return f"bar={inputs.bar}, E[T]~={round_sig(inputs.prior_ev)}, T_loc={pri_loc}, T_scale={pri_scale}, sd(B)~={inputs.sd_B}"


def rel_idfn(p):
	return f"rel={p}"


def iter_idfn(p):
	return f"iter={p}"


def const_inputs(linsp_n) -> List[SimulationInputs]:
	inputs = []
	for sd_B in np.linspace(.5, 5, num=linsp_n):
		prior = stats.norm(1.23, 5)
		i = SimulationInputs(
			prior=prior,
			sd_B=sd_B,
			bar=5
		)
		inputs.append(i)

	for prior_stdev in np.linspace(3, 10, num=linsp_n):
		prior = stats.norm(1.23, prior_stdev)
		i = SimulationInputs(
			prior=prior,
			sd_B=3,
			bar=5
		)
		inputs.append(i)

	for distance_to_bar in np.linspace(-5, 5, num=linsp_n):
		# These distances are quite pitiful, but we can be much more aggressive
		# in the `extra_slow` tests.
		bar = 1
		prior_mean = bar + distance_to_bar

		prior = stats.norm(prior_mean, 5)
		i = SimulationInputs(
			prior=prior,
			sd_B=3,
			bar=bar
		)
		inputs.append(i)

	return inputs


@contextlib.contextmanager
def temp_seed(seed):
	state = np.random.get_state()
	np.random.seed(seed)
	try:
		yield
	finally:
		np.random.set_state(state)


def inputs_from_seed(n) -> List[SimulationInputs]:
	inputs = []
	if n > len(RANDOM_SEEDS_1000):
		raise ValueError
	for i in range(n):
		with temp_seed(RANDOM_SEEDS_1000[i]):
			prior_mean, prior_sd = np.random.randint(-100, 100), np.random.randint(1, 50)
			pop_sd = np.random.randint(1, 50)
			distance_to_bar = np.random.randint(-20, 20)
		bar = prior_mean + distance_to_bar

		kwargs = {
			'prior': stats.norm(prior_mean, prior_sd),
			'study_sample_size': 100,
			'population_std_dev': pop_sd,
			'bar': bar,
		}
		inputs.append(SimulationInputs(**kwargs))
	return inputs