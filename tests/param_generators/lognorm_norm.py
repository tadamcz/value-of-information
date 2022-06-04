from typing import List

import numpy as np
from scipy import stats

from value_of_information import utils
from value_of_information.simulation import SimulationParameters


def linsp(n) -> List[SimulationParameters]:
	"""
	Linearly spaced parameters.
	"""
	if n < 4:
		raise ValueError
	inputs = linsp_mu(n // 2) + linsp_distance_to_bar(n // 2)
	return inputs


def linsp_mu(n) -> List[SimulationParameters]:
	inputs = []

	for prior_mu in np.linspace(0.5, 2, num=n):
		prior_sigma = 1
		prior = utils.lognormal(prior_mu, prior_sigma)
		prior_expect, prior_sd = utils.get_lognormal_moments(prior_mu, prior_sigma)

		i = SimulationParameters(
			prior=prior,
			sd_B=10,
			bar=prior_expect + 1 * prior_sd
		)
		inputs.append(i)
	return inputs


def linsp_distance_to_bar(n) -> List[SimulationParameters]:
	inputs = []
	prior_mu = 1
	prior_sigma = 1
	prior = utils.lognormal(prior_mu, prior_sigma)
	prior_expect, prior_sd = utils.get_lognormal_moments(prior_mu, prior_sigma)
	prior_median = np.exp(prior_mu)

	for bar in np.linspace(prior_expect, prior_expect + 3 * prior_sd, num=n):
		i = SimulationParameters(
			prior=prior,
			sd_B=10,
			bar=bar
		)
		inputs.append(i)
	return inputs
