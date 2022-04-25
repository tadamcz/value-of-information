from typing import List

import numpy as np
from scipy import stats

from value_of_information.simulation import SimulationInputs


def linsp(n) -> List[SimulationInputs]:
	"""
	Linearly spaced parameters.
	"""
	if n < 4:
		raise ValueError
	inputs = linsp_mu(n // 2) + linsp_distance_to_bar(n // 2)
	return inputs


def linsp_mu(n) -> List[SimulationInputs]:
	inputs = []

	for prior_mu in np.linspace(0.5, 2, num=n):
		prior_sigma = 1
		prior = stats.lognorm(scale=np.exp(prior_mu), s=prior_sigma)

		i = SimulationInputs(
			prior=prior,
			sd_B=10,
			bar=5
		)
		inputs.append(i)
	return inputs


def linsp_distance_to_bar(n) -> List[SimulationInputs]:
	inputs = []
	prior_mu = 1
	prior_sigma = 1
	prior = stats.lognorm(scale=np.exp(prior_mu), s=prior_sigma)
	prior_var = (np.exp(prior_sigma ** 2) - 1) * np.exp(2 * prior_mu + prior_sigma ** 2)
	prior_sd = np.sqrt(prior_var)
	prior_median = np.exp(prior_mu)

	for bar in np.linspace(prior_median, 7 * prior_sd, num=n):
		i = SimulationInputs(
			prior=prior,
			sd_B=10,
			bar=bar
		)
		inputs.append(i)
	return inputs
