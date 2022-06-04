import contextlib
from typing import List

import numpy as np
from scipy import stats

from tests.seeds import RANDOM_SEEDS
from value_of_information.simulation import SimulationParameters

PRIOR_MEAN = 1.23
PRIOR_SD = 10  # A large value makes the signal have >0 benefit more often, so convergence is easier
SD_B = 3
BAR = 5


def linsp(n) -> List[SimulationParameters]:
	"""
	Linearly spaced parameters.
	"""
	if n < 6:
		raise ValueError
	inputs = linsp_sd_B(n // 3) + linsp_prior_sd(n // 3) + linsp_distance_to_bar(n // 3)
	return inputs


def linsp_distance_to_bar(n) -> List[SimulationParameters]:
	inputs = []
	for distance_to_bar in np.linspace(-PRIOR_SD, PRIOR_SD, num=n):
		bar = 1
		prior_mean = bar + distance_to_bar

		prior = stats.norm(prior_mean, PRIOR_SD)
		i = SimulationParameters(
			prior=prior,
			sd_B=SD_B,
			bar=bar
		)
		inputs.append(i)
	return inputs


def linsp_prior_sd(n) -> List[SimulationParameters]:
	inputs = []
	for prior_stdev in np.linspace(PRIOR_SD, 2 * PRIOR_SD, num=n):
		prior = stats.norm(PRIOR_MEAN, prior_stdev)
		i = SimulationParameters(
			prior=prior,
			sd_B=SD_B,
			bar=BAR
		)
		inputs.append(i)
	return inputs


def linsp_sd_B(n):
	inputs = []
	for sd_B in np.linspace(3, 10, num=n):
		prior = stats.norm(PRIOR_MEAN, PRIOR_SD)
		i = SimulationParameters(
			prior=prior,
			sd_B=sd_B,
			bar=BAR
		)
		inputs.append(i)
	return inputs


def from_seed(n) -> List[SimulationParameters]:
	"""
	Pseudo-randomly generated parameters, with fixed seeds for reproducibility.
	"""
	inputs = []
	if n > len(RANDOM_SEEDS):
		raise ValueError
	for i in range(n):
		with temp_seed(RANDOM_SEEDS[i]):
			prior_mean = -1  # Makes little difference

			# A large value makes the the signal have >0 benefit more often, so convergence is easier
			prior_sd = np.random.randint(3, 20)

			sd_B = np.random.randint(1, 10)
			distance_to_bar = np.random.randint(-2.5 * prior_sd, 2.5 * prior_sd)
		bar = prior_mean + distance_to_bar

		kwargs = {
			'prior': stats.norm(prior_mean, prior_sd),
			'sd_B': sd_B,
			'bar': bar,
		}
		inputs.append(SimulationParameters(**kwargs))
	return inputs


@contextlib.contextmanager
def temp_seed(seed):
	state = np.random.get_state()
	np.random.seed(seed)
	try:
		yield
	finally:
		np.random.set_state(state)
