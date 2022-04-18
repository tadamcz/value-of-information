import contextlib
from typing import List

import numpy as np
from scipy import stats

from tests.seeds import RANDOM_SEEDS_1000
from value_of_information.simulation import SimulationInputs


class NormNormGenerator:
	PRIOR_MEAN = 1.23
	PRIOR_SD = 5
	SD_B = 3
	BAR = 5
	@staticmethod
	def from_seed(n) -> List[SimulationInputs]:
		"""
		Pseudo-randomly generated parameters, with fixed seeds for reproducibility.
		"""
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

	@staticmethod
	def linsp(n) -> List[SimulationInputs]:
		"""
		Linearly spaced parameters.
		"""
		if n < 6:
			raise ValueError
		inputs = NormNormGenerator.linsp_sd_B(n // 3) + NormNormGenerator.linsp_prior_sd(n // 3) + NormNormGenerator.linsp_distance_to_bar(n // 3)
		return inputs

	@staticmethod
	def linsp_sd_B(n):
		inputs = []
		for sd_B in np.linspace(.5, 5, num=n):
			prior = stats.norm(NormNormGenerator.PRIOR_MEAN, NormNormGenerator.PRIOR_SD)
			i = SimulationInputs(
				prior=prior,
				sd_B=sd_B,
				bar=NormNormGenerator.BAR
			)
			inputs.append(i)
		return inputs

	@staticmethod
	def linsp_prior_sd(n):
		inputs = []
		for prior_stdev in np.linspace(3, 10, num=n):
			prior = stats.norm(NormNormGenerator.PRIOR_MEAN, prior_stdev)
			i = SimulationInputs(
				prior=prior,
				sd_B=NormNormGenerator.SD_B,
				bar=NormNormGenerator.BAR
			)
			inputs.append(i)
		return inputs

	@staticmethod
	def linsp_distance_to_bar(n):
		inputs = []
		for distance_to_bar in np.linspace(-5, 5, num=n):
			# These distances are quite pitiful, but we can be much more aggressive
			# in the `extra_slow` tests.
			bar = 1
			prior_mean = bar + distance_to_bar

			prior = stats.norm(prior_mean, NormNormGenerator.PRIOR_SD)
			i = SimulationInputs(
				prior=prior,
				sd_B=NormNormGenerator.SD_B,
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
