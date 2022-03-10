from unittest.mock import patch

from scipy import stats

import tests.shared as shared
from simulation import Simulation


class TestExtreme:
	def extreme_bar(self, bar):
		prior = stats.norm(1, 1)
		study_sample_size = 100
		population_std_dev = 20

		with patch('simulation.Posterior') as patched_posterior:
			patched_posterior.side_effect = shared.normal_normal_closed_form
			simulation = Simulation(
				prior=prior,
				study_sample_size=study_sample_size,
				population_std_dev=population_std_dev,
				bar=bar)
			assert simulation.run(max_runs=500) == 0

	def test_extreme_high_bar(self):
		"""
		If both prior expected value and all values of posterior are less than the bar,
		the study value is 0.
		"""
		self.extreme_bar(1e9)

	def test_extreme_low_bar(self):
		"""
		If both prior expected value and all values of posterior are greater than the bar,
		the study value is 0.
		"""
		self.extreme_bar(-1e9)
