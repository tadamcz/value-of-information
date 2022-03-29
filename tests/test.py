from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

import tests.shared as shared
from simulation import Simulation


class TestInfiniteBar:
	def simulate(self, bar):
		prior = stats.norm(1, 1)
		study_sample_size = 100
		population_std_dev = 20

		with patch('simulation.Simulation.posterior') as patched_posterior:
			patched_posterior.side_effect = shared.normal_normal_closed_form
			simulation = Simulation(
				prior=prior,
				study_sample_size=study_sample_size,
				population_std_dev=population_std_dev,
				bar=bar)
			assert simulation.run(iterations=500).mean_value_study() == 0

	def test_high(self):
		"""
		If both prior expected value and all values of posterior are less than the bar,
		the study value is 0.
		"""
		self.simulate(bar=1e9)

	def test_low(self):
		"""
		If both prior expected value and all values of posterior are greater than the bar,
		the study value is 0.
		"""
		self.simulate(bar=-1e9)


class TestInfiniteSample:
	prior_mean = 1.23456
	prior_sd = 1

	def simulate(self, iterations):
		prior = stats.norm(self.prior_mean, self.prior_sd)
		study_sample_size = 1e12
		population_std_dev = 1/10_000
		bar = 0  # Has no effect

		with patch('simulation.Simulation.posterior') as patched_posterior:
			patched_posterior.side_effect = shared.normal_normal_closed_form
			simulation = Simulation(
				prior=prior,
				study_sample_size=study_sample_size,
				population_std_dev=population_std_dev,
				bar=bar)
			return simulation.run(iterations=iterations)

	def test_each_iteration(self):
		"""
		If the sample size is essentially infinite, the signal we receive is infinitely precise.
		So the posterior mean is equal to T_i at each iteration.
		"""
		simulation_run = self.simulate(100)
		T_is = np.asarray(simulation_run.iterations_data['T_i'])
		expected_values = np.asarray(simulation_run.iterations_data['posterior_ev'])
		assert T_is == pytest.approx(expected_values, rel=1e-5)

	def test_mean(self):
		"""
		If the posterior mean is equal to T_i at each iteration,
		the mean of posterior means is equal to the prior mean.
		"""
		simulation_run = self.simulate(5_000)
		# A generous tolerance is necessary so the tests finish in a reasonable time
		absolute_tolerance = self.prior_sd / 10
		assert simulation_run.iterations_data['posterior_ev'].mean() == pytest.approx(self.prior_mean, abs=absolute_tolerance)

