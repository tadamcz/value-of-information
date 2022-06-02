from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

import tests.shared as shared
from value_of_information.simulation import SimulationInputs, SimulationExecutor


class TestInfiniteBar:
	def simulate(self, bar):
		prior = stats.norm(1, 1)
		study_sample_size = 100
		population_std_dev = 20

		with patch('value_of_information.bayes.posterior') as patched_posterior:
			patched_posterior.side_effect = shared.normal_normal_closed_form
			inputs = SimulationInputs(
				prior=prior,
				study_sample_size=study_sample_size,
				population_std_dev=population_std_dev,
				bar=bar)
			assert SimulationExecutor(inputs, print_every=1e9).execute(iterations=10_000).mean_voi() == 0

	def test_high(self):
		"""
		If both prior expected value and all values of posterior are less than the bar,
		the signal value is 0.
		"""
		self.simulate(bar=1e9)

	def test_low(self):
		"""
		If both prior expected value and all values of posterior are greater than the bar,
		the signal value is 0.
		"""
		self.simulate(bar=-1e9)


class TestInfiniteSample:
	prior_mean = 1.23456
	prior_sd = 1

	def simulate(self, iterations):
		prior = stats.norm(self.prior_mean, self.prior_sd)
		bar = 0  # Has no effect

		with patch('value_of_information.bayes.posterior') as patched_posterior:
			patched_posterior.side_effect = shared.normal_normal_closed_form
			inputs = SimulationInputs(
				prior=prior,
				sd_B=1e-12,
				bar=bar)
			return SimulationExecutor(inputs, force_explicit_bayes=True, print_every=1e9).execute(iterations=iterations)

	def test_each_iteration(self):
		"""
		If the sample size is essentially infinite, the signal we receive is infinitely precise.
		So the posterior mean is equal to T_i at each iteration_explicit_b.
		"""
		simulation_run = self.simulate(100)
		T_is = np.asarray(simulation_run.get_column('T_i'))
		expected_values = np.asarray(simulation_run.get_column('E[T|b_i]'))
		assert T_is == pytest.approx(expected_values, rel=1 / 100)

	def mean_helper(self, relative_tolerance, iterations):
		"""
		If the posterior mean is equal to T_i at each iteration_explicit_b,
		the mean of posterior means is equal to the prior mean.
		"""
		simulation_run = self.simulate(iterations)
		print(simulation_run.get_column('E[T|b_i]').mean(), self.prior_mean)
		assert simulation_run.get_column('E[T|b_i]').mean() == pytest.approx(self.prior_mean, rel=relative_tolerance)

	def test_mean(self):
		"""
		A generous tolerance is necessary so the tests finish in a reasonable time
		"""
		self.mean_helper(iterations=5_000, relative_tolerance=5 / 100)

	def test_mean_strict(self):
		self.mean_helper(relative_tolerance=1 / 100, iterations=50_000)
