import random
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

import tests.shared as shared
from value_of_information.rounding import round_sig
from value_of_information.simulation import SimulationInputs, SimulationExecutor


def const_inputs():
	inputs = []
	linsp_n = 3
	for pop_stdev in np.linspace(.5, 5, num=linsp_n):
		prior = stats.norm(1.23, 5)
		i = SimulationInputs(
			prior=prior,
			study_sample_size=1,
			population_std_dev=pop_stdev,
			bar=5
		)
		inputs.append(i)

	for prior_stdev in np.linspace(3, 10, num=linsp_n):
		prior = stats.norm(1.23, prior_stdev)
		i = SimulationInputs(
			prior=prior,
			study_sample_size=1,
			population_std_dev=3,
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
			study_sample_size=1,
			population_std_dev=3,
			bar=bar
		)
		inputs.append(i)

	return inputs


def simulation_input_id_func(inputs: SimulationInputs):
	pri_loc, pri_scale = shared.get_location_scale(inputs.prior_T)
	return f"bar={inputs.bar}, E[T]~={round_sig(inputs.prior_ev)}, T_loc={pri_loc}, T_scale={pri_scale}, sd(B)~={inputs.sd_B}"


def random_inputs():
	inputs = []
	for _ in range(100):
		prior_mean, prior_sd = random.randint(-100, 100), random.randint(1, 50)
		pop_sd = random.randint(1, 50)
		distance_to_bar = random.randint(-100, 100)
		bar = prior_mean + distance_to_bar

		kwargs = {
			'prior': stats.norm(prior_mean, prior_sd),
			'study_sample_size': 100,
			'population_std_dev': pop_sd,
			'bar': bar,
		}
		inputs.append(SimulationInputs(**kwargs))
	return inputs


class TestThresholdUpdate:
	"""
	Additional ideas:
	1.
	using mocking, pass in the same explicit arrays of T_is and b_is to both methods,
	then check that the value of the study is the same in each row.
	"""

	def helper(self, inputs, iterations=12_000, relative_tolerance=None):
		if relative_tolerance is None:
			# A generous tolerance is necessary so the tests finish in a reasonable time
			# todo: try to find ways to improve performance further
			relative_tolerance = 10 / 100
		with patch('value_of_information.simulation.SimulationExecutor.posterior') as patched_posterior:
			patched_posterior.side_effect = shared.normal_normal_closed_form
			explicit = SimulationExecutor(inputs, force_explicit=True).execute(iterations=iterations)
			threshold = SimulationExecutor(inputs, force_explicit=False).execute(iterations=iterations)

			assert explicit.mean_value_study() == pytest.approx(
				threshold.mean_value_study(), rel=relative_tolerance)

	@pytest.mark.parametrize('simulation_inputs', const_inputs(), ids=simulation_input_id_func)
	def test(self, simulation_inputs):
		self.helper(inputs=simulation_inputs)

	# extra_slow below

	def iterations_helper(self, inputs, relative_tolerance, max_iterations):
		"""
		todo consider refactoring this into `shared.py`
		"""
		last_assertion_error = None
		min_iterations = 8_000
		iterations = min_iterations
		while iterations < max_iterations:
			try:
				self.helper(inputs=inputs, iterations=iterations, relative_tolerance=relative_tolerance)
			except AssertionError as err:
				last_assertion_error = err
				iterations *= 2
			else:
				return
		raise last_assertion_error

	@pytest.mark.parametrize('simulation_inputs', random_inputs(), ids=simulation_input_id_func)
	@pytest.mark.extra_slow
	def test_random(self, simulation_inputs):
		self.iterations_helper(inputs=simulation_inputs, relative_tolerance=1e-4, max_iterations=1e5)

	@pytest.mark.extra_slow
	def test_strict(self):
		inputs = const_inputs()[0]
		with patch('value_of_information.simulation.SimulationRun.print_intermediate') as patched_print:
			patched_print.side_effect = lambda: None
			self.iterations_helper(inputs=inputs, relative_tolerance=1e-4, max_iterations=10e6)
