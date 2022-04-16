import contextlib
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

import tests.shared as shared
from tests.shared import simulation_input_idfn, rel_idfn, iter_idfn
from value_of_information.simulation import SimulationInputs, SimulationExecutor


def const_inputs(linsp_n):
	inputs = []
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


@contextlib.contextmanager
def temp_seed(seed):
	state = np.random.get_state()
	np.random.seed(seed)
	try:
		yield
	finally:
		np.random.set_state(state)

def random_inputs(n):
	inputs = []
	if n > len(shared.RANDOM_SEEDS_1000):
		raise ValueError
	for i in range(n):
		with temp_seed(shared.RANDOM_SEEDS_1000[i]):
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

	@pytest.mark.parametrize('simulation_inputs', const_inputs(linsp_n=3), ids=simulation_input_idfn)
	def test_linsp(self, simulation_inputs):
		self.helper(inputs=simulation_inputs)

	@pytest.mark.extra_slow
	@pytest.mark.parametrize('simulation_inputs', random_inputs(10), ids=simulation_input_idfn)
	@pytest.mark.parametrize('relative_tolerance', (1 / 10, 1 / 100, 1 / 1000), ids=rel_idfn)
	@pytest.mark.parametrize('iterations', np.geomspace(5_000, 1_000_000, dtype=int, num=5), ids=iter_idfn)
	def test_random(self, simulation_inputs, relative_tolerance, iterations):
		self.helper(inputs=simulation_inputs, relative_tolerance=relative_tolerance, iterations=iterations)

	@pytest.mark.extra_slow
	@pytest.mark.parametrize('simulation_inputs', const_inputs(linsp_n=3), ids=simulation_input_idfn)
	@pytest.mark.parametrize('relative_tolerance', (1 / 10, 1 / 100, 1 / 1000), ids=rel_idfn)
	@pytest.mark.parametrize('iterations', np.geomspace(5_000, 1_000_000, dtype=int, num=5), ids=iter_idfn)
	def test_linsp_extra(self, simulation_inputs, relative_tolerance, iterations):
		self.helper(inputs=simulation_inputs, relative_tolerance=relative_tolerance, iterations=iterations)