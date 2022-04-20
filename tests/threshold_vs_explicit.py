from unittest.mock import patch

import numpy as np
import pytest

import tests.param_generators.norm_norm as gen_norm_norm

import tests.shared as shared
from value_of_information.simulation import SimulationExecutor


class TestThresholdvsExplicit:
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

	@pytest.mark.parametrize('simulation_inputs', (9), ids=shared.simulation_input_idfn)
	def test_linsp(self, simulation_inputs):
		self.helper(inputs=simulation_inputs)

	@pytest.mark.extra_slow
	@pytest.mark.parametrize('simulation_inputs', gen_norm_norm.from_seed(10), ids=shared.simulation_input_idfn)
	@pytest.mark.parametrize('relative_tolerance', (1 / 10, 1 / 100, 1 / 1000), ids=shared.rel_idfn)
	@pytest.mark.parametrize('iterations', np.geomspace(5_000, 1_000_000, dtype=int, num=5), ids=shared.iter_idfn)
	def test_random(self, simulation_inputs, relative_tolerance, iterations):
		self.helper(inputs=simulation_inputs, relative_tolerance=relative_tolerance, iterations=iterations)

	@pytest.mark.extra_slow
	@pytest.mark.parametrize('simulation_inputs', gen_norm_norm.linsp(9), ids=shared.simulation_input_idfn)
	@pytest.mark.parametrize('relative_tolerance', (1 / 10, 1 / 100, 1 / 1000), ids=shared.rel_idfn)
	@pytest.mark.parametrize('iterations', np.geomspace(5_000, 1_000_000, dtype=int, num=5), ids=shared.iter_idfn)
	def test_linsp_extra(self, simulation_inputs, relative_tolerance, iterations):
		self.helper(inputs=simulation_inputs, relative_tolerance=relative_tolerance, iterations=iterations)