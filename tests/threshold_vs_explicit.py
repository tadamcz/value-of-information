from unittest.mock import patch

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

	def helper(self, inputs, iterations, relative_tolerance):
		with patch('value_of_information.simulation.SimulationExecutor.posterior') as patched_posterior:
			patched_posterior.side_effect = shared.normal_normal_closed_form
			explicit = SimulationExecutor(inputs, force_explicit=True).execute(iterations=iterations)
			threshold = SimulationExecutor(inputs, force_explicit=False).execute(iterations=iterations)

			assert explicit.mean_value_study() == pytest.approx(
				threshold.mean_value_study(), rel=relative_tolerance)

	@pytest.mark.parametrize('simulation_inputs',
							gen_norm_norm.linsp_distance_to_bar(2), ids=shared.simulation_input_idfn)
	def test(self, simulation_inputs):
		self.helper(inputs=simulation_inputs, iterations=12_000, relative_tolerance=15 / 100)

	@pytest.mark.extra_slow
	@pytest.mark.parametrize('simulation_inputs', gen_norm_norm.linsp(6), ids=shared.simulation_input_idfn)
	def test_linsp(self, simulation_inputs):
		self.helper(inputs=simulation_inputs, relative_tolerance=1 / 100, iterations=1_000_000)

	@pytest.mark.extra_slow
	@pytest.mark.parametrize('simulation_inputs', gen_norm_norm.from_seed(6), ids=shared.simulation_input_idfn)
	def test_seed(self, simulation_inputs):
		self.helper(inputs=simulation_inputs, relative_tolerance=1 / 100, iterations=1_000_000)
