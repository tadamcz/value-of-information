from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from tests import shared
import tests.param_generators.norm_norm as gen_norm_norm
import tests.param_generators.lognorm_norm as gen_log_norm_norm
from tests.shared import get_location_scale, is_decreasing, is_increasing

from value_of_information.simulation import SimulationExecutor, SimulationInputs


class Test_sdB:
	"""
	The value of the study is decreasing in sd(B).
	"""

	def helper(self, central_simulation_inputs, iterations, num_sds):
		central_sd_B = central_simulation_inputs.sd_B
		list_sd_Bs = np.linspace(central_sd_B * 0.5, central_sd_B * 1.5, num=num_sds)
		means = []
		for sd_B in list_sd_Bs:
			central_simulation_inputs.sd_B = sd_B
			mean = SimulationExecutor(central_simulation_inputs, print_every=1e9).execute(iterations=iterations).mean_value_study()
			means.append(mean)
		assert is_decreasing(means)

	def test(self):
		central_simulation_inputs = SimulationInputs(
			prior=stats.norm(1,1),
			sd_B=1,
			bar=2
		)
		self.helper(central_simulation_inputs, iterations=2_000, num_sds=2)

	@pytest.mark.extra_slow
	@pytest.mark.parametrize('central_simulation_inputs', gen_log_norm_norm.linsp_mu(3), ids=shared.simulation_input_idfn)
	def test_extra_slow_lognorm_prior(self, central_simulation_inputs):
		self.helper(central_simulation_inputs, iterations=100_000, num_sds=3)

	@pytest.mark.extra_slow
	@pytest.mark.parametrize('central_simulation_inputs', gen_norm_norm.linsp(6) + gen_norm_norm.from_seed(10), ids=shared.simulation_input_idfn)
	def test_extra_slow_normal_prior(self, central_simulation_inputs):
		with patch('value_of_information.simulation.SimulationExecutor.posterior') as patched_posterior:
			patched_posterior.side_effect = shared.normal_normal_closed_form
			self.helper(central_simulation_inputs, iterations=100_000, num_sds=3)


class Test_sd_prior_T:
	"""
	The value of the study is increasing in sd(prior_T) for a normal prior.
	"""

	def helper(self, central_simulation_inputs, iterations, num_sds):
		with patch('value_of_information.simulation.SimulationExecutor.posterior') as patched_posterior:
			patched_posterior.side_effect = shared.normal_normal_closed_form
			central_mean, central_sd = get_location_scale(central_simulation_inputs.prior_T)
			list_sd_Ts = np.linspace(central_sd * 0.75, central_sd * 1.25, num=num_sds)
			means = []
			for sd_T in list_sd_Ts:
				central_simulation_inputs.prior_T = stats.norm(central_mean, sd_T)
				mean = SimulationExecutor(central_simulation_inputs, print_every=1e9).execute(iterations=iterations).mean_value_study()
				means.append(mean)
			assert is_increasing(means)

	def test(self):
		central_simulation_inputs = SimulationInputs(
			prior=stats.norm(1, 1),
			sd_B=1,
			bar=2
		)
		self.helper(central_simulation_inputs, iterations=2_000, num_sds=2)

	@pytest.mark.extra_slow
	@pytest.mark.parametrize('central_simulation_inputs', gen_norm_norm.linsp(6)+gen_norm_norm.from_seed(10), ids=shared.simulation_input_idfn)
	def test_extra_slow(self, central_simulation_inputs):
		self.helper(central_simulation_inputs, iterations=100_000, num_sds=3)
