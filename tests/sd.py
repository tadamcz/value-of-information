from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

import tests.param_generators.lognorm_norm as gen_log_norm_norm
import tests.param_generators.norm_norm as gen_norm_norm
from tests import shared
from tests.shared import get_location_scale, is_decreasing, is_increasing
from value_of_information.simulation import SimulationExecutor


class Test_sdB:
	"""
	The value of the signal is decreasing in sd(B).
	"""

	def helper(self, central_simulation_inputs, iterations, num_sds):
		central_sd_B = central_simulation_inputs.sd_B
		list_sd_Bs = np.linspace(central_sd_B * 0.5, central_sd_B * 1.5, num=num_sds)
		means = []
		for sd_B in list_sd_Bs:
			central_simulation_inputs.sd_B = sd_B
			mean = SimulationExecutor(central_simulation_inputs).execute(
				iterations=iterations).mean_voi()
			means.append(mean)
		assert is_decreasing(means)

	@pytest.mark.parametrize('central_simulation_inputs', gen_log_norm_norm.linsp(4), ids=shared.simulation_input_idfn)
	def test_lognorm(self, central_simulation_inputs):
		self.helper(central_simulation_inputs, iterations=150_000, num_sds=3)

	@pytest.mark.parametrize('central_simulation_inputs', gen_norm_norm.linsp(6) + gen_norm_norm.from_seed(3),
							 ids=shared.simulation_input_idfn)
	def test_norm(self, central_simulation_inputs):
		with patch('value_of_information.bayes.posterior') as patched_posterior:
			patched_posterior.side_effect = shared.normal_normal_closed_form
			self.helper(central_simulation_inputs, iterations=150_000, num_sds=3)


class Test_sd_prior_T:
	"""
	The value of the signal is increasing in sd(prior_T) for a normal prior.
	"""

	def helper(self, central_simulation_inputs, iterations, num_sds):
		with patch('value_of_information.bayes.posterior') as patched_posterior:
			patched_posterior.side_effect = shared.normal_normal_closed_form
			central_mean, central_sd = get_location_scale(central_simulation_inputs.prior_T)
			list_sd_Ts = np.linspace(central_sd * 0.75, central_sd * 1.25, num=num_sds)
			means = []
			for sd_T in list_sd_Ts:
				central_simulation_inputs.prior_T = stats.norm(central_mean, sd_T)
				mean = SimulationExecutor(central_simulation_inputs).execute(
					iterations=iterations).mean_voi()
				means.append(mean)
			assert is_increasing(means)

	@pytest.mark.parametrize('central_simulation_inputs', gen_norm_norm.linsp(6) + gen_norm_norm.from_seed(3),
							 ids=shared.simulation_input_idfn)
	def test(self, central_simulation_inputs):
		self.helper(central_simulation_inputs, iterations=150_000, num_sds=3)
