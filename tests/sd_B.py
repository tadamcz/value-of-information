import numpy as np
import pytest
from scipy import stats

from tests import shared
from tests.input_generators import NormNormGenerator, LogNormNormGenerator
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
			mean = SimulationExecutor(central_simulation_inputs).execute(iterations=iterations).mean_value_study()
			means.append(mean)
		assert self.is_decreasing(means)

	def test(self):
		central_simulation_inputs = SimulationInputs(
			prior=stats.lognorm(scale=np.exp(1), s=1),
			sd_B=5,
			bar=5
		)
		self.helper(central_simulation_inputs, iterations=2_000, num_sds=2)

	@pytest.mark.extra_slow
	@pytest.mark.parametrize('central_simulation_inputs', NormNormGenerator.from_seed(10)+LogNormNormGenerator.linsp_mu(5), ids=shared.simulation_input_idfn)
	def test_extra_slow(self, central_simulation_inputs):
		self.helper(central_simulation_inputs, iterations=50_000, num_sds=5)

	def is_decreasing(self, array):
		diff = np.diff(array)
		return np.all(diff <= 0)
