"""
Tests based on the direct simplified expression for `VOI(t) = E_B[VOI(T,B) | T=t]` (see README and voi.py)
"""
import pytest
from scipy import integrate

import tests.param_generators.lognorm_norm as gen_lognorm_norm
import tests.param_generators.norm_norm as gen_norm_norm
from tests import shared
from value_of_information.simulation import SimulationExecutor
from value_of_information.voi import threshold_b, expected_voi


@pytest.mark.parametrize('simulation_inputs',
						 argvalues=gen_lognorm_norm.linsp(8) + gen_norm_norm.linsp(12),
						 ids=shared.simulation_input_idfn)
def test_integral(simulation_inputs):
	"""
	Check that integrating over the expression returns the same as the simulation mean.
	"""
	b_threshold = threshold_b(simulation_inputs.prior_T, simulation_inputs.sd_B, simulation_inputs.bar)

	voi = lambda t: expected_voi(t,
								 b_threshold=b_threshold,
								 sd_B=simulation_inputs.sd_B,
								 bar=simulation_inputs.bar,
								 prior_ev=simulation_inputs.prior_ev)

	f_to_integrate = lambda t: simulation_inputs.prior_T.pdf(t) * voi(t)
	left, right = simulation_inputs.prior_T.support()

	voi_integral = integrate.quad(f_to_integrate, a=left, b=right)[0]

	voi_simulation = SimulationExecutor(simulation_inputs).execute(iterations=500_000).mean_voi()

	assert voi_simulation == pytest.approx(voi_integral, rel=5 / 100)


@pytest.mark.parametrize('simulation_inputs',
						 argvalues=gen_lognorm_norm.linsp(8) + gen_norm_norm.linsp(12) + gen_norm_norm.from_seed(10),
						 ids=shared.simulation_input_idfn)
def test_every_iteration(simulation_inputs):
	"""
	Test checking that the expected VOI calculated for every T_i in SimulationExecutor, is the same as that given by the
	expression.
	"""
	executor = SimulationExecutor(simulation_inputs)
	executor.do_explicit_b_draw = False  # i.e., calculate VOI(T_i) for each T_i
	simulation_run = executor.execute(iterations=1000)

	b_threshold = threshold_b(simulation_inputs.prior_T, simulation_inputs.sd_B, simulation_inputs.bar)

	for row in simulation_run.iterations_data:
		voi_simulated = row['E_B[VOI]']

		voi_from_expression = expected_voi(
			t=row['T_i'],
			b_threshold=b_threshold,
			sd_B=simulation_inputs.sd_B,
			bar=simulation_inputs.bar,
			prior_ev=simulation_inputs.prior_ev,
		)

		assert voi_simulated == pytest.approx(voi_from_expression)