"""
Tests based on the direct simplified expression for `VOI(t) = E_B[VOI(T,B) | T=t]` (see README and voi.py)
"""
import numpy as np
import pytest
from scipy import integrate

import tests.param_generators.lognorm_norm as gen_lognorm_norm
import tests.param_generators.norm_norm as gen_norm_norm
from tests import shared
from tests.shared import expected_voi_t
from value_of_information.simulation import SimulationExecutor
from value_of_information.voi import solve_threshold_b, value_of_information


@pytest.mark.parametrize('simulation_inputs',
						 argvalues=gen_lognorm_norm.linsp(4) + gen_norm_norm.linsp(6),
						 ids=shared.simulation_input_idfn)
def test_integral(simulation_inputs):
	"""
	Check that integrating over the expression returns the same as the simulation mean.
	"""
	threshold_b = solve_threshold_b(simulation_inputs.prior_T, simulation_inputs.sd_B, simulation_inputs.bar)

	voi = lambda t: expected_voi_t(t,
								   threshold_b=threshold_b,
								   sd_B=simulation_inputs.sd_B,
								   bar=simulation_inputs.bar,
								   prior_ev=simulation_inputs.prior_T_ev)

	f_to_integrate = lambda t: simulation_inputs.prior_T.pdf(t) * voi(t)
	left, right = simulation_inputs.prior_T.support()

	voi_integral = integrate.quad(f_to_integrate, a=left, b=right)[0]

	voi_simulation = SimulationExecutor(simulation_inputs).execute(iterations=2_000_000).mean_voi()

	assert voi_simulation == pytest.approx(voi_integral, rel=5 / 100)


@pytest.mark.parametrize('params',
						 gen_norm_norm.linsp(6) + gen_lognorm_norm.linsp(6) + gen_norm_norm.from_seed(5),
						 ids=shared.simulation_input_idfn)
def test(params):
	sd_B = params.sd_B
	bar = params.bar
	prior_T = params.prior_T

	prior_T_ev = prior_T.expect()
	threshold_b = solve_threshold_b(prior_T, sd_B, bar)

	for T in np.linspace(prior_T.ppf(0.01), prior_T.ppf(0.99)):
		voi_from_expression = expected_voi_t(T, threshold_b, sd_B, bar, prior_T_ev)
		voi_from_program = value_of_information(T, sd_B, bar, prior_T, prior_T_ev, threshold_b=threshold_b)["E_B[VOI]"]
		assert voi_from_program == pytest.approx(voi_from_expression)
