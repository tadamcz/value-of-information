import pytest
from scipy import integrate

from tests import shared
from tests.param_generators import lognorm_norm as gen_lognorm_norm, norm_norm as gen_norm_norm
from value_of_information.simulation import SimulationExecutor
from value_of_information.voi import solve_threshold_b


@pytest.mark.parametrize('simulation_inputs',
						 argvalues=gen_lognorm_norm.linsp(4) + gen_norm_norm.linsp(6),
						 ids=shared.simulation_input_idfn)
def test_integral(simulation_inputs):
	"""
	Based on the direct simplified expression for `VOI(t) = E_B[VOI(T,B) | T=t]` (see README and shared.py).

	Check that integrating over the expression returns the same as the simulation mean.
	"""
	threshold_b = solve_threshold_b(simulation_inputs.prior_T, simulation_inputs.sd_B, simulation_inputs.bar)

	voi = lambda t: shared.expected_voi_t(t,
								   threshold_b=threshold_b,
								   sd_B=simulation_inputs.sd_B,
								   bar=simulation_inputs.bar,
								   prior_ev=simulation_inputs.prior_T_ev)

	f_to_integrate = lambda t: simulation_inputs.prior_T.pdf(t) * voi(t)
	left, right = simulation_inputs.prior_T.support()

	voi_integral = integrate.quad(f_to_integrate, a=left, b=right)[0]

	voi_simulation = SimulationExecutor(simulation_inputs).execute(iterations=2_000_000).mean_voi()

	assert voi_simulation == pytest.approx(voi_integral, rel=5 / 100)