import pytest
from scipy import integrate
from scipy import stats

import tests.param_generators.lognorm_norm as gen_lognorm_norm
import tests.param_generators.norm_norm as gen_norm_norm
from tests import shared
from value_of_information.simulation import SimulationExecutor
from value_of_information.voi import threshold_b


def voi(t, b_threshold, sd_B, bar, prior_ev):
	"""
	= F(b_*) * (bar-t) + t - U(decision_0, t)
	"""
	if prior_ev > bar:
		payoff_no_signal = t
	else:
		payoff_no_signal = bar

	return stats.norm.cdf(b_threshold, loc=t, scale=sd_B) * (bar - t) + t - payoff_no_signal


@pytest.mark.parametrize('simulation_inputs',
						 argvalues=gen_lognorm_norm.linsp(8) + gen_norm_norm.linsp(12),
						 ids=shared.simulation_input_idfn)
def test(simulation_inputs):
	"""
	Test based on integrating over the expression for `VOI(t)` (see README)

	Integration has some advantages over simulation, and could be incorporated into the main program at a later stage.
	"""
	b_threshold = threshold_b(simulation_inputs.prior_T, simulation_inputs.sd_B, simulation_inputs.bar)

	voi_explicit = lambda t: voi(t,
								 b_threshold=b_threshold,
								 sd_B=simulation_inputs.sd_B,
								 bar=simulation_inputs.bar,
								 prior_ev=simulation_inputs.prior_ev)

	f_to_integrate = lambda t: simulation_inputs.prior_T.pdf(t) * voi_explicit(t)
	left, right = simulation_inputs.prior_T.support()

	voi_integral = integrate.quad(f_to_integrate, a=left, b=right)[0]

	voi_simulation = SimulationExecutor(simulation_inputs).execute(iterations=500_000).mean_voi()

	assert voi_simulation == pytest.approx(voi_integral, rel=5 / 100)
