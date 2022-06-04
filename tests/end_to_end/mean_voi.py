import pytest
from scipy import integrate

from tests import shared
from tests.param_generators import lognorm_norm as gen_lgn_n, norm_norm as gen_n_n, metalog_norm as gen_mlog
from value_of_information.simulation import SimulationExecutor
from value_of_information.voi import solve_threshold_b


@pytest.mark.parametrize('params', gen_lgn_n.linsp(4) + gen_n_n.linsp(6) + gen_mlog.gen(), ids=shared.sim_param_idfn)
def test_integral(params, random_seed):
	"""
	Based on the direct simplified expression for `VOI(t) = E_B[VOI(T,B) | T=t]` (see README and shared.py).

	Check that integrating over the expression returns the same as the simulation mean.
	"""
	threshold_b = solve_threshold_b(params.prior_T, params.sd_B, params.bar)

	voi = lambda t: shared.expected_voi_t(t,
										  threshold_b=threshold_b,
										  sd_B=params.sd_B,
										  bar=params.bar,
										  prior_ev=params.prior_T_ev)

	f_to_integrate = lambda t: params.prior_T.pdf(t) * voi(t)
	left, right = params.prior_T.support()

	voi_integral = integrate.quad(f_to_integrate, a=left, b=right)[0]

	voi_simulation = SimulationExecutor(params).execute(iterations=2_000_000).mean_voi()

	assert voi_simulation == pytest.approx(voi_integral, rel=5 / 100)
