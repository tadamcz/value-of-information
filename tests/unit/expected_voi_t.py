import numpy as np
import pytest

import tests.param_generators.lognorm_norm as gen_lgn_n
import tests.param_generators.metalog_norm as gen_mlog
import tests.param_generators.norm_norm as gen_n_n
from tests import shared
from value_of_information.voi import solve_threshold_b, value_of_information


@pytest.mark.parametrize('params', gen_n_n.linsp(6) + gen_lgn_n.linsp(6) + gen_n_n.from_seed(5) + gen_mlog.gen(),
						 ids=shared.sim_param_idfn)
def test(params):
	"""
	Based on the direct simplified expression for `VOI(t) = E_B[VOI(T,B) | T=t]` (see README and shared.py).
	"""
	sd_B = params.sd_B
	bar = params.bar
	prior_T = params.prior_T

	prior_T_ev = prior_T.expect()
	threshold_b = solve_threshold_b(prior_T, sd_B, bar)

	for T in np.linspace(prior_T.ppf(0.01), prior_T.ppf(0.99)):
		voi_from_expression = shared.expected_voi_t(T, threshold_b, sd_B, bar, prior_T_ev)
		voi_from_program = value_of_information(T, sd_B, bar, prior_T, prior_T_ev, threshold_b=threshold_b)["E_B[VOI]"]
		assert voi_from_program == pytest.approx(voi_from_expression)
