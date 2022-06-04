import numpy as np
import pytest
from scipy import stats

import tests.param_generators.lognorm_norm as gen_lgn_n
import tests.param_generators.norm_norm as gen_n_n
import tests.shared as shared
from value_of_information.voi import value_of_information


def helper(T, sd_B, bar, prior_T, prior_T_ev, b_i_draws):

	threshold_b = shared.patched_threshold_b(prior_T, sd_B, bar)

	b_draw_no = value_of_information(T, sd_B, bar, prior_T, prior_T_ev, b=None, threshold_b=threshold_b)[
		"E_B[VOI]"]
	b_draw_yes = []
	for b in b_i_draws:
		voi = value_of_information(T, sd_B, bar, prior_T, prior_T_ev, b=b, threshold_b=threshold_b)["VOI"]
		b_draw_yes.append(voi)

	b_draw_yes = np.mean(b_draw_yes)
	assert b_draw_no == pytest.approx(b_draw_yes, rel=5 / 100)


@pytest.mark.parametrize('params',
						 argvalues=gen_lgn_n.linsp(4) + gen_n_n.linsp(6),
						 ids=shared.sim_param_idfn)
def test(params, random_seed):
	"""
	Compare mean VOI with or without explicit b_i draws.
	"""
	sd_B = params.sd_B
	bar = params.bar
	prior_T = params.prior_T
	prior_T_ev = params.prior_T_ev

	n_bi = 500_000

	# Outside the T-loop below for efficiency
	b_i_distance_draws = stats.norm(0, sd_B).rvs(n_bi)

	for T in np.linspace(prior_T.ppf(0.95), prior_T.ppf(0.95), num=3):
		b_i_draws = T + b_i_distance_draws
		helper(T, sd_B, bar, prior_T, prior_T_ev, b_i_draws)
