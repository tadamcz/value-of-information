import numpy as np
import pytest

import tests.param_generators.norm_norm as gen_n_n
import tests.shared as shared
from value_of_information.voi import solve_threshold_b
from value_of_information.voi import value_of_information


@pytest.mark.parametrize('params', gen_n_n.linsp_distance_to_bar(2), ids=shared.sim_param_idfn)
def test(params):
	sd_B = params.sd_B
	bar = params.bar
	prior_T = params.prior_T
	prior_T_ev = params.prior_T_ev

	threshold_b = solve_threshold_b(prior_T, sd_B, bar)

	for T in np.linspace(prior_T.ppf(0.01), prior_T.ppf(0.99), num=5):
		for b in np.linspace(T - 5 * sd_B, T + 5 * sd_B, num=5):
			threshold = value_of_information(T, sd_B, bar, prior_T, prior_T_ev, b, threshold_b=threshold_b)
			explicit = value_of_information(T, sd_B, bar, prior_T, prior_T_ev, b, explicit_bayes=True)

			for key in ['w_out_signal', 'payoff_w_out_signal', 'w_signal', 'payoff_w_signal', 'VOI']:
				assert threshold[key] == explicit[key]
