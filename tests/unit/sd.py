import numpy as np
import pytest
from scipy import stats

import tests.param_generators.lognorm_norm as gen_log_norm_norm
import tests.param_generators.norm_norm as gen_n_n
from tests import shared
from tests.shared import get_location_scale, is_decreasing, is_increasing, patched_threshold_b
from value_of_information.voi import value_of_information


class Test_sdB:
	"""
	when prior_T_ev<bar (i.e. when without the signal, we would choose the bar) then E_B[VOI(t)] is decreasing in sd(B).
	"""

	def helper(self, T, bar, prior_T, prior_T_ev, central_sd_B, num_sds):
		vois = []
		list_sd_Bs = np.linspace(central_sd_B * 0.5, central_sd_B * 1.5, num=num_sds)
		for sd_B in list_sd_Bs:
			threshold_b = patched_threshold_b(prior_T, sd_B, bar)
			voi = value_of_information(T, sd_B, bar, prior_T, prior_T_ev, threshold_b=threshold_b)["E_B[VOI]"]
			vois.append(voi)
		return vois

	@pytest.mark.parametrize('params', gen_log_norm_norm.linsp(4) + gen_n_n.linsp(6), ids=shared.sim_param_idfn)
	def test(self, params):
		central_sd_B = params.sd_B
		prior_T = params.prior_T
		prior_T_ev = params.prior_T_ev

		bar = prior_T_ev + 1
		for T in np.linspace(prior_T.ppf(0.95), prior_T.ppf(0.95), num=10):
			vois = self.helper(T, bar, prior_T, prior_T_ev, central_sd_B, num_sds=2)
			assert is_decreasing(vois)


class Test_sd_prior_T:
	"""
	when prior_T_ev<bar (i.e. when without the signal, we would choose the bar), then
	the value of the signal is increasing in sd(prior_T) for a normal prior.
	"""

	def helper(self, T, bar, prior_T_ev, central_prior_sd, sd_B, num_sds):
		vois = []
		list_prior_sds = np.linspace(central_prior_sd * 0.5, central_prior_sd * 1.5, num=num_sds)
		priors = [stats.norm(prior_T_ev, sd) for sd in list_prior_sds]
		for prior_T in priors:
			threshold_b = patched_threshold_b(prior_T, sd_B, bar)
			voi = value_of_information(T, sd_B, bar, prior_T, prior_T_ev, threshold_b=threshold_b)["E_B[VOI]"]
			vois.append(voi)
		return vois

	@pytest.mark.parametrize('params', gen_log_norm_norm.linsp(4) + gen_n_n.linsp(6), ids=shared.sim_param_idfn)
	def test(self, params):
		central_prior_T = params.prior_T
		_, central_prior_sd, = get_location_scale(central_prior_T)

		bar = params.prior_T_ev + 1
		prior_T_ev = params.prior_T_ev
		sd_B = params.sd_B

		for T in np.linspace(central_prior_T.ppf(0.95), central_prior_T.ppf(0.95), num=10):
			vois = self.helper(T, bar, prior_T_ev, central_prior_sd, sd_B, num_sds=2)
			assert is_increasing(vois)
