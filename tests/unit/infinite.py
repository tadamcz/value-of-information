"""
These tests by their nature rely on extreme numbers, so they are not feasible using integration routines.
So all test in this module use a normal prior and calculate the posterior in closed form.
"""
from unittest.mock import patch

import numpy as np
import pytest

import tests.shared as shared
from tests.param_generators import norm_norm as gen_n_n
from value_of_information.voi import value_of_information, solve_threshold_b


class TestInfiniteBar:
	def helper(self, params, bar):
		prior_T = params.prior_T
		prior_T_ev = params.prior_T_ev
		sd_B = params.sd_B

		with patch('value_of_information.bayes.posterior') as patched_posterior:
			patched_posterior.side_effect = shared.normal_normal_closed_form
			threshold_b = solve_threshold_b(prior_T, sd_B, bar)

		for T in np.linspace(prior_T.ppf(0.01), prior_T.ppf(0.99), num=5):
			for b in np.linspace(-5 * sd_B, 5 * sd_B, num=5):
				voi = value_of_information(T, sd_B, bar, prior_T, prior_T_ev, b, threshold_b=threshold_b)
				assert voi["VOI"] == 0

	@pytest.mark.parametrize('params', gen_n_n.from_seed(10), ids=shared.sim_param_idfn)
	def test_high(self, params):
		"""
		If both prior expected value and all values of posterior are less than the bar,
		the signal value is 0.
		"""
		self.helper(params, bar=1e9)

	@pytest.mark.parametrize('params', gen_n_n.from_seed(10), ids=shared.sim_param_idfn)
	def test_low(self, params):
		"""
		If both prior expected value and all values of posterior are greater than the bar,
		the signal value is 0.
		"""
		self.helper(params, bar=-1e9)


@pytest.mark.parametrize('params', gen_n_n.from_seed(10), ids=shared.sim_param_idfn)
def test_infinite_precision(params):
	"""
	The posterior mean E[T|b] (calculated using explicit_bayes=True) is equal to T.
	"""
	sd_B = 1e-9
	bar = params.bar
	prior_T = params.prior_T
	prior_T_ev = params.prior_T_ev

	for T in np.linspace(prior_T.ppf(0.01), prior_T.ppf(0.99), num=5):
		for b in np.linspace(T - 5 * sd_B, T + 5 * sd_B, num=5):
			with patch('value_of_information.bayes.posterior') as patched_posterior:
				patched_posterior.side_effect = shared.normal_normal_closed_form
				voi = value_of_information(T, sd_B, bar, prior_T, prior_T_ev, b=b, explicit_bayes=True)

			assert voi["E[T|b]"] == pytest.approx(T)
