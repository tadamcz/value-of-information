from unittest.mock import patch

import pytest
from scipy import stats

from tests import shared
from value_of_information import voi


@pytest.fixture(params=[-1, 0, 2.345], ids=lambda p: f"mu={p}")
def prior_mu(request):
	return request.param


@pytest.fixture(params=[1, 2, 3], ids=lambda p: f"sigma={p}")
def prior_sigma(request):
	return request.param


@pytest.fixture(params=[1, 2, 3], ids=lambda p: f"sd(B)={p}")
def sd_B(request):
	return request.param


@pytest.fixture(params=[-1, 0, 0.5], ids=lambda p: f"bar={p}")
def bar(request):
	return request.param


def test_closed_form(prior_mu, prior_sigma, sd_B, bar):
	"""
	Closed-form expression for the threshold value `b_*`. See README.
	"""

	closed_form = (sd_B ** 2 * (bar - prior_mu)) / prior_sigma ** 2 + bar

	prior = stats.norm(loc=prior_mu, scale=prior_sigma)

	assert voi.solve_threshold_b(prior, sd_B, bar) == pytest.approx(closed_form, rel=1 / 100_000)


def test_infinite_precision(prior_mu, prior_sigma, bar):
	"""
	If the signal is infinitely precise, then we can ignore the prior: we choose the object of study iff b>bar
	"""
	sd_B = 1e-12

	prior = stats.norm(loc=prior_mu, scale=prior_sigma)
	with patch('value_of_information.bayes.posterior') as patched_posterior:
		patched_posterior.side_effect = shared.normal_normal_closed_form
		solution = voi.solve_threshold_b(prior, sd_B, bar)
	assert solution == pytest.approx(bar, rel=1/1000)


