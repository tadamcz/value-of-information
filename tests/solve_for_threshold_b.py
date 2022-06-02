import pytest
from scipy import stats

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


def test_solve_for_threshold_b(prior_mu, prior_sigma, sd_B, bar):
	"""
	Closed-form expression for the threshold value `b_*`. See README.
	"""

	closed_form = (sd_B ** 2 * (bar - prior_mu)) / prior_sigma ** 2 + bar

	prior = stats.norm(loc=prior_mu, scale=prior_sigma)

	assert voi.threshold_b(prior, sd_B, bar) == pytest.approx(closed_form, rel=1 / 100_000)
