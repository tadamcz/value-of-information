import numpy as np
from scipy.stats import stats


def is_increasing(array, rtol=0, atol=0):
	prev = -float("inf")
	for element in array:
		if np.isclose(element, prev, rtol, atol):
			continue
		elif element <= prev:
			return False

		prev = element

	return True


def get_lognormal_moments(mu, sigma):
	var = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
	sd = np.sqrt(var)
	expect = np.exp(mu + sigma ** 2 / 2)

	return expect, sd