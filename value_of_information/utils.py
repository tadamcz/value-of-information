import textwrap

import numpy as np


def is_increasing(array, rtol=0, atol=0):
	prev = -float("inf")
	for element in array:
		if np.isclose(element, prev, rtol, atol):
			continue
		elif element <= prev:
			return False

		prev = element

	return True


def get_lognormal_moments(mu, sigma):  # todo add tests
	var = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
	sd = np.sqrt(var)
	expect = np.exp(mu + sigma ** 2 / 2)

	return expect, sd


def print_wrapped(string, width=120, replace_whitespace=False):
	array = textwrap.wrap(string, width, replace_whitespace=replace_whitespace, drop_whitespace=False)
	print("\n".join(array))
