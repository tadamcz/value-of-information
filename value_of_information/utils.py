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