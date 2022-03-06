import numpy as np
from scipy import stats

from simulation import Simulation

prior_mu, prior_sigma = 1, 1
prior = stats.lognorm(scale=np.exp(prior_mu), s=prior_sigma)
study_sample_size = 100
population_std_dev = 20


def test_extreme_high_bar():
	"""
	If both prior expected value and all values of posterior are less than the bar,
	the study value is 0.
	"""
	bar = 1e9
	simulation = Simulation(
		prior=prior,
		study_sample_size=study_sample_size,
		population_std_dev=population_std_dev,
		bar=bar)
	assert simulation.run(max_runs=500) == 0

def test_extreme_low_bar():
	"""
	If both prior expected value and all values of posterior are greater than the bar,
	the study value is 0.
	"""
	bar = -1e9
	simulation = Simulation(
		prior=prior,
		study_sample_size=study_sample_size,
		population_std_dev=population_std_dev,
		bar=bar)
	assert simulation.run(max_runs=500) == 0
