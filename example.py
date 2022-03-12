import numpy as np
from scipy import stats

from simulation import Simulation

prior_mu, prior_sigma = 1, 1
prior = stats.lognorm(scale=np.exp(prior_mu), s=prior_sigma)
study_sample_size = 100
population_std_dev = 20
bar = 5
simulation = Simulation(
	prior=prior,
	study_sample_size=study_sample_size,
	population_std_dev=population_std_dev,
	bar=bar)
simulation.run(max_iterations=100)

