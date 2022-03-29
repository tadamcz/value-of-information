import numpy as np
from scipy import stats
from metalogistic import MetaLogistic

from simulation import Simulation

prior_ps = [.15, .5, .9]
prior_xs = [-20, 5, 50]
prior = MetaLogistic(cdf_ps=prior_ps, cdf_xs=prior_xs)
study_sample_size = 100
population_std_dev = 20
bar = 5
simulation = Simulation(
	prior=prior,
	study_sample_size=study_sample_size,
	population_std_dev=population_std_dev,
	bar=bar)
simulation.run(max_iterations=1000)

