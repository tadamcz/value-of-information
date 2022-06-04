from unittest.mock import patch

import numpy as np
from scipy import stats
from scipy.stats import kstest

from tests import shared
from value_of_information.simulation import SimulationInputs, SimulationExecutor, SimulationRun


def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]


def test(random_seed):
	"""
	Run lots of simulations, and check that the distribution of mean VOIs across simulations is approximately normal
	with the expected mean and variance (using the Kolmogorov-Smirnov statistic).
	"""
	inputs = SimulationInputs(
		prior=stats.norm(1, 10),
		sd_B=1,
		bar=2)
	executor = SimulationExecutor(inputs)

	with patch('value_of_information.bayes.posterior') as patched_posterior:
		patched_posterior.side_effect = shared.normal_normal_closed_form

		# Draw samples all at once, then split them (for efficiency)
		big_simulation = executor.execute(iterations=5_000_000)

	simulation_runs = []
	for data in chunks(big_simulation.iterations_data, 5_000):
		simulation_run = SimulationRun(inputs, executor)
		simulation_run.iterations_data = data
		simulation_runs.append(simulation_run)

	means = []
	for run in simulation_runs:
		mean = run.mean_voi()
		means.append(mean)

	big_mean = big_simulation.mean_voi()
	claimed_standard_err = simulation_runs[0].standard_error_mean_voi()

	theoretical_cdf = stats.norm(big_mean, claimed_standard_err).cdf

	ks_statistic = kstest(means, theoretical_cdf).statistic
	assert ks_statistic < 4 / 100
