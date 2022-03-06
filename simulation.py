import warnings

import numpy as np
import pandas as pd
from scipy import stats

from bayes_continuous.likelihood_func import NormalLikelihood
from bayes_continuous.posterior import Posterior


class Simulation:
	def __init__(self, prior, study_sample_size, population_std_dev, bar):
		"""
		We can call T the parameter over which we want to conduct inference.
		(It's conventional to use theta, but it's difficult to distinguish uppercase and lowercase
		theta without LaTeX, so I use `T` and `t`).
		"""
		self.prior_T = prior
		self.prior_ev = self.prior_T.expect()
		self.study_sample_size = study_sample_size
		self.population_std_dev = population_std_dev  # Assumed to be known
		self.sd_B = population_std_dev / np.sqrt(study_sample_size)
		self.bar = bar

		print(f"Bar = {self.bar}")
		print(f"Prior EV = {round(self.prior_ev,2)}")
		print(f"sd(B) = {self.sd_B}")

	def run(self, max_runs=1000, convergence_target=0.1):
		"""
		Stops after `standard_error_of_mean < convergence_target*mean`,
		or after `max_runs`.
		"""
		self.max_runs = max_runs
		self.convergence_target = convergence_target
		simulation_runs = None

		# For each run i of the simulation, we draw a true value T_i from the prior.
		# For efficiency, it's better to do this outside the loop
		# See: https://github.com/scipy/scipy/issues/9394
		T_is = self.prior_T.rvs(size=max_runs)

		i = 0
		while i < max_runs:
			T_i = T_is[i]

			# Our study has the point estimator B_i for the parameter T_i.
			# sd(B_i) is a constant
			sd_B_i = self.sd_B

			# We draw an estimate b_i from Normal(T_i,sd(B_i)).
			b_i = stats.norm(T_i, sd_B_i).rvs()

			likelihood = NormalLikelihood(b_i, sd_B_i)

			posterior = Posterior(self.prior_T, likelihood)

			posterior_ev = posterior.expect()

			# Without study
			if self.prior_ev > self.bar:
				decision_w_out_study = "candidate"
				value_w_out_study = T_i
			else:
				decision_w_out_study = "fallback"
				value_w_out_study = self.bar

			# With study
			if posterior_ev > self.bar:
				decision_w_study = "candidate"
				value_w_study = T_i
			else:
				decision_w_study = "fallback"
				value_w_study = self.bar

			value_of_study = value_w_study - value_w_out_study

			simulation_run = {
				'T_i': T_i,

				'posterior_ev': posterior_ev,
				'p_beat_bar': 1 - posterior.cdf(self.bar),

				'w_study': decision_w_study,
				'w_out_study': decision_w_out_study,

				'value_w_study': value_w_study,
				'value_w_out_study': value_w_out_study,

				'value_of_study': value_of_study,
			}
			simulation_run = pd.DataFrame([simulation_run])
			if simulation_runs is None:
				simulation_runs = simulation_run
			else:
				simulation_runs = pd.concat([simulation_runs, simulation_run], ignore_index=True)

			std_err = simulation_runs['value_of_study'].sem()
			mean = simulation_runs['value_of_study'].mean()
			if std_err < self.convergence_target * mean:
				self.print_temporary(simulation_runs)
				print(f"Converged after {len(simulation_runs)} runs!")
				break
			if len(simulation_runs) % 10 == 0:
				self.print_temporary(simulation_runs)

			i += 1
		else:
			print(
				f"Did not converge after {len(simulation_runs)} runs. Standard error of mean study value: {simulation_runs['value_of_study'].sem().round(2)}")

		self.print_final(simulation_runs)

		return simulation_runs['value_of_study'].mean()

	def print_temporary(self, data_frame):
		run_number = len(data_frame)
		std_err = data_frame['value_of_study'].sem()
		mean = data_frame['value_of_study'].mean()
		information = {
			'Run of simulation': run_number,
			"Mean study value": round(mean,2),
			"Standard error of mean": round(std_err,2),
		}
		df = pd.DataFrame([information])
		print(df)

	def print_final(self, data_frame):
		with pd.option_context('display.max_columns', None, 'display.max_rows', None, 'display.width', None):
			print(data_frame)

		mean_value_of_study = data_frame['value_of_study'].mean()
		sem_of_study = data_frame['value_of_study'].sem()
		runs = len(data_frame)

		if mean_value_of_study < 0:
			warnings.warn(f"Value of study is negative with {runs} runs. Try more runs?")

		information = {
			"Mean of posterior expected values across draws": data_frame['posterior_ev'].mean(),
			"Fraction of posterior means > bar": (data_frame['posterior_ev'] > self.bar).sum() / runs,
			"Mean value of study": mean_value_of_study,
			"Standard error of mean value of study": sem_of_study,
		}
		quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
		for q in quantiles:
			key = f"Quantile {q} value of study"
			information[key] = data_frame['value_of_study'].quantile(q)

		df = pd.DataFrame([information]).T
		print(df)
