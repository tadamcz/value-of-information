import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from bayes_continuous.likelihood_func import NormalLikelihood, LikelihoodFunction
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

		print(self)

	def __repr__(self):
		information = {
			"Bar": self.bar,
			"Prior EV": self.prior_ev,
			"Study sample size": self.study_sample_size,
			"sd(B)": self.sd_B,
		}
		return pd.DataFrame([information]).to_string(index=False)

	def run(self, max_iterations=1000, convergence_target=0.1, iterations=None):
		"""
		If `iterations` is set, it will be honored. Otherwise, the simulations stops
		after `standard_error_of_mean < convergence_target*mean`,
		or after `max_iterations`, whichever comes first.
		"""
		if iterations is None:
			max_iterations = max_iterations  # no need for self. here (or with self.this run)
			convergence_target = convergence_target
		else:
			max_iterations = iterations
			convergence_target = 0  # Can never be reached

		this_run = SimulationRun(self)

		# For each iteration i of the simulation, we draw a true value T_i from the prior.
		# For efficiency, it's better to do this outside the loop
		# See: https://github.com/scipy/scipy/issues/9394
		T_is = self.prior_T.rvs(size=max_iterations)

		i = 0
		while i < max_iterations:
			T_i = T_is[i]

			# Our study has the point estimator B_i for the parameter T_i.
			# sd(B_i) is a constant
			sd_B_i = self.sd_B

			# We draw an estimate b_i from Normal(T_i,sd(B_i)).
			b_i = stats.norm(T_i, sd_B_i).rvs()

			likelihood = NormalLikelihood(b_i, sd_B_i)

			posterior = self.posterior(self.prior_T, likelihood)

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

			iteration_output = {
				'T_i': T_i,

				'posterior_ev': posterior_ev,
				'p_beat_bar': 1 - posterior.cdf(self.bar),

				'w_study': decision_w_study,
				'w_out_study': decision_w_out_study,

				'value_w_study': value_w_study,
				'value_w_out_study': value_w_out_study,

				'value_of_study': value_of_study,
			}
			iteration_output = pd.DataFrame([iteration_output])
			if this_run.iterations_data is None:
				this_run.iterations_data = iteration_output
			else:
				this_run.iterations_data = pd.concat([this_run.iterations_data, iteration_output], ignore_index=True)

			std_err = this_run.iterations_data['value_of_study'].sem()
			mean = this_run.iterations_data['value_of_study'].mean()
			if std_err < convergence_target * mean:
				this_run.print_intermediate()
				print(f"Converged after {len(this_run.iterations_data)} iterations!")
				break
			if len(this_run.iterations_data) % 10 == 0:
				this_run.print_intermediate()

			i += 1
		else:
			print(
				f"Did not converge after {len(this_run.iterations_data)} iterations. Standard error of mean study value: {this_run.iterations_data['value_of_study'].sem().round(2)}")

		this_run.print_final()

		return this_run

	def posterior(self, prior: rv_frozen, likelihood: LikelihoodFunction):
		return Posterior(prior, likelihood)


class SimulationRun:
	def __init__(self, simulation: Simulation):
		self.simulation = simulation
		self.iterations_data = None

	def print_intermediate(self):
		if self.iterations_data is None:
			raise ValueError
		iteration_number = len(self.iterations_data)
		std_err = self.iterations_data['value_of_study'].sem()
		mean = self.iterations_data['value_of_study'].mean()
		information = {
			'Iteration of simulation': iteration_number,
			"Mean study value": round(mean, 2),
			"Standard error of mean": round(std_err, 2),
		}
		df = pd.DataFrame([information])
		print(df)

	def print_final(self):
		if self.iterations_data is None:
			raise ValueError

		# Once the display.max_rows is exceeded, the display.min_rows options determines how many rows are shown in
		# the truncated repr.
		with pd.option_context('display.max_columns', None, 'display.max_rows', 50, 'display.min_rows', 50,
							   'display.width', None):
			print(self.iterations_data)

		mean_value_of_study = self.iterations_data['value_of_study'].mean()
		sem_of_study = self.iterations_data['value_of_study'].sem()
		iterations = len(self.iterations_data)

		if mean_value_of_study < 0:
			warnings.warn(f"Value of study is negative with {iterations} iterations. Try more iterations?")

		information = {
			"Mean of posterior expected values across draws": self.iterations_data['posterior_ev'].mean(),
			"Fraction of posterior means > bar": (self.iterations_data[
													  'posterior_ev'] > self.simulation.bar).sum() / iterations,
			"Mean value of study": mean_value_of_study,
			"Standard error of mean value of study": sem_of_study,
		}
		quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
		for q in quantiles:
			key = f"Quantile {q} value of study"
			information[key] = self.iterations_data['value_of_study'].quantile(q)

		df = pd.DataFrame([information]).T
		print(df)

	def mean_value_study(self):
		return self.iterations_data['value_of_study'].mean()